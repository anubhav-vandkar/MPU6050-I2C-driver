#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define FPGA_BASE   0xff200000u
#define MAP_SIZE    0x00040000u   // 256 KB LW window

static volatile uint32_t *fpga = NULL;

static void map_fpga(void)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open(/dev/mem)");
        exit(1);
    }

    fpga = (volatile uint32_t *)mmap(
        NULL,
        MAP_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        FPGA_BASE
    );

    close(fd);

    if (fpga == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    printf("Mapped FPGA LW bridge at 0x%08x\n", FPGA_BASE);
}

int main(void)
{
    map_fpga();

    printf("\nScanning FPGA LW bridge...\n\n");

    int hits = 0;

    for (int i = 0; i < 0x10000; i += 4) {
        volatile uint32_t *addr = fpga + (i / 4);
        uint32_t v1 = *addr;
        uint32_t v2 = *addr;

        // Look for non-zero or unstable registers
        if (v1 != 0 || v2 != 0) {
            printf("Offset 0x%04x : 0x%08x  (readback)\n", i, v1);
            hits++;
        }

        // Also detect “real hardware” behavior (sometimes mirrored registers)
        if (v1 != v2) {
            printf("⚠ unstable read at 0x%04x\n", i);
        }
    }

    if (hits == 0) {
        printf("\nNo FPGA responses detected in LW bridge region\n");
        printf("   -> Linux is NOT seeing your FPGA peripherals\n");
    } else {
        printf("\nFPGA region is alive. Hits: %d\n", hits);
    }

    return 0;
}
