#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define BASE 0xff200000u
#define LEN  0x100000u   // scan 1MB of lightweight bridge space

static volatile uint8_t *base8;

static inline void w32(uint32_t off, uint32_t val)
{
    *(volatile uint32_t *)(base8 + off) = val;
}

static inline uint32_t r32(uint32_t off)
{
    return *(volatile uint32_t *)(base8 + off);
}

int main()
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    void *base = mmap(NULL, LEN,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED,
                      fd,
                      BASE);

    if (base == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    base8 = (volatile uint8_t *)base;
    close(fd);

    printf("Scanning for responsive FPGA regions...\n");

    for (uint32_t off = 0; off < 0x10000; off += 4) {

        uint32_t before = r32(off);

        w32(off, 0xA5A5A5A5);
        uint32_t after = r32(off);

        if (after != before) {
            printf("HIT at offset 0x%08x : %08x -> %08x\n",
                   off, before, after);
        }

        // restore
        w32(off, before);
    }

    printf("Done.\n");

    while (1) sleep(1);
}
