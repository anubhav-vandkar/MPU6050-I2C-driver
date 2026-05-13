#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define VGA_PHYS_BASE 0xff200000u
#define VGA_MAP_LEN   0x10000u

#define VGA_BUF0_WORD 0x0000
#define VGA_CTRL_WORD 0x4000

#define VGA_WIDTH 640
#define VGA_HEIGHT 480
#define WORDS_PER_ROW 16

static volatile uint32_t *vga;

static inline uint32_t pack(int xs, int xe, uint8_t color) {
    return ((uint32_t)color << 24) |
           ((uint32_t)(xe & 0x0FFF) << 12) |
           ((uint32_t)(xs & 0x0FFF));
}

int main() {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    void *base = mmap(NULL, VGA_MAP_LEN,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd,
                      VGA_PHYS_BASE);

    if (base == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    vga = (volatile uint32_t *)base;

    uint32_t red = pack(0, VGA_WIDTH - 1, 0xE0);

    printf("Writing red frame...\n");

    for (int y = 0; y < VGA_HEIGHT; y++) {
        for (int s = 0; s < WORDS_PER_ROW; s++) {

            uint32_t addr =
                VGA_BUF0_WORD +
                y * WORDS_PER_ROW +
                s;

            vga[addr] = red;
        }
    }

    vga[VGA_CTRL_WORD] = 0;

    printf("Done.\n");

    while (1) sleep(1);
}
