#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

enum {
    VGA_WIDTH  = 640,
    VGA_HEIGHT = 480,
    WORDS_PER_ROW = 16,

    VGA_BUF0_WORD = 0x0000,
    VGA_CTRL_WORD = 0x4000,
};

#define VGA_PHYS_BASE 0xff200000u
#define VGA_MAP_LEN   0x10000u

static volatile uint32_t *vga = NULL;

static void write_word(int addr, uint32_t value) {
    vga[addr] = value;
}

// pack: [31:24]=color, [23:12]=xstop, [11:0]=xstart
static uint32_t pack(int xs, int xe, uint8_t color) {
    return ((uint32_t)color << 24) |
           ((uint32_t)(xe & 0x0FFF) << 12) |
           ((uint32_t)(xs & 0x0FFF));
}

int main(void) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    vga = mmap(NULL, VGA_MAP_LEN, PROT_READ | PROT_WRITE,
               MAP_SHARED, fd, VGA_PHYS_BASE);

    if (vga == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    close(fd);

    uint8_t RED = 0xE0;

    printf("Filling screen red...\n");

    for (int y = 0; y < VGA_HEIGHT; y++) {
        for (int seg = 0; seg < WORDS_PER_ROW; seg++) {

            // one full-width segment per entry
            uint32_t word = pack(0, VGA_WIDTH - 1, RED);

            vga[VGA_BUF0_WORD + y * WORDS_PER_ROW + seg] = word;
        }
    }

    // request buffer 0 (just in case control works)
    vga[VGA_CTRL_WORD] = 0;

    printf("Done.\n");

    while (1) { usleep(1000000); }
}
