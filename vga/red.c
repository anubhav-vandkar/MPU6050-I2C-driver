#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define VGA_PHYS_BASE 0xff200000u
#define VGA_MAP_LEN   0x20000u   // 128 KB safe

#define VGA_BUF0_WORD 0x0000
#define VGA_CTRL_WORD 0x4000

#define VGA_WIDTH 640
#define VGA_HEIGHT 480
#define WORDS_PER_ROW 16

// RGB332 red = 11100000
#define RED 0xE0

// Convert WORD address -> byte offset
#define WADDR(w) ((w) * 4)

// Safe register access
#define REG32(base8, w) (*(volatile uint32_t *)((base8) + WADDR(w)))

static uint32_t pack_segment(int xs, int xe, uint8_t color)
{
    return ((uint32_t)color << 24) |
           ((uint32_t)(xe & 0x0FFF) << 12) |
           ((uint32_t)(xs & 0x0FFF));
}

int main(void)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open(/dev/mem)");
        return 1;
    }

    void *base = mmap(NULL,
                      VGA_MAP_LEN,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED,
                      fd,
                      VGA_PHYS_BASE);

    if (base == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    close(fd);

    volatile uint8_t *base8 = (volatile uint8_t *)base;

    printf("Mapped base = %p\n", base);

    uint32_t red_word = pack_segment(0, VGA_WIDTH - 1, RED);

    printf("Writing full red frame...\n");

    for (int y = 0; y < VGA_HEIGHT; y++) {
        for (int seg = 0; seg < WORDS_PER_ROW; seg++) {

            uint32_t word_addr =
                VGA_BUF0_WORD +
                y * WORDS_PER_ROW +
                seg;

            REG32(base8, word_addr) = red_word;
        }
    }

    printf("Requesting buffer 0 display...\n");
    REG32(base8, VGA_CTRL_WORD) = 0;

    printf("Done. If VGA is working, screen should be solid red.\n");

    while (1) {
        sleep(1);
    }

    return 0;
}
