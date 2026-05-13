#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define VGA_PHYS_BASE 0xff200000u
#define VGA_MAP_LEN   0x20000u

#define VGA_BUF0_WORD 0x0000
#define VGA_BUF1_WORD 0x2000
#define VGA_CTRL_WORD 0x4000

static inline uint32_t pack(int xs, int xe, uint8_t c)
{
    return ((uint32_t)c << 24) |
           ((xe & 0x0FFF) << 12) |
           (xs & 0x0FFF);
}

static volatile uint8_t *base8;

static inline void w32(uint32_t word_addr, uint32_t value)
{
    *(volatile uint32_t *)(base8 + word_addr * 4) = value;
}

static inline uint32_t r32(uint32_t word_addr)
{
    return *(volatile uint32_t *)(base8 + word_addr * 4);
}

int main(void)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    void *base = mmap(NULL, VGA_MAP_LEN,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED,
                      fd,
                      VGA_PHYS_BASE);

    if (base == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    close(fd);

    base8 = (volatile uint8_t *)base;

    printf("BASE mapped at %p\n", base);

    // ------------------------------------------------------------
    // TEST 1: single register write/readback (control register)
    // ------------------------------------------------------------
    printf("CTRL before = 0x%x\n", r32(VGA_CTRL_WORD));

    w32(VGA_CTRL_WORD, 0x1);
    printf("CTRL after  = 0x%x\n", r32(VGA_CTRL_WORD));

    // ------------------------------------------------------------
    // TEST 2: write ONE visible pixel segment
    // ------------------------------------------------------------
    uint32_t red = pack(0, 639, 0xE0);

    printf("Writing test segment...\n");
    w32(VGA_BUF0_WORD, red);

    uint32_t check = r32(VGA_BUF0_WORD);
    printf("Readback BUF0[0] = 0x%08x\n", check);

    // ------------------------------------------------------------
    // TEST 3: address scaling probe
    // ------------------------------------------------------------
    printf("Writing pattern across first 8 words...\n");

    for (int i = 0; i < 8; i++) {
        w32(VGA_BUF0_WORD + i, 0xE0E0E0E0);
    }

    printf("Done. Watch VGA screen.\n");

    while (1) sleep(1);
}
