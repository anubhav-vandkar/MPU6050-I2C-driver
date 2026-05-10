#define _GNU_SOURCE

#include "fpga_vga.h"

#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <math.h>

static int mem_fd = -1;
static void *vga_base = NULL;

static inline void vga_reg_write(uint32_t offset, uint32_t value)
{
    volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)vga_base + offset);
    *reg = value;
}

int fpga_vga_open(void)
{
    mem_fd = open("/dev/mem", O_RDWR);
    if (mem_fd < 0) {
        perror("fpga_vga_open: cannot open /dev/mem");
        return -1;
    }

    vga_base = mmap(NULL,
                    FPGA_VGA_MAP_SIZE,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    mem_fd,
                    FPGA_VGA_BASE);

    if (vga_base == MAP_FAILED) {
        perror("fpga_vga_open: mmap failed");
        close(mem_fd);
        mem_fd = -1;
        return -1;
    }

    printf("VGA peripheral mapped: phys=0x%08lX virt=%p\n", (unsigned long)FPGA_VGA_BASE, vga_base);
    return 0;
}

void fpga_vga_set_background(uint8_t r, uint8_t g, uint8_t b)
{
    vga_reg_write(VGA_REG_RED, r);
    vga_reg_write(VGA_REG_GREEN, g);
    vga_reg_write(VGA_REG_BLUE, b);
}

void fpga_vga_update(int16_t pitch_q39, int16_t roll_q39)
{
    uint32_t pitch_word = (uint32_t)((int32_t)pitch_q39 & 0x0FFF);

    float roll_rad  = (float)((int16_t)(roll_q39 << 4) >> 4) / 512.0f;
    float slope_f   = tanf(roll_rad) * 512.0f;

    if (slope_f >  32767.0f) slope_f =  32767.0f;
    if (slope_f < -32768.0f) slope_f = -32768.0f;

    int16_t  slope_q39  = (int16_t)slope_f;
    uint32_t slope_word = (uint32_t)(uint16_t)slope_q39;

    uint32_t combined = ((uint32_t)(uint16_t)slope_q39 << 16)| ((uint32_t)pitch_q39 & 0x0FFF);
    vga_reg_write(VGA_REG_PITCH_SLOPE, combined);
}

void fpga_vga_close(void)
{
    if (vga_base != NULL) {
        munmap(vga_base, FPGA_VGA_MAP_SIZE);
        vga_base = NULL;
    }
    if (mem_fd >= 0) {
        close(mem_fd);
        mem_fd = -1;
    }
    printf("VGA peripheral unmapped\n");
}
