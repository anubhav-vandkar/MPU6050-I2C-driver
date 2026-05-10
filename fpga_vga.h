#define _GNU_SOURCE

#ifndef FPGA_VGA_H
#define FPGA_VGA_H

#include <stdint.h>

#define FPGA_VGA_BASE 0xFF200000UL
#define FPGA_VGA_MAP_SIZE 0x1000

#define VGA_REG_RED 0x00
#define VGA_REG_GREEN 0x04
#define VGA_REG_BLUE 0x08
#define VGA_REG_PITCH_SLOPE 0x0C    /* pitch_q39: signed Q3.9 in bits[11:0], slope_q39: signed Q3.9 in bits[15:0] */

int  fpga_vga_open(void);

void fpga_vga_set_background(uint8_t r, uint8_t g, uint8_t b);

void fpga_vga_update(int16_t pitch_q39, int16_t roll_q39);

void fpga_vga_close(void);

#endif /* FPGA_VGA_H */
