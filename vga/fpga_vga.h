#define _GNU_SOURCE

#ifndef FPGA_VGA_H
#define FPGA_VGA_H

#include <stdint.h>
#include "../fpga_avalon.h"

#define FPGA_VGA_BASE 0xFF200000UL
#define FPGA_VGA_MAP_SIZE 0x1000

#define VGA_REG_RED 0x00
#define VGA_REG_GREEN 0x04
#define VGA_REG_BLUE 0x08
#define VGA_REG_PITCH_SLOPE 0x0C

int fpga_vga_open(void);

void fpga_vga_set_background(uint8_t r, uint8_t g, uint8_t b);

void fpga_vga_update(kalman_result_t *kalman_result);

void fpga_vga_close(void);

#endif /* FPGA_VGA_H */
