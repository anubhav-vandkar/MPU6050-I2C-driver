#ifndef FPGA_AVALON_H
#define FPGA_AVALON_H

#include <stdint.h>
#include "imu_angles.h"

//addresses
#define FPGA_AVALON_BASE 0xFF200000UL
#define FPGA_AVALON_MAP_SIZE 0x1000

//register offsets
#define REG_ROLL 0x40
#define REG_PITCH 0x44
#define REG_GX 0x48
#define REG_GY 0x4C
#define REG_DATA_READY 0x50
#define REG_DATA_STATUS 0x54
#define REG_RESULT_ROLL 0x58
#define REG_RESULT_PITCH 0x5C

typedef struct {
    uint32_t kalman_roll;
    uint32_t kalman_pitch;
} kalman_result_t;

#define FPGA_POLL_TIMEOUT_US 50000

int fpga_avalon_open(void);

int fpga_avalon_write(const imu_angle_frame_t *frame);

int fpga_avalon_poll_read(kalman_result_t *result, uint32_t timeout_us);

void fpga_avalon_close(void);

void print_kalman_result(const kalman_result_t *res, const imu_angle_frame_t *angles);

#endif /* FPGA_AVALON_H */
