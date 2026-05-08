#ifndef FPGA_AVALON_H
#define FPGA_AVALON_H


#include <stdint.h>
#include "imu_angles.h"

//addresses
#define HPS_TO_FPGA_LW_BASE     0xFF200000UL
#define FPGA_AVALON_BASE        0xFF200000UL   /* change to your Qsys offset */
#define FPGA_AVALON_MAP_SIZE    0x1000         /* 4 KB */

//register offsets
#define REG_ROLL_PITCH      0x00
#define REG_GX_GY           0x04
#define REG_DATA_READY      0x08
#define REG_RESULT_0        0x0C
#define REG_RESULT_1        0x10

typedef struct {
    uint32_t result_0;
    uint32_t result_1;
} kalman_result_t;

#define FPGA_POLL_TIMEOUT_US    50000   /* 50 ms */

int  fpga_avalon_open(void);
int  fpga_avalon_write(const imu_angle_frame_t *frame);
int  fpga_avalon_poll_read(kalman_result_t *result, uint32_t timeout_us);
void fpga_avalon_close(void);

#endif /* FPGA_AVALON_H */
