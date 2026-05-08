#ifndef FPGA_AVALON_H
#define FPGA_AVALON_H

/*
 * fpga_avalon.h
 *
 * Writes imu_angle_frame_t directly into the Kalman filter's
 * Avalon-MM slave registers over the HPS-to-FPGA lightweight bridge.
 *
 * No SRAM component. The kalman_filter.sv IS the Avalon slave.
 *
 * ── Register map ────────────────────────────────────────────────
 *
 *  Offset | Register    | Dir         | Content
 *  -------+-------------+-------------+---------------------------
 *  0x00   | roll_pitch  | HPS writes  | roll_q39[31:16] | pitch_q39[15:0]
 *  0x04   | gx_gy       | HPS writes  | gx_q88[31:16]   | gy_q88[15:0]
 *  0x08   | data_ready  | Kalman clrs | 0 = results ready, poll this
 *  0x0C   | result_0    | Kalman wrs  | first Kalman output word
 *  0x10   | result_1    | Kalman wrs  | second Kalman output word
 *
 * ── Handshake ───────────────────────────────────────────────────
 *
 *  HPS writes roll_pitch then gx_gy. The Avalon write strobe on
 *  the gx_gy transaction (or a dedicated "start" register if you
 *  prefer) triggers the Kalman filter to begin. No data_ready=1
 *  write needed from the HPS -- the bus transaction IS the trigger.
 *
 *  Kalman filter:
 *    - Latches inputs on write strobe
 *    - Runs filter
 *    - Writes results to result_0, result_1
 *    - Clears data_ready register to 0
 *
 *  HPS polls REG_DATA_READY until it reads 0, then reads results.
 *
 * !! Set FPGA_AVALON_BASE to match your Qsys address map !!
 */

#include <stdint.h>
#include "imu_angles.h"

/* ── Addresses ──────────────────────────────────────────────────── */
#define HPS_TO_FPGA_LW_BASE     0xFF200000UL
#define FPGA_AVALON_BASE        0xFF200000UL   /* change to your Qsys offset */
#define FPGA_AVALON_MAP_SIZE    0x1000         /* 4 KB */

/* ── Register offsets (bytes) ───────────────────────────────────── */
#define REG_ROLL_PITCH      0x00
#define REG_GX_GY           0x04
#define REG_DATA_READY      0x08
#define REG_RESULT_0        0x0C
#define REG_RESULT_1        0x10

/* ── Kalman readback struct ─────────────────────────────────────── */
/*
 * Adjust fields to match whatever your kalman_filter.sv
 * writes into result_0 and result_1.
 */
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
