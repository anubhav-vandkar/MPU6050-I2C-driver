#define _GNU_SOURCE

/*
 * main.c
 *
 * MPU-6050 -> angle computation -> Kalman filter (FPGA) pipeline.
 *
 * Flow per frame:
 *   1. Read raw IMU frame
 *   2. Apply bias correction
 *   3. Compute roll / pitch angles + gx / gy
 *   4. Write packed registers to Kalman filter Avalon slave
 *   5. Poll REG_DATA_READY until Kalman clears it to 0
 *   6. Read Kalman results into kalman_result
 *   7. TODO: pass kalman_result to VGA computation
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "mpu6050.h"
#include "imu_calibrate.h"
#include "imu_angles.h"
#include "fpga_avalon.h"

int main(void)
{
    imu_raw_frame_t   raw;
    imu_angle_frame_t angles;
    kalman_result_t   kalman_result;
    imu_bias_t        bias;
    uint16_t          sample_count = 0;
    int               ret;

    struct timespec sleep_time = { 0, 1000000 };   /* 1 ms */

    /* -- 1. Init sensor ----------------------------------------- */
    if (mpu6050_init() < 0) {
        fprintf(stderr, "Sensor init failed.\n");
        return 1;
    }

    /* -- 2. Calibrate ------------------------------------------- */
    if (mpu6050_calibrate(&bias) < 0) {
        fprintf(stderr, "Calibration failed.\n");
        mpu6050_close();
        return 1;
    }
    mpu6050_print_bias(&bias);

    /* -- 3. Open Avalon bridge ----------------------------------- */
    if (fpga_avalon_open() < 0) {
        fprintf(stderr, "Avalon bridge open failed.\n");
        mpu6050_close();
        return 1;
    }

    /* -- 4. Read loop ------------------------------------------- */
    printf("Starting read loop. Ctrl+C to stop.\n\n");

    while (1) {
        ret = mpu6050_read_frame(&raw, &sample_count);

        if (ret < 0) {
            fprintf(stderr, "I2C read error\n");
            break;
        }
        if (ret == 1) {
            usleep(100);
            continue;
        }

        /* apply calibration offsets */
        mpu6050_apply_bias(&raw, &bias);

        /* compute angles and pack into struct */
        imu_compute_angles(&raw, &angles);

        /* write roll_pitch and gx_gy to Kalman Avalon registers */
        if (fpga_avalon_write(&angles) < 0) {
            fprintf(stderr, "Avalon write failed\n");
            break;
        }

        /* poll until Kalman clears data_ready, read results back */
        if (fpga_avalon_poll_read(&kalman_result, FPGA_POLL_TIMEOUT_US) < 0) {
            fprintf(stderr, "Kalman timeout -- is the FPGA running?\n");
            break;
        }

        /*
         * kalman_result.result_0 and result_1 hold the Kalman output.
         * TODO: pass to VGA computation, e.g.:
         *   vga_update(&kalman_result);
         */

        /* debug */
        imu_angles_print(&angles);
        printf("  kalman -> r0=0x%08X r1=0x%08X\n",
               kalman_result.result_0, kalman_result.result_1);

        nanosleep(&sleep_time, NULL);
    }

    fpga_avalon_close();
    mpu6050_close();
    return 0;
}
