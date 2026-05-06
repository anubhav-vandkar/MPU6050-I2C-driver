#define _GNU_SOURCE
/*
 * MPU-6050 -> angle computation -> FPGA SRAM pipeline on DE1-SoC HPS.
 *
 * Flow per frame:
 *   1. Read raw IMU frame
 *   2. Apply bias correction
 *   3. Compute roll / pitch / tilt angles
 *   4. Write angle frame to FPGA SRAM  (data_ready = 1)
 *   5. Poll SRAM until FPGA clears data_ready to 0
 *      (FPGA runs Kalman filter and writes results back)
 *   6. Read FPGA-modified frame into fpga_result
 *   7. TODO: pass fpga_result to VGA computation
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "mpu6050.h"
#include "imu_calibrate.h"
#include "imu_angles.h"
#include "fpga_sram.h"

#define FPGA_TIMEOUT_US   50000

int main(void)
{
    imu_raw_frame_t   raw;
    imu_angle_frame_t angles;
    imu_angle_frame_t fpga_result;   /* FPGA-modified frame read back from SRAM */
    imu_bias_t        bias;
    uint16_t          sample_count = 0;
    int               ret;

    struct timespec sleep_time = { 0, 1000000 };   /* 1 ms */

    // Init
    if (mpu6050_init() < 0) {
        fprintf(stderr, "Sensor init failed.\n");
        return 1;
    }

    // Calibrate
    if (mpu6050_calibrate(&bias) < 0) {
        fprintf(stderr, "Calibration failed.\n");
        mpu6050_close();
        return 1;
    }
    mpu6050_print_bias(&bias);

    // Open FPGA SRAM bridge
    if (fpga_sram_open() < 0) {
        fprintf(stderr, "FPGA SRAM open failed.\n");
        mpu6050_close();
        return 1;
    }

    // Read loop
    printf("Starting read loop. Ctrl+C to stop.\n\n");

    while (1) {
        ret = mpu6050_read_frame(&raw, &sample_count);

        if (ret < 0) {
            fprintf(stderr, "I2C read error at sample %u\n", sample_count);
            break;
        }
        if (ret == 1) {
            usleep(100);
            continue;
        }

        // apply calibration offsets
        mpu6050_apply_bias(&raw, &bias);

        // compute roll / pitch / tilt, pack into angle struct
        imu_compute_angles(&raw, &angles);

        // write to FPGA SRAM, set data_ready = 1
        if (fpga_sram_write(&angles) < 0) {
            fprintf(stderr, "SRAM write failed at sample %u\n", sample_count);
            break;
        }

        // Polling to send to VGA 
        if (fpga_sram_poll_read(&fpga_result, FPGA_TIMEOUT_US) < 0) {
            fprintf(stderr, "FPGA timeout at sample %u -- is the FPGA running?\n",
                    sample_count);
            break;
        }

        // TODO: send to VGA
        printf("[FPGA] ");
        imu_angles_print(&fpga_result);

        nanosleep(&sleep_time, NULL);
    }

    fpga_sram_close();
    mpu6050_close();
    return 0;
}
