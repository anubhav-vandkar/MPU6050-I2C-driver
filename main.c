#define _GNU_SOURCE

/*
 * main.c
 *
 * Entry point for the MPU-6050 data pipeline on the DE1-SoC HPS.
 *
 * Flow:
 *   1. Init sensor (I2C open + register config)
 *   2. Calibrate (collect 1000 still samples, compute bias offsets)
 *   3. Read loop:
 *        a. Read raw frame
 *        b. Apply bias correction
 *        c. Convert to Q-format fixed-point
 *        d. Print / pass to Avalon-MM bridge (TODO)
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "mpu6050.h"
#include "imu_calibrate.h"
#include "imu_fixedpoint.h"

int main(void)
{
    imu_raw_frame_t   raw;
    imu_fixed_frame_t fixed;
    imu_bias_t        bias;
    uint16_t          sample_count = 0;
    int               ret;

    /* 1 ms sleep keeps the polling loop near 1 kHz */
    struct timespec sleep_time = { 0, 1000000 };

    /* ── Init ──────────────────────────────────────────────────── */
    if (mpu6050_init() < 0) {
        fprintf(stderr, "Sensor init failed. Check wiring and I2C address.\n");
        return 1;
    }

    /* ── Calibrate ─────────────────────────────────────────────── */
    /*
     * Keep the sensor perfectly still and flat on a level surface
     * until the calibration routine finishes (about 1 second).
     */
    if (mpu6050_calibrate(&bias) < 0) {
        fprintf(stderr, "Calibration failed. Exiting.\n");
        mpu6050_close();
        return 1;
    }

    mpu6050_print_bias(&bias);

    /* ── Read loop ─────────────────────────────────────────────── */
    printf("Starting read loop. Ctrl+C to stop.\n\n");

    while (1) {
        ret = mpu6050_read_frame(&raw, &sample_count);

        if (ret < 0) {
            fprintf(stderr, "I2C read error at sample %u\n", sample_count);
            break;
        }

        if (ret == 1) {
            /* data not ready -- brief sleep and retry */
            usleep(100);
            continue;
        }

        /* subtract calibration offsets */
        mpu6050_apply_bias(&raw, &bias);

        /* convert to fixed-point */
        imu_to_fixed(&raw, &fixed);

        /*
         * TODO: write fixed frame to Avalon-MM bridge
         *   e.g. avalon_write(&fixed);
         */

        /* debug print */
        imu_fixed_print(&fixed);

        nanosleep(&sleep_time, NULL);
    }

    mpu6050_close();
    return 0;
}
