#define _GNU_SOURCE

#include "imu_calibrate.h"
#include "mpu6050.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>

/* Expected accelerometer reading on z-axis when flat (1g at ±2g FSR) */
#define GRAVITY_LSB  16384

int mpu6050_calibrate(imu_bias_t *bias)
{
    imu_raw_frame_t frame;

    /* 32-bit accumulators so we don't overflow during summation */
    int32_t sum_ax = 0, sum_ay = 0, sum_az = 0;
    int32_t sum_gx = 0, sum_gy = 0, sum_gz = 0;

    int collected  = 0;
    int errors     = 0;
    int max_errors = 50;   /* bail if more than 5% of reads fail */

    printf("Calibrating -- keep sensor still and flat...\n");

    while (collected < CALIB_NUM_SAMPLES) {
        int ret = mpu6050_read_frame(&frame);

        if (ret < 0) {
            errors++;
            if (errors >= max_errors) {
                fprintf(stderr, "calibrate: too many I2C errors (%d), aborting\n",
                        errors);
                return -1;
            }
            usleep(500);
            continue;
        }

        if (ret == 1) {
            /* data not ready yet */
            usleep(100);
            continue;
        }

        /* accumulate raw values */
        sum_ax += frame.ax;
        sum_ay += frame.ay;
        sum_az += frame.az;
        sum_gx += frame.gx;
        sum_gy += frame.gy;
        sum_gz += frame.gz;

        collected++;

        /* print a dot every 100 samples so the user sees progress */
        if (collected % 100 == 0) {
            printf(".");
            fflush(stdout);
        }
    }

    printf(" done (%d samples)\n", collected);

    /*
     * Average the accumulated sums.
     *
     * For accel: the bias is the average reading.
     * For az we remove the expected gravity component so that after
     * applying the bias, az reads 0 when still (makes the math cleaner
     * downstream -- the Kalman filter adds gravity back explicitly).
     *
     * If you prefer to keep gravity in az, just remove the
     * "- GRAVITY_LSB" line below.
     */
    bias->ax = (int16_t)(sum_ax / CALIB_NUM_SAMPLES);
    bias->ay = (int16_t)(sum_ay / CALIB_NUM_SAMPLES);
    bias->az = (int16_t)(sum_az / CALIB_NUM_SAMPLES);

    bias->gx = (int16_t)(sum_gx / CALIB_NUM_SAMPLES);
    bias->gy = (int16_t)(sum_gy / CALIB_NUM_SAMPLES);
    bias->gz = (int16_t)(sum_gz / CALIB_NUM_SAMPLES);

    return 0;
}

void mpu6050_apply_bias(imu_raw_frame_t *frame, const imu_bias_t *bias)
{
    frame->ax -= bias->ax;
    frame->ay -= bias->ay;
    frame->az -= bias->az;
    frame->gx -= bias->gx;
    frame->gy -= bias->gy;
    frame->gz -= bias->gz;
}

void mpu6050_print_bias(const imu_bias_t *bias)
{
    printf("--- Calibration offsets (LSB) ---\n");
    printf("  accel:  ax=%-6d  ay=%-6d  az=%-6d\n",
           bias->ax, bias->ay, bias->az);
    printf("  gyro:   gx=%-6d  gy=%-6d  gz=%-6d\n",
           bias->gx, bias->gy, bias->gz);
    printf("---------------------------------\n");
}
