#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

#include "mpu6050.h"
#include "imu_calibrate.h"
#include "imu_angles.h"
#include "fpga_avalon.h"
#include "fpga_vga.h"

static volatile int running = 1;

static void signal_handler(int sig)
{
    (void)sig;
    running = 0;
}

int main(void)
{
    imu_raw_frame_t raw;
    imu_angle_frame_t angles;
    kalman_result_t kalman_result;
    imu_bias_t bias;
    uint16_t sample_count = 0;
    int ret;

    struct timespec sleep_time = { 0, 1000000 };

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    // sensor init
    if (mpu6050_init() < 0) {
        fprintf(stderr, "Sensor init failed.\n");
        return 1;
    }

    // calibrate
    if (mpu6050_calibrate(&bias) < 0) {
        fprintf(stderr, "Calibration failed.\n");
        mpu6050_close();
        return 1;
    }
    mpu6050_print_bias(&bias);

    // avalon init
    if (fpga_avalon_open() < 0) {
        fprintf(stderr, "Avalon bridge open failed.\n");
        mpu6050_close();
        return 1;
    }

    // VGA init
    if (fpga_vga_open() < 0) {
        fprintf(stderr, "VGA open failed.\n");
        fpga_avalon_close();
        mpu6050_close();
        return 1;
    }

    fpga_vga_set_background(0x00, 0x00, 0x80);

    /* -- 5. Read loop ------------------------------------------- */
    printf("Starting. Ctrl+C to stop.\n\n");

    while (running) {
        ret = mpu6050_read_frame(&raw, &sample_count);

        if (ret < 0) {
            fprintf(stderr, "I2C read error\n");
            break;
        }
        if (ret == 1) {
            usleep(100);
            continue;
        }

        mpu6050_apply_bias(&raw, &bias);

        imu_compute_angles(&raw, &angles);

        /* write roll, pitch, gx, gy to Kalman Avalon registers */
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

        int16_t filtered_pitch = (int16_t)(kalman_result.kalman_pitch & 0x0FFF);
        int16_t filtered_roll  = (int16_t)(kalman_result.kalman_roll & 0x0FFF);

        /* sign extend 12-bit values */
        filtered_pitch = (int16_t)(filtered_pitch << 4) >> 4;
        filtered_roll  = (int16_t)(filtered_roll  << 4) >> 4;

        fpga_vga_update(filtered_pitch, filtered_roll);

        imu_angles_print(&angles);

        nanosleep(&sleep_time, NULL);
    }

    fpga_vga_close();
    fpga_avalon_close();
    mpu6050_close();
    return 0;
}
