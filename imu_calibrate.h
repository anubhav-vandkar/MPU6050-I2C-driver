#ifndef IMU_CALIBRATE_H
#define IMU_CALIBRATE_H

#include <stdint.h>
#include "mpu6050.h"

// Number of samples to collect during calibration 
#define CALIB_NUM_SAMPLES   1000

typedef struct {
    int16_t ax, ay, az;    /* accel bias offsets */
    int16_t gx, gy, gz;    /* gyro  bias offsets */
} imu_bias_t;

int mpu6050_calibrate(imu_bias_t *bias);

void mpu6050_apply_bias(imu_raw_frame_t *frame, const imu_bias_t *bias);

void mpu6050_print_bias(const imu_bias_t *bias);

#endif /* IMU_CALIBRATE_H */
