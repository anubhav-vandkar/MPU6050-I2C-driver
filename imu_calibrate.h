#ifndef IMU_CALIBRATE_H
#define IMU_CALIBRATE_H

/*
 * imu_calibrate.h
 *
 * Static bias calibration for the MPU-6050.
 *
 * Strategy: collect N samples while the sensor is perfectly still,
 * average them, and store the result as an offset to subtract from
 * every subsequent reading.
 *
 * Assumptions:
 *   - Sensor is flat and motionless during calibration.
 *   - az should read ~+16384 (1g upward) when flat; we account for that.
 *   - Gyro should read ~0 on all axes when still.
 *
 * These offsets are in raw LSB units, applied before any scaling.
 */

#include <stdint.h>
#include "mpu6050.h"

/* Number of samples averaged during calibration.
 * More samples = better average, but takes longer.
 * At 1 kHz this is 1 second of data. */
#define CALIB_NUM_SAMPLES   1000

/*
 * imu_bias_t
 *
 * Stores the per-axis bias offsets computed during calibration.
 * All values are in raw int16 LSB units.
 *
 * Apply like this:
 *   corrected_ax = raw_ax - bias.ax
 */
typedef struct {
    int16_t ax, ay, az;    /* accel bias offsets */
    int16_t gx, gy, gz;    /* gyro  bias offsets */
} imu_bias_t;

/*
 * mpu6050_calibrate()
 *
 * Collects CALIB_NUM_SAMPLES frames from the sensor and averages
 * them to compute per-axis bias offsets.
 *
 * The sensor MUST be still and flat (z-axis pointing up) before
 * calling this.  Prints progress dots to stdout.
 *
 * Returns 0 on success, -1 if too many read errors occurred.
 */
int mpu6050_calibrate(imu_bias_t *bias);

/*
 * mpu6050_apply_bias()
 *
 * Subtracts the stored bias from a raw frame in-place.
 * Call this every frame after mpu6050_read_frame().
 */
void mpu6050_apply_bias(imu_raw_frame_t *frame, const imu_bias_t *bias);

/*
 * mpu6050_print_bias()
 *
 * Prints the bias values to stdout -- useful for sanity checking.
 */
void mpu6050_print_bias(const imu_bias_t *bias);

#endif /* IMU_CALIBRATE_H */
