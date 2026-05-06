#ifndef IMU_ANGLES_H
#define IMU_ANGLES_H

/*
 * imu_angles.h
 *
 * Computes tilt angles from calibrated accelerometer readings
 * using the three-axis arctan formulas:
 *
 *   roll  (ρ) = arctan( Ax / sqrt(Ay^2 + Az^2) )   -- rotation about X
 *   pitch (φ) = arctan( Ay / sqrt(Ax^2 + Az^2) )   -- rotation about Y
 *   tilt  (θ) = arctan( sqrt(Ax^2 + Ay^2) / Az )   -- inclination from vertical

 * Q8.8 chosen because:
 *   - arctan output range is ±90° (roll/pitch) or 0–180° (tilt)
 *   - Q8.8 covers ±255° with 1/256 deg resolution (~0.004°)
 *   - Fits in int16_t, matches the gyro format already chosen
 */

#include <stdint.h>
#include "mpu6050.h"
#include "imu_fixedpoint.h"

/*
 * All angle fields are Q8.8: divide by 256 to get degrees.
 * All gyro  fields are Q8.8: divide by 256 to get deg/s.
 *
 * Total size: 2 + 2 + 2 + 2 + 2 + 2 + 1 = 13 bytes
 * The compiler will pad to 14 bytes (2-byte alignment on int16_t).
 * Use __attribute__((packed)) if the FPGA side expects no padding.
 */
typedef struct {
    uint16_t sample_count;
    int16_t  roll_q88;
    int16_t  pitch_q88;
    int16_t  tilt_q88;
    int16_t  gx_q88;
    int16_t  gy_q88;
    uint8_t  data_ready;
} imu_angle_frame_t;

void imu_compute_angles(const imu_raw_frame_t *raw, imu_angle_frame_t *out);

void imu_angles_print(const imu_angle_frame_t *frame);

#endif /* IMU_ANGLES_H */
