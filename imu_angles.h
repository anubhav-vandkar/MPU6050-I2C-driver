#ifndef IMU_ANGLES_H
#define IMU_ANGLES_H

#include <stdint.h>
#include "mpu6050.h"
#include "imu_fixedpoint.h"

#define Q39_FRAC_BITS   9
#define Q39_SCALE       (1 << Q39_FRAC_BITS)

/*
 * 12-bit signed range: -2048 to +2047
 * Mask to keep only the lower 12 bits after scaling.
 */
#define Q39_MASK        0x0FFF

/* Decode helper: sign-extend 12-bit field then divide by scale */
static inline float q39_to_float(int16_t val){

    int16_t sign_extended = (int16_t)((val << 4)) >> 4;
    return (float)sign_extended / (float)Q39_SCALE;
}

typedef struct {
    uint16_t sample_count;
    int16_t  roll;
    int16_t  pitch;
    int16_t  tilt;
    int16_t  gx;
    int16_t  gy;
    uint8_t  data_ready;
} imu_angle_frame_t;

void imu_compute_angles(const imu_raw_frame_t *raw, imu_angle_frame_t *out);

void imu_angles_print(const imu_angle_frame_t *frame);

#endif /* IMU_ANGLES_H */
