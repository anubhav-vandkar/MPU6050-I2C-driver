#define _GNU_SOURCE

#include "imu_angles.h"
#include "mpu6050.h"

#include <math.h>
#include <stdio.h>

// const float alpha = 0.01f;
// float roll_filtered = 0.0f;
// float pitch_filtered = 0.0f;

static int16_t float_to_q39(float rad)
{
    float scaled = rad * (float)Q39_SCALE;

    if (scaled >  2047.0f) scaled =  2047.0f;
    if (scaled < -2048.0f) scaled = -2048.0f;

    return (int16_t)((int16_t)scaled & (int16_t)Q39_MASK);
}

void imu_compute_angles(const imu_raw_frame_t *raw, imu_angle_frame_t *out)
{
    float ax = (float)raw->ax / ACCEL_SENSITIVITY;
    float ay = (float)raw->ay / ACCEL_SENSITIVITY;
    float az = (float)raw->az / ACCEL_SENSITIVITY;

    float gx = (float)raw->gx / GYRO_SENSITIVITY;
    float gy = (float)raw->gy / GYRO_SENSITIVITY;

    // float roll_rad  = atan2f(ax, sqrtf(ay*ay + az*az));
    // float pitch_rad = atan2f(ay, sqrtf(ax*ax + az*az));

    float roll_rad  = atan2f(ay, az);
    float pitch_rad = atan2f(ax, sqrtf(ay*ay + az*az));
    /* float tilt_rad = atan2f(sqrtf(ax*ax + ay*ay), az); */

    // roll_filtered  = alpha * roll_rad  + (1.0f - alpha) * roll_filtered;
    // pitch_filtered = alpha * pitch_rad + (1.0f - alpha) * pitch_filtered;

    /* convert to Q-format */
    int16_t roll_q39  = float_to_q39(roll_rad);
    int16_t pitch_q39 = float_to_q39(pitch_rad);
    int16_t gx_q39    = float_to_q39(gx);
    int16_t gy_q39    = float_to_q39(gy);

    out->roll = roll_q39;
    out->pitch = pitch_q39;
    out->gx = gx_q39;
    out->gy = gy_q39;
}

void imu_angles_print(const imu_angle_frame_t *f)
{
    printf("roll=%5d(%7.4f rad) pitch=%5d(%7.4f rad) | "
           "gx=%6d(%7.3f d/s) gy=%6d(%7.3f d/s) | rdy=%u\n",
           f->roll,  q39_to_float(f->roll),
           f->pitch, q39_to_float(f->pitch),
           f->gx,    q39_to_float(f->gx),
           f->gy,    q39_to_float(f->gy),
           f->data_ready);
}
