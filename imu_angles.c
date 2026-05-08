#define _GNU_SOURCE

/*
 * imu_angles.c
 */

#include "imu_angles.h"
#include "mpu6050.h"

#include <math.h>
#include <stdio.h>

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

    float roll_rad  = atan2f(ax, sqrtf(ay*ay + az*az));
    float pitch_rad = atan2f(ay, sqrtf(ax*ax + az*az));
    /* tilt computed but not sent -- Kalman only needs roll + pitch */
    /* float tilt_rad = atan2f(sqrtf(ax*ax + ay*ay), az); */

    /* convert to Q-format */
    int16_t roll_q39  = float_to_q39(roll_rad);
    int16_t pitch_q39 = float_to_q39(pitch_rad);
    int16_t gx_q88    = float_to_q88(gx);
    int16_t gy_q88    = float_to_q88(gy);

    out->roll_pitch = ((int32_t)roll_q39  << 16) | (uint16_t)pitch_q39;
    out->gx_gy      = ((int32_t)gx_q88   << 16) | (uint16_t)gy_q88;
    out->data_ready = 0;   /* only Kalman filter writes this field */
}

void imu_angles_print(const imu_angle_frame_t *f)
{
    int16_t roll  = ROLL_FROM_FRAME(f);
    int16_t pitch = PITCH_FROM_FRAME(f);
    int16_t gx    = GX_FROM_FRAME(f);
    int16_t gy    = GY_FROM_FRAME(f);

    printf("roll_pitch=0x%08X  gx_gy=0x%08X | "
           "roll=%5d(%7.4f rad) pitch=%5d(%7.4f rad) | "
           "gx=%6d(%7.3f d/s) gy=%6d(%7.3f d/s) | rdy=%u\n",
           (uint32_t)f->roll_pitch,
           (uint32_t)f->gx_gy,
           roll,  q39_to_float(roll),
           pitch, q39_to_float(pitch),
           gx,    q39_to_float(gx),
           gy,    q39_to_float(gy),
           f->data_ready);
}
