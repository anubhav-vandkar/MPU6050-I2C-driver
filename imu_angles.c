#define _GNU_SOURCE

#include "imu_angles.h"
#include "mpu6050.h"

#include <math.h>
#include <stdio.h>

static int16_t float_to_q88(float deg)
{
    float scaled = deg * (float)Q88_SCALE;   /* deg * 256 */

    if (scaled >  32767.0f) scaled =  32767.0f;
    if (scaled < -32768.0f) scaled = -32768.0f;

    return (int16_t)scaled;
}

void imu_compute_angles(const imu_raw_frame_t *raw, imu_angle_frame_t *out)
{
    /* convert raw LSB to g (float) */
    float ax = (float)raw->ax / ACCEL_SENSITIVITY;
    float ay = (float)raw->ay / ACCEL_SENSITIVITY;
    float az = (float)raw->az / ACCEL_SENSITIVITY;

    /* convert raw LSB to deg/s (float) */
    float gx = (float)raw->gx / GYRO_SENSITIVITY;
    float gy = (float)raw->gy / GYRO_SENSITIVITY;
    /* gz discarded */

    float roll_rad  = atan2f(ax, sqrtf(ay*ay + az*az));
    float pitch_rad = atan2f(ay, sqrtf(ax*ax + az*az));
    float tilt_rad  = atan2f(sqrtf(ax*ax + ay*ay), az);

    float rad_to_deg = 180.0f / (float)M_PI;

    float roll_deg  = roll_rad  * rad_to_deg;
    float pitch_deg = pitch_rad * rad_to_deg;
    float tilt_deg  = tilt_rad  * rad_to_deg;

    /* pack into output struct */
    out->sample_count = raw->sample_count;
    out->roll_q88     = float_to_q88(roll_deg);
    out->pitch_q88    = float_to_q88(pitch_deg);
    out->tilt_q88     = float_to_q88(tilt_deg);
    out->gx_q88       = float_to_q88(gx);
    out->gy_q88       = float_to_q88(gy);
    out->data_ready   = 1;
}

void imu_angles_print(const imu_angle_frame_t *f)
{
    // printf("%-5u,"
    //        "%6d(%7.3f°),"
    //        "%6d(%7.3f°),"
    //        "%6d(%7.3f°),"
    //        "%6d(%7.3f°/s),"
    //        "%6d(%7.3f°/s),"
    //        "%u\n",
    //        f->sample_count,
    //        f->roll_q88,  q88_to_float(f->roll_q88),
    //        f->pitch_q88, q88_to_float(f->pitch_q88),
    //        f->tilt_q88,  q88_to_float(f->tilt_q88),
    //        f->gx_q88,    q88_to_float(f->gx_q88),
    //        f->gy_q88,    q88_to_float(f->gy_q88),
    //        f->data_ready);

    printf("%u,"
           "%d(%f°),"
           "%d(%f°),"
           "%d(%f°),"
           "%d(%f°/s),"
           "%d(%f°/s),"
           "%u\n",
           f->sample_count,
           f->roll_q88,  q88_to_float(f->roll_q88),
           f->pitch_q88, q88_to_float(f->pitch_q88),
           f->tilt_q88,  q88_to_float(f->tilt_q88),
           f->gx_q88,    q88_to_float(f->gx_q88),
           f->gy_q88,    q88_to_float(f->gy_q88),
           f->data_ready);
}
