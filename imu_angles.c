#define _GNU_SOURCE

/*
 * imu_angles.c
 *
 * Converts calibrated raw accelerometer + gyro readings into
 * tilt angles (roll, pitch, tilt) in radians, packed as Q3.9
 * in the lower 12 bits of an int16_t.
 *
 * The upper 4 bits of each angle field are always 0.
 * The FPGA discards them and reads only bits [11:0].
 */

#include "imu_angles.h"
#include "mpu6050.h"

#include <math.h>
#include <stdio.h>

/* ─────────────────────────────────────────
 * float_to_q39()
 *
 * Converts a float angle in radians to Q3.9 packed in 12 bits.
 *
 * Steps:
 *   1. Multiply by 512 (2^9) to shift fractional part into integer
 *   2. Round to nearest integer
 *   3. Clamp to 12-bit signed range [-2048, +2047]
 *   4. Mask to 12 bits and store in int16_t (upper 4 bits = 0)
 * ───────────────────────────────────────── */
static int16_t float_to_q39(float rad)
{
    float scaled = rad * (float)Q39_SCALE;   /* radians * 512 */

    /* clamp to 12-bit signed range before truncating */
    if (scaled >  2047.0f) scaled =  2047.0f;
    if (scaled < -2048.0f) scaled = -2048.0f;

    int16_t quantised = (int16_t)scaled;

    /* mask to 12 bits -- upper nibble is always 0 */
    return (int16_t)(quantised & (int16_t)Q39_MASK);
}

void imu_compute_angles(const imu_raw_frame_t *raw, imu_angle_frame_t *out)
{
    /* convert raw LSB to g */
    float ax = (float)raw->ax / ACCEL_SENSITIVITY;
    float ay = (float)raw->ay / ACCEL_SENSITIVITY;
    float az = (float)raw->az / ACCEL_SENSITIVITY;

    /* convert raw LSB to deg/s (gyro stays in deg/s for Q3.9) */
    float gx = (float)raw->gx / GYRO_SENSITIVITY;
    float gy = (float)raw->gy / GYRO_SENSITIVITY;
    /* gz discarded */

    float roll_rad  = atan2f(ax, sqrtf(ay*ay + az*az));
    float pitch_rad = atan2f(ay, sqrtf(ax*ax + az*az));
    float tilt_rad  = atan2f(sqrtf(ax*ax + ay*ay), az);

    /* pack into output struct */
    out->sample_count = raw->sample_count;
    out->roll = float_to_q39(roll_rad);
    out->pitch = float_to_q39(pitch_rad);
    out->tilt = float_to_q39(tilt_rad);
    out->gx = float_to_q39(gx);
    out->gy = float_to_q39(gy);
    out->data_ready   = 1;
}

void imu_angles_print(const imu_angle_frame_t *f)
{

    // printf("cnt=%-5u | "
    //        "roll=%5d(%7.4f rad) pitch=%5d(%7.4f rad) tilt=%5d(%7.4f rad) | "
    //        "gx=%6d(%7.3f d/s) gy=%6d(%7.3f d/s) | rdy=%u\n",
    //        f->sample_count,
    //        f->roll_q39,  q39_to_float(f->roll_q39),
    //        f->pitch_q39, q39_to_float(f->pitch_q39),
    //        f->tilt_q39,  q39_to_float(f->tilt_q39),
    //        f->gx_q39,    q39_to_float(f->gx_q39),
    //        f->gy_q39,    q39_to_float(f->gy_q39),
    //        f->data_ready);

    printf("%f,"
           "%f,"
           "%f,"
           "%f,"
           "%f\n", 
           q39_to_float(f->roll),
           q39_to_float(f->pitch),
           q39_to_float(f->tilt),
           q39_to_float(f->gx),
           q39_to_float(f->gy));
}
