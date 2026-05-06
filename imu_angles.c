#define _GNU_SOURCE
/*
 * imu_angles.c
 *
 * Converts calibrated raw accelerometer + gyro readings into
 * tilt angles (roll, pitch, tilt) and packs them as Q8.8 int16_t.
 *
 * Uses float internally for the arctan computation.
 * The FPGA only ever sees the final int16_t values.
 */

#include "imu_angles.h"
#include "mpu6050.h"

#include <math.h>
#include <stdio.h>

/* ─────────────────────────────────────────
 * float_to_q88()
 *
 * Converts a float in degrees to Q8.8 fixed-point.
 * Clamps to int16_t range to be safe.
 * ───────────────────────────────────────── */
static int16_t float_to_q88(float deg)
{
    float scaled = deg * (float)Q88_SCALE;   /* deg * 256 */

    if (scaled >  32767.0f) scaled =  32767.0f;
    if (scaled < -32768.0f) scaled = -32768.0f;

    return (int16_t)scaled;
}

/* ─────────────────────────────────────────
 * imu_compute_angles()
 *
 * Formulas (from three-axis tilt measurement):
 *
 *   roll  (ρ) = atan2(Ax,  sqrt(Ay² + Az²))
 *   pitch (φ) = atan2(Ay,  sqrt(Ax² + Az²))
 *   tilt  (θ) = atan2(sqrt(Ax² + Ay²), Az)
 *
 * We use atan2f() instead of atanf() to handle the case where
 * the denominator is zero without a divide-by-zero crash.
 *
 * Inputs are raw LSB values -- we convert to physical g first
 * so the units cancel cleanly inside atan2 (the scale factor
 * divides out anyway, but it is cleaner to be explicit).
 * ───────────────────────────────────────── */
void imu_compute_angles(const imu_raw_frame_t *raw,
                        imu_angle_frame_t     *out)
{
    /* convert raw LSB to g (float) */
    float ax = (float)raw->ax / ACCEL_SENSITIVITY;
    float ay = (float)raw->ay / ACCEL_SENSITIVITY;
    float az = (float)raw->az / ACCEL_SENSITIVITY;

    /* convert raw LSB to deg/s (float) */
    float gx = (float)raw->gx / GYRO_SENSITIVITY;
    float gy = (float)raw->gy / GYRO_SENSITIVITY;
    /* gz discarded */

    /*
     * Compute angles in radians then convert to degrees.
     * atan2f(y, x) gives the angle of the vector (x, y) from
     * the positive x-axis, handling all quadrants correctly.
     *
     *   roll  = atan2(Ax, sqrt(Ay² + Az²))
     *   pitch = atan2(Ay, sqrt(Ax² + Az²))
     *   tilt  = atan2(sqrt(Ax² + Ay²), Az)
     */
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

/* ─────────────────────────────────────────
 * imu_angles_print()
 * ───────────────────────────────────────── */
void imu_angles_print(const imu_angle_frame_t *f)
{
    printf("cnt=%-5u | "
           "roll=%6d(%7.3f°) pitch=%6d(%7.3f°) tilt=%6d(%7.3f°) | "
           "gx=%6d(%7.3f°/s) gy=%6d(%7.3f°/s) | rdy=%u\n",
           f->sample_count,
           f->roll_q88,  q88_to_float(f->roll_q88),
           f->pitch_q88, q88_to_float(f->pitch_q88),
           f->tilt_q88,  q88_to_float(f->tilt_q88),
           f->gx_q88,    q88_to_float(f->gx_q88),
           f->gy_q88,    q88_to_float(f->gy_q88),
           f->data_ready);
}
