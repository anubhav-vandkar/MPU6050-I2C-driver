/*
 * imu_fixedpoint.c
 *
 * Converts calibrated raw MPU-6050 readings to Q-format fixed-point.
 *
 * Step 1: raw LSB  →  physical float   (divide by sensitivity)
 * Step 2: float    →  Q-format int16   (multiply by 2^frac_bits, round)
 *
 * The two steps can be collapsed into one multiply -- see comments below.
 * We keep them separate here so the intent is easy to follow.
 */

#include "imu_fixedpoint.h"
#include "mpu6050.h"

#include <stdio.h>
#include <math.h>    /* for roundf() */

/*
 * raw_to_q79()
 *
 * Converts a raw accelerometer LSB value to Q7.9.
 *
 * Derivation:
 *   physical_g = raw / ACCEL_SENSITIVITY          (= raw / 16384.0)
 *   q79        = physical_g * 2^9                 (= physical_g * 512)
 *
 * Combined:
 *   q79 = raw * 512 / 16384 = raw / 32.0
 *
 * Clamped to int16_t range to avoid undefined behaviour on overflow.
 * In practice ±2g at ±16384 LSB gives ±512 Q7.9 counts, well within
 * the int16_t range of ±32767.
 */
static int16_t raw_to_q79(int16_t raw)
{
    float physical = (float)raw / ACCEL_SENSITIVITY;       /* g */
    float scaled   = physical * (float)Q79_SCALE;          /* Q7.9 */
    float rounded  = roundf(scaled);

    /* clamp -- should never trigger at ±2g but good practice */
    if (rounded >  32767.0f) rounded =  32767.0f;
    if (rounded < -32768.0f) rounded = -32768.0f;

    return (int16_t)rounded;
}

/*
 * raw_to_q88()
 *
 * Converts a raw gyroscope LSB value to Q8.8.
 *
 * Derivation:
 *   physical_dps = raw / GYRO_SENSITIVITY         (= raw / 131.0)
 *   q88          = physical_dps * 2^8             (= physical_dps * 256)
 *
 * Combined multiplier: 256 / 131 ≈ 1.954
 *
 * At full scale: ±32768 LSB * 1.954 ≈ ±64026, which exceeds int16_t.
 * However the sensor is configured for ±250 deg/s, and at that FSR
 * only ±32768 LSB can be produced, giving ±250 * 256 = ±64000.
 * That DOES overflow int16_t (max ±32767).
 *
 * Resolution: the Kalman filter clamps gyro input to ±127 deg/s
 * (the range actually useful for attitude estimation with this sensor)
 * OR you widen to int32_t downstream. For now we clamp here and note it.
 *
 * If you need ±250 deg/s full range at Q8.8 you have two options:
 *   a) Switch gyro to Q7.1 (7 integer bits, 1 fractional bit). Ugly.
 *   b) Use int32_t for the fixed frame and let the FPGA shift it.
 * We pick (a) only as a last resort -- Q8.8 is fine for ±127 deg/s.
 */
static int16_t raw_to_q88(int16_t raw)
{
    float physical = (float)raw / GYRO_SENSITIVITY;        /* deg/s */
    float scaled   = physical * (float)Q88_SCALE;          /* Q8.8 */
    float rounded  = roundf(scaled);

    /* hard clamp to int16_t range */
    if (rounded >  32767.0f) rounded =  32767.0f;
    if (rounded < -32768.0f) rounded = -32768.0f;

    return (int16_t)rounded;
}

/* ─────────────────────────────────────────
 * imu_to_fixed()
 * ───────────────────────────────────────── */
void imu_to_fixed(const imu_raw_frame_t *raw, imu_fixed_frame_t *fixed)
{
    /* copy metadata unchanged */
    fixed->timestamp_us = raw->timestamp_us;
    fixed->sample_count = raw->sample_count;

    /* accelerometer: Q7.9 */
    fixed->ax_q79 = raw_to_q79(raw->ax);
    fixed->ay_q79 = raw_to_q79(raw->ay);
    fixed->az_q79 = raw_to_q79(raw->az);

    /* gyroscope: Q8.8 */
    fixed->gx_q88 = raw_to_q88(raw->gx);
    fixed->gy_q88 = raw_to_q88(raw->gy);
    fixed->gz_q88 = raw_to_q88(raw->gz);
}

/* ─────────────────────────────────────────
 * imu_fixed_print()
 * ───────────────────────────────────────── */
void imu_fixed_print(const imu_fixed_frame_t *fixed)
{
    /*
     * Print both the raw Q-format integer and the decoded float so
     * it is easy to verify the scaling looks correct.
     */
    printf("t=%-10u cnt=%-5u | "
           "ax=%6d(%.4fg) ay=%6d(%.4fg) az=%6d(%.4fg) | "
           "gx=%6d(%.2f) gy=%6d(%.2f) gz=%6d(%.2f) dps\n",
           fixed->timestamp_us,
           fixed->sample_count,
           fixed->ax_q79, q79_to_float(fixed->ax_q79),
           fixed->ay_q79, q79_to_float(fixed->ay_q79),
           fixed->az_q79, q79_to_float(fixed->az_q79),
           fixed->gx_q88, q88_to_float(fixed->gx_q88),
           fixed->gy_q88, q88_to_float(fixed->gy_q88),
           fixed->gz_q88, q88_to_float(fixed->gz_q88));
}
