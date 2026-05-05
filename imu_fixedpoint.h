#ifndef IMU_FIXEDPOINT_H
#define IMU_FIXEDPOINT_H

/*
 * imu_fixedpoint.h
 *
 * Fixed-point conversion layer between raw MPU-6050 LSB values
 * and the Q-format integers expected by the FPGA Kalman filter.
 *
 * ── Format choice rationale ──────────────────────────────────────
 *
 * Both formats are 16-bit signed (int16_t):
 *
 *   Q7.9  -- 1 sign bit, 7 integer bits, 9 fractional bits
 *            Range:      ±127.998  (step ~0.00195)
 *            Used for:   accelerometer output in g
 *                        Max expected: ±2g  → fits with headroom
 *                        Resolution:   1/512 g ≈ 0.002 g  (good)
 *
 *   Q8.8  -- 1 sign bit, 8 integer bits, 8 fractional bits
 *            Range:      ±255.996  (step ~0.00391)
 *            Used for:   gyroscope output in deg/s
 *                        Max expected: ±250 deg/s → just fits
 *                        Resolution:   1/256 deg/s ≈ 0.004 deg/s (ok)
 *
 *   Why not Q7.9 for gyro?
 *     ±250 deg/s exceeds the Q7.9 integer range of ±127. Overflow.
 *     Q8.8 gives ±255 range which covers ±250 deg/s with margin.
 *
 * ── Conversion formula ───────────────────────────────────────────
 *
 *   Q7.9 value = round( physical_value * 2^9 )  = round( val * 512 )
 *   Q8.8 value = round( physical_value * 2^8 )  = round( val * 256 )
 *
 *   Physical values are computed from raw LSBs:
 *     accel (g)     = raw_ax / 16384.0
 *     gyro  (deg/s) = raw_gx / 131.0
 *
 *   Combined (no intermediate float needed in principle, but we use
 *   float here for clarity -- the FPGA does integer-only arithmetic):
 *     accel Q7.9 = round( (raw / 16384.0) * 512  ) = round( raw / 32.0  )
 *     gyro  Q8.8 = round( (raw / 131.0)   * 256  ) = round( raw * 1.954 )
 *
 * ── To decode on the FPGA side ───────────────────────────────────
 *
 *   true_value = fixed_int >> fractional_bits    (integer part only)
 *   true_value = fixed_int / (1 << frac_bits)    (full precision, float)
 */

#include <stdint.h>
#include "mpu6050.h"

/* Fractional bit counts for each format */
#define Q79_FRAC_BITS   9
#define Q88_FRAC_BITS   8

/* Scaling multipliers (precomputed for readability) */
#define Q79_SCALE   (1 << Q79_FRAC_BITS)    /* 512 */
#define Q88_SCALE   (1 << Q88_FRAC_BITS)    /* 256 */

/*
 * imu_fixed_frame_t
 *
 * Fixed-point version of imu_raw_frame_t.
 * Passed to the Avalon-MM bridge and consumed by the Kalman filter.
 *
 * Accel fields: Q7.9  (int16_t, multiply by 1/512 to get g)
 * Gyro  fields: Q8.8  (int16_t, multiply by 1/256 to get deg/s)
 */
typedef struct {
    uint32_t timestamp_us;   /* copied from raw frame unchanged */
    uint16_t sample_count;   /* copied from raw frame unchanged */

    /* accelerometer -- Q7.9 format, units: g */
    int16_t ax_q79;
    int16_t ay_q79;
    int16_t az_q79;

    /* gyroscope -- Q8.8 format, units: deg/s */
    int16_t gx_q88;
    int16_t gy_q88;
    int16_t gz_q88;
} imu_fixed_frame_t;

/* ── Function prototypes ─────────────────────────────────────── */

/*
 * imu_to_fixed()
 *
 * Converts one calibrated raw frame into fixed-point format.
 * The raw frame should already have bias applied before calling this.
 *
 * Internally uses float arithmetic for the conversion step.
 * The FPGA never sees floats -- only the resulting int16 values
 * are written to the Avalon-MM bridge.
 */
void imu_to_fixed(const imu_raw_frame_t *raw, imu_fixed_frame_t *fixed);

/*
 * imu_fixed_print()
 *
 * Debug print of both the fixed-point integers and their decoded
 * floating-point equivalents, so you can sanity check the scaling.
 */
void imu_fixed_print(const imu_fixed_frame_t *fixed);

/*
 * Helper: decode a Q7.9 value back to float (for debug/logging only)
 */
static inline float q79_to_float(int16_t val)
{
    return (float)val / (float)Q79_SCALE;
}

/*
 * Helper: decode a Q8.8 value back to float (for debug/logging only)
 */
static inline float q88_to_float(int16_t val)
{
    return (float)val / (float)Q88_SCALE;
}

#endif /* IMU_FIXEDPOINT_H */
