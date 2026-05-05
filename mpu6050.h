#ifndef MPU6050_H
#define MPU6050_H

/*
 * mpu6050.h
 *
 * Register map, configuration constants, data types, and
 * function prototypes for the MPU-6050 6-axis IMU.
 *
 * Assumes the I2C bus has already been opened via i2c_hal_open().
 */

#include <stdint.h>

/* ── Device address ──────────────────────────────────────────── */
/*
 * AD0 pin low  -> 0x68  (default on most breakout boards)
 * AD0 pin high -> 0x69
 */
#define MPU6050_ADDR    0x68
#define I2C_DEV         "/dev/i2c-1"

/* ── Register map ────────────────────────────────────────────── */
#define REG_SMPLRT_DIV   0x19   /* sample rate divider */
#define REG_CONFIG       0x1A   /* DLPF config */
#define REG_GYRO_CONFIG  0x1B   /* gyro full-scale range */
#define REG_ACCEL_CONFIG 0x1C   /* accel full-scale range */
#define REG_INT_PIN_CFG  0x37   /* interrupt pin config */
#define REG_INT_ENABLE   0x38   /* interrupt enable */
#define REG_INT_STATUS   0x3A   /* bit 0 = data ready */
#define REG_ACCEL_XOUT_H 0x3B   /* burst read starts here (14 bytes total) */
#define REG_PWR_MGMT_1   0x6B   /* power management, default 0x40 (sleep) */
#define REG_WHO_AM_I     0x75   /* always returns 0x68 on MPU-6050 */

/* ── Full-scale range settings ───────────────────────────────── */
/*
 * Accel bits [4:3] in ACCEL_CONFIG:
 *   0x00 -> ±2g    16384 LSB/g
 *   0x08 -> ±4g     8192 LSB/g
 *   0x10 -> ±8g     4096 LSB/g
 *   0x18 -> ±16g    2048 LSB/g
 *
 * Gyro bits [4:3] in GYRO_CONFIG:
 *   0x00 -> ±250  deg/s   131.0 LSB/deg/s
 *   0x08 -> ±500  deg/s    65.5 LSB/deg/s
 *   0x10 -> ±1000 deg/s    32.8 LSB/deg/s
 *   0x18 -> ±2000 deg/s    16.4 LSB/deg/s
 */
#define ACCEL_FSR_2G      0x00
#define GYRO_FSR_250DPS   0x00

/* Sensitivity divisors matching the FSR settings above */
#define ACCEL_SENSITIVITY  16384.0f   /* LSB/g   at ±2g  */
#define GYRO_SENSITIVITY     131.0f   /* LSB/°/s at ±250 °/s */

/* ── Raw data frame ──────────────────────────────────────────── */
/*
 * One sample read from the sensor.
 * Values are the raw 16-bit two's-complement integers directly
 * out of the registers -- no scaling applied yet.
 *
 * To convert:
 *   accel (g)     = ax / ACCEL_SENSITIVITY
 *   gyro  (deg/s) = gx / GYRO_SENSITIVITY
 */
typedef struct {
    uint32_t timestamp_us;   /* monotonic clock, microseconds */
    uint16_t sample_count;   /* frame counter, wraps at 65535 */
    int16_t  ax, ay, az;     /* raw accelerometer */
    int16_t  gx, gy, gz;     /* raw gyroscope */
} imu_raw_frame_t;

/* ── Function prototypes ─────────────────────────────────────── */

/*
 * mpu6050_init()
 * Opens the I2C bus, checks WHO_AM_I, wakes the sensor,
 * and writes all configuration registers.
 * Returns 0 on success, -1 on error.
 */
int  mpu6050_init(void);

/*
 * mpu6050_read_frame()
 * Reads one 14-byte burst from the sensor and populates *frame.
 * Returns  0 -- new frame ready
 *          1 -- data not ready yet, retry
 *         -1 -- I2C error
 */
int  mpu6050_read_frame(imu_raw_frame_t *frame, uint16_t *sample_count);

/*
 * mpu6050_close()
 * Puts sensor back to sleep and closes the I2C bus.
 */
void mpu6050_close(void);

#endif /* MPU6050_H */
