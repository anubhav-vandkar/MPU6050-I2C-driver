#ifndef MPU6050_H
#define MPU6050_H

#include <stdint.h>

/*
 * AD0 pin low  -> 0x68  (default on most breakout boards)
 * AD0 pin high -> 0x69
 */
#define MPU6050_ADDR    0x68
#define I2C_DEV         "/dev/i2c-1"

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
#define ACCEL_FSR_2G 0x00
#define GYRO_FSR_250DPS   0x00

/* Sensitivity divisors matching the FSR settings above */
#define ACCEL_SENSITIVITY  16384.0f   /* LSB/g   at ±2g  */
#define GYRO_SENSITIVITY 131.0f   /* LSB/°/s at ±250 °/s */

typedef struct {
    uint32_t timestamp_us;   /* monotonic clock, microseconds */
    uint16_t sample_count;   /* frame counter, wraps at 65535 */
    int16_t  ax, ay, az;     /* raw accelerometer */
    int16_t  gx, gy, gz;     /* raw gyroscope */
} imu_raw_frame_t;

int  mpu6050_init(void);

int  mpu6050_read_frame(imu_raw_frame_t *frame, uint16_t *sample_count);

void mpu6050_close(void);

#endif /* MPU6050_H */
