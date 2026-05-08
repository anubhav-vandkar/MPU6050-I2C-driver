#define _GNU_SOURCE

#include "mpu6050.h"
#include "i2c_hal.h"

#include <stdio.h>
#include <unistd.h>
#include <time.h>

static uint32_t get_timestamp_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000000UL + ts.tv_nsec / 1000);
}

int mpu6050_init(void)
{
    if (i2c_hal_open(I2C_DEV) < 0)
        return -1;

    //sanity check who_am_i
    uint8_t who_am_i;
    if (i2c_read_regs(MPU6050_ADDR, REG_WHO_AM_I, &who_am_i, 1) < 0) {
        fprintf(stderr, "mpu6050_init: WHO_AM_I read failed\n");
        return -1;
    }
    if (who_am_i != 0x68) {
        fprintf(stderr, "mpu6050_init: unexpected WHO_AM_I 0x%02X (want 0x68)\n",
                who_am_i);
        return -1;
    }
    printf("MPU-6050 found (WHO_AM_I=0x%02X)\n", who_am_i);

    if (i2c_write_reg(MPU6050_ADDR, REG_PWR_MGMT_1, 0x01) < 0)
        return -1;
    usleep(10000); /* 10 ms -- let the PLL lock */

    /*
     * Sample rate divider.
     * Rate = gyro_output_rate / (1 + SMPLRT_DIV)
     * DLPF enabled -> gyro output rate = 1 kHz.
     * 0x00 -> 1 kHz / 1 = 1 kHz
     */
    if (i2c_write_reg(MPU6050_ADDR, REG_SMPLRT_DIV, 0x00) < 0)
        return -1;

    /*
     * Digital low pass filter.
     * 0x01 -> accel BW 184 Hz, gyro BW 188 Hz.
     * Reduces noise; also locks gyro output rate to 1 kHz.
     */
    if (i2c_write_reg(MPU6050_ADDR, REG_CONFIG, 0x01) < 0)
        return -1;

    /* Gyro FSR = ±250 deg/s  (131 LSB/deg/s) */
    if (i2c_write_reg(MPU6050_ADDR, REG_GYRO_CONFIG, GYRO_FSR_250DPS) < 0)
        return -1;

    /* Accel FSR = ±2g  (16384 LSB/g) */
    if (i2c_write_reg(MPU6050_ADDR, REG_ACCEL_CONFIG, ACCEL_FSR_2G) < 0)
        return -1;

    /*
     * Interrupt pin: active high, push-pull, cleared on any read,
     * I2C bypass enabled (harmless if not using aux I2C).
     */
    if (i2c_write_reg(MPU6050_ADDR, REG_INT_PIN_CFG, 0x22) < 0)
        return -1;

    /* Enable data-ready interrupt on INT pin */
    if (i2c_write_reg(MPU6050_ADDR, REG_INT_ENABLE, 0x01) < 0)
        return -1;

    printf("MPU-6050 ready: 1 kHz, ±2g, ±250 deg/s\n");
    return 0;
}

int mpu6050_read_frame(imu_raw_frame_t *frame, uint16_t *sample_count)
{
    uint8_t buf[14];
    uint8_t int_status;

    frame->timestamp_us = get_timestamp_us();
    frame->sample_count = (*sample_count)++;

    // Poll the data-ready bit before reading 
    if (i2c_read_regs(MPU6050_ADDR, REG_INT_STATUS, &int_status, 1) < 0)
        return -1;

    if (!(int_status & 0x01))
        return 1;

    /*
     * Burst read 14 bytes starting at ACCEL_XOUT_H (0x3B).
     * Register layout:
     *   [0-1]   ACCEL_X H/L
     *   [2-3]   ACCEL_Y H/L
     *   [4-5]   ACCEL_Z H/L
     *   [6-7]   TEMP    H/L  (unused)
     *   [8-9]   GYRO_X  H/L
     *   [10-11] GYRO_Y  H/L
     *   [12-13] GYRO_Z  H/L
     * All values big-endian, signed two's complement.
     */
    if (i2c_read_regs(MPU6050_ADDR, REG_ACCEL_XOUT_H, buf, 14) < 0)
        return -1;

    frame->ax = (int16_t)((buf[0]  << 8) | buf[1]);
    frame->ay = (int16_t)((buf[2]  << 8) | buf[3]);
    frame->az = (int16_t)((buf[4]  << 8) | buf[5]);
    frame->gx = (int16_t)((buf[8]  << 8) | buf[9]);
    frame->gy = (int16_t)((buf[10] << 8) | buf[11]);
    frame->gz = (int16_t)((buf[12] << 8) | buf[13]);

    return 0;
}

void mpu6050_close(void)
{
    /* put sensor back to sleep before closing the bus */
    i2c_write_reg(MPU6050_ADDR, REG_PWR_MGMT_1, 0x40);
    i2c_hal_close();
    printf("MPU-6050 closed\n");
}
