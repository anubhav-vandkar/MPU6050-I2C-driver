#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

/*
 * MPU-6050 I2C address.
 * AD0 pin low  -> 0x68 (default, nothing connected to AD0)
 * AD0 pin high -> 0x69
 * Check your breakout board -- most ship with AD0 pulled low.
 */
#define MPU6050_ADDR     0x68
#define I2C_DEV          "/dev/i2c-0"

/*
 * MPU-6050 register map.
 * These are internal register indices, not memory addresses.
 * The master sends one of these as the first byte of a write
 * transaction to point the sensor's internal register pointer.
 */
#define REG_SMPLRT_DIV    0x19  /* sample rate divider */
#define REG_CONFIG        0x1A  /* DLPF (digital low pass filter) config */
#define REG_GYRO_CONFIG   0x1B  /* gyro full-scale range */
#define REG_ACCEL_CONFIG  0x1C  /* accel full-scale range */
#define REG_INT_PIN_CFG   0x37  /* interrupt pin configuration */
#define REG_INT_ENABLE    0x38  /* interrupt enable */
#define REG_INT_STATUS    0x3A  /* interrupt status -- bit 0 = data ready */
#define REG_ACCEL_XOUT_H  0x3B  /* first accel register, burst read starts here */
#define REG_PWR_MGMT_1    0x6B  /* power management -- defaults to 0x40 (sleep) */
#define REG_WHO_AM_I      0x75  /* identity register -- always returns 0x68 */

/*
 * Full-scale range settings.
 * Written to GYRO_CONFIG bits [4:3] and ACCEL_CONFIG bits [4:3].
 *
 * Accel: 0x00 = ±2g   (16384 LSB/g)
 *        0x08 = ±4g   (8192  LSB/g)
 *        0x10 = ±8g   (4096  LSB/g)
 *        0x18 = ±16g  (2048  LSB/g)
 *
 * Gyro:  0x00 = ±250  deg/s (131   LSB/deg/s)
 *        0x08 = ±500  deg/s (65.5  LSB/deg/s)
 *        0x10 = ±1000 deg/s (32.8  LSB/deg/s)
 *        0x18 = ±2000 deg/s (16.4  LSB/deg/s)
 */
#define ACCEL_FSR_2G      0x00
#define GYRO_FSR_250DPS   0x00

/*
 * imu_raw_frame_t
 *
 * Populated after each I2C read. Contains raw int16 values
 * directly from the sensor registers plus metadata fields
 * needed by the Kalman filter.
 *
 * Note: no magnetometer fields -- MPU-6050 has no onboard mag.
 */
typedef struct {
    uint32_t timestamp_us;   /* monotonic timestamp for Kalman dt computation */
    uint16_t sample_count;   /* increments each frame, used to detect dropped frames */
    int16_t  ax, ay, az;     /* raw accelerometer -- divide by 16384.0 for g */
    int16_t  gx, gy, gz;     /* raw gyroscope    -- divide by 131.0 for deg/s */
} imu_raw_frame_t;

/* global file descriptor for the I2C bus */
static int i2c_fd = -1;

/* ─────────────────────────────────────────
 * i2c_write_reg()
 *
 * Single register write transaction:
 * START → addr+W → ACK → reg → ACK → value → ACK → STOP
 *
 * Used exclusively during init to configure sensor registers.
 * ───────────────────────────────────────── */
static int i2c_write_reg(uint8_t addr, uint8_t reg, uint8_t value)
{
    uint8_t buf[2] = { reg, value };

    if (ioctl(i2c_fd, I2C_SLAVE, addr) < 0) {
        perror("I2C_SLAVE ioctl failed");
        return -1;
    }

    if (write(i2c_fd, buf, 2) != 2) {
        perror("I2C write failed");
        return -1;
    }

    return 0;
}

/* ─────────────────────────────────────────
 * i2c_read_regs()
 *
 * Write-then-read with repeated START:
 * START → addr+W → reg → REPEATED START → addr+R → bytes → STOP
 *
 * The write phase sets the internal register pointer.
 * The read phase retrieves 'len' consecutive bytes.
 * The MPU-6050 auto-increments its pointer after each byte,
 * enabling burst reads of contiguous registers in one transaction.
 *
 * I2C_RDWR ioctl is used instead of separate write()/read() calls
 * to avoid a STOP between the two phases, which some sensors reject.
 * ───────────────────────────────────────── */
static int i2c_read_regs(uint8_t addr, uint8_t reg, uint8_t *buf, int len)
{
    struct i2c_msg msgs[2];
    struct i2c_rdwr_ioctl_data data;

    /* message 0: write register address -- no STOP generated after this */
    msgs[0].addr  = addr;
    msgs[0].flags = 0;          /* write */
    msgs[0].len   = 1;
    msgs[0].buf   = &reg;

    /* message 1: read len bytes -- issues REPEATED START before this */
    msgs[1].addr  = addr;
    msgs[1].flags = I2C_M_RD;  /* read */
    msgs[1].len   = len;
    msgs[1].buf   = buf;

    data.msgs  = msgs;
    data.nmsgs = 2;

    if (ioctl(i2c_fd, I2C_RDWR, &data) < 0) {
        perror("I2C_RDWR ioctl failed");
        return -1;
    }

    return 0;
}

/* ─────────────────────────────────────────
 * get_timestamp_us()
 *
 * Monotonic microsecond timestamp.
 * Used to compute dt between frames in the Kalman filter.
 * CLOCK_MONOTONIC never jumps backwards unlike CLOCK_REALTIME.
 * ───────────────────────────────────────── */
static uint32_t get_timestamp_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000000UL + ts.tv_nsec / 1000);
}

/* ─────────────────────────────────────────
 * mpu6050_init()
 *
 * Opens the I2C bus and writes config registers.
 * This is the one-time setup sequence -- run once on boot
 * before the read loop starts.
 *
 * Register write order matters:
 * 1. Wake the sensor first (PWR_MGMT_1)
 * 2. Wait for oscillator
 * 3. Set sample rate, FSR, filters, interrupts
 * ───────────────────────────────────────── */
int mpu6050_init(void)
{
    i2c_fd = open(I2C_DEV, O_RDWR);
    if (i2c_fd < 0) {
        perror("Failed to open I2C device");
        return -1;
    }

    /*
     * Verify sensor identity.
     * WHO_AM_I (0x75) always returns 0x68 on the MPU-6050
     * regardless of the I2C address set by AD0.
     * If this fails your wiring or address is wrong.
     */
    uint8_t who_am_i;
    if (i2c_read_regs(MPU6050_ADDR, REG_WHO_AM_I, &who_am_i, 1) < 0) {
        fprintf(stderr, "Failed to read WHO_AM_I\n");
        return -1;
    }
    if (who_am_i != 0x68) {
        fprintf(stderr, "MPU-6050 not found. WHO_AM_I=0x%02X (expected 0x68)\n",
                who_am_i);
        return -1;
    }
    printf("MPU-6050 found. WHO_AM_I=0x%02X\n", who_am_i);

    /*
     * Wake from sleep.
     * PWR_MGMT_1 resets to 0x40 which sets the SLEEP bit.
     * Writing 0x00 clears SLEEP and selects the internal
     * 8 MHz oscillator. Writing 0x01 selects the gyro PLL
     * as clock source which is more stable -- recommended.
     */
    if (i2c_write_reg(MPU6050_ADDR, REG_PWR_MGMT_1, 0x01) < 0)
        return -1;
    usleep(10000); /* wait 10ms for oscillator to stabilise */

    /*
     * Sample rate = gyro output rate / (1 + SMPLRT_DIV)
     * Gyro output rate = 1 kHz when DLPF is enabled.
     * 0x00 -> 1 kHz / (1 + 0) = 1 kHz
     */
    if (i2c_write_reg(MPU6050_ADDR, REG_SMPLRT_DIV, 0x00) < 0)
        return -1;

    /*
     * Digital low pass filter (DLPF).
     * 0x01 -> accel 184 Hz bandwidth, gyro 188 Hz bandwidth.
     * Reduces high-frequency noise at the cost of slight latency.
     * Also sets gyro output rate to 1 kHz (required for SMPLRT_DIV).
     */
    if (i2c_write_reg(MPU6050_ADDR, REG_CONFIG, 0x01) < 0)
        return -1;

    /*
     * Gyro full-scale range.
     * 0x00 -> ±250 deg/s, sensitivity 131 LSB/deg/s.
     * Bits [4:3] select FSR -- 0x00 means both bits are 0.
     */
    if (i2c_write_reg(MPU6050_ADDR, REG_GYRO_CONFIG, GYRO_FSR_250DPS) < 0)
        return -1;

    /*
     * Accel full-scale range.
     * 0x00 -> ±2g, sensitivity 16384 LSB/g.
     * Bits [4:3] select FSR.
     */
    if (i2c_write_reg(MPU6050_ADDR, REG_ACCEL_CONFIG, ACCEL_FSR_2G) < 0)
        return -1;

    /*
     * Interrupt pin configuration.
     * 0x22 -> INT pin active high, push-pull, cleared on any read,
     *         I2C bypass enabled (not needed here but harmless).
     */
    if (i2c_write_reg(MPU6050_ADDR, REG_INT_PIN_CFG, 0x22) < 0)
        return -1;

    /*
     * Enable data-ready interrupt.
     * 0x01 -> INT pin asserts when a new sample is ready.
     * The HPS I2C controller triggers on this to start the read.
     */
    if (i2c_write_reg(MPU6050_ADDR, REG_INT_ENABLE, 0x01) < 0)
        return -1;

    printf("MPU-6050 initialised at 1 kHz, ±2g, ±250 deg/s\n");
    return 0;
}

/* ─────────────────────────────────────────
 * mpu6050_read_frame()
 *
 * Reads one complete frame from the sensor.
 *
 * Burst read of 14 bytes starting at ACCEL_XOUT_H (0x3B):
 *   Bytes 0-1:   ACCEL_XOUT  H/L
 *   Bytes 2-3:   ACCEL_YOUT  H/L
 *   Bytes 4-5:   ACCEL_ZOUT  H/L
 *   Bytes 6-7:   TEMP_OUT    H/L  (discarded)
 *   Bytes 8-9:   GYRO_XOUT   H/L
 *   Bytes 10-11: GYRO_YOUT   H/L
 *   Bytes 12-13: GYRO_ZOUT   H/L
 *
 * All values are big-endian (high byte first).
 * Cast to int16_t is critical -- raw values are signed and
 * without the cast negative readings will be interpreted as
 * large positive numbers.
 * ───────────────────────────────────────── */
int mpu6050_read_frame(imu_raw_frame_t *frame, uint16_t *sample_count)
{
    uint8_t buf[14];

    /* timestamp before read to minimise latency error */
    frame->timestamp_us = get_timestamp_us();
    frame->sample_count = (*sample_count)++;

    /*
     * Check data-ready bit in INT_STATUS before reading.
     * Bit 0 = DATA_RDY_INT -- asserted when a new sample is available.
     * This is optional if you are using the INT pin interrupt,
     * but useful as a sanity check in polling mode.
     */
    uint8_t int_status;
    if (i2c_read_regs(MPU6050_ADDR, REG_INT_STATUS, &int_status, 1) < 0)
        return -1;

    if (!(int_status & 0x01)) {
        /* no new data yet -- caller should retry */
        return 1;
    }

    /*
     * Burst read 14 bytes from 0x3B.
     * Single I2C transaction with repeated START.
     * MPU-6050 auto-increments register pointer after each byte.
     */
    if (i2c_read_regs(MPU6050_ADDR, REG_ACCEL_XOUT_H, buf, 14) < 0)
        return -1;

    /* combine high and low bytes into signed 16-bit integers */
    frame->ax = (int16_t)((buf[0]  << 8) | buf[1]);
    frame->ay = (int16_t)((buf[2]  << 8) | buf[3]);
    frame->az = (int16_t)((buf[4]  << 8) | buf[5]);
    /* buf[6], buf[7] = temperature -- not needed, skipped */
    frame->gx = (int16_t)((buf[8]  << 8) | buf[9]);
    frame->gy = (int16_t)((buf[10] << 8) | buf[11]);
    frame->gz = (int16_t)((buf[12] << 8) | buf[13]);

    return 0;
}

/* ─────────────────────────────────────────
 * mpu6050_close()
 *
 * Puts sensor back to sleep and closes the I2C file descriptor.
 * Always call this on program exit.
 * ───────────────────────────────────────── */
void mpu6050_close(void)
{
    if (i2c_fd >= 0) {
        /* set SLEEP bit in PWR_MGMT_1 */
        i2c_write_reg(MPU6050_ADDR, REG_PWR_MGMT_1, 0x40);
        close(i2c_fd);
        i2c_fd = -1;
        printf("MPU-6050 closed\n");
    }
}

/* ─────────────────────────────────────────
 * main()
 *
 * Initialises sensor then runs the 1 kHz read loop.
 * In the full project, fixed-point conversion and
 * Avalon-MM bridge writes slot in after mpu6050_read_frame().
 * ───────────────────────────────────────── */
int main(void)
{
    imu_raw_frame_t frame;
    uint16_t sample_count = 0;
    int ret;

    /* 1ms sleep interval for 1 kHz loop */
    struct timespec sleep_time = { 0, 1000000 };

    if (mpu6050_init() < 0)
        return 1;

    printf("Reading at 1 kHz. Ctrl+C to stop.\n");

    while (1) {
        ret = mpu6050_read_frame(&frame, &sample_count);

        if (ret < 0) {
            fprintf(stderr, "Read error at sample %u\n", sample_count);
            break;
        }

        if (ret == 1) {
            /* data not ready yet -- sleep briefly and retry */
            usleep(100);
            continue;
        }

        /*
         * TODO: fixed-point conversion and Avalon-MM bridge write
         *
         * imu_fixed_frame_t fixed;
         * fixed.ax = (int32_t)(frame.ax);   scale as needed
         * ...
         * fixed.data_ready = 1;
         * write_to_avalon_bridge(&fixed);
         */

        printf("t=%-10u cnt=%-5u "
               "ax=%-6d ay=%-6d az=%-6d "
               "gx=%-6d gy=%-6d gz=%-6d\n",
               frame.timestamp_us, frame.sample_count,
               frame.ax, frame.ay, frame.az,
               frame.gx, frame.gy, frame.gz);

        nanosleep(&sleep_time, NULL);
    }

    mpu6050_close();
    return 0;
}