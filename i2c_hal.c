#define _GNU_SOURCE

#include "i2c_hal.h"

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

static int i2c_fd = -1;

int i2c_hal_open(const char *dev_path)
{
    i2c_fd = open(dev_path, O_RDWR);
    if (i2c_fd < 0) {
        perror("i2c_hal_open: failed to open device");
        return -1;
    }
    return 0;
}

void i2c_hal_close(void)
{
    if (i2c_fd >= 0) {
        close(i2c_fd);
        i2c_fd = -1;
    }
}

int i2c_write_reg(uint8_t addr, uint8_t reg, uint8_t value)
{
    uint8_t buf[2] = { reg, value };

    /*
     * Tell the kernel which slave we are addressing.
     * This sets the 7-bit address used for subsequent read()/write() calls.
     */
    if (ioctl(i2c_fd, I2C_SLAVE, addr) < 0) {
        perror("i2c_write_reg: I2C_SLAVE ioctl failed");
        return -1;
    }

    if (write(i2c_fd, buf, 2) != 2) {
        perror("i2c_write_reg: write failed");
        return -1;
    }

    return 0;
}

int i2c_read_regs(uint8_t addr, uint8_t reg, uint8_t *buf, int len)
{
    struct i2c_msg msgs[2];
    struct i2c_rdwr_ioctl_data data;

    //write phase
    msgs[0].addr = addr;
    msgs[0].flags = 0;
    msgs[0].len = 1;
    msgs[0].buf = &reg;

    //read phase
    msgs[1].addr  = addr;
    msgs[1].flags = I2C_M_RD;
    msgs[1].len   = len;
    msgs[1].buf   = buf;

    data.msgs  = msgs;
    data.nmsgs = 2;

    if (ioctl(i2c_fd, I2C_RDWR, &data) < 0) {
        perror("i2c_read_regs: I2C_RDWR ioctl failed");
        return -1;
    }

    return 0;
}
