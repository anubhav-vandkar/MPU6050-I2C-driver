#ifndef I2C_HAL_H
#define I2C_HAL_H

#include <stdint.h>

/* Open the I2C bus. Returns 0 on success, -1 on error. */
int  i2c_hal_open(const char *dev_path);

/* Close the I2C bus file descriptor. */
void i2c_hal_close(void);

/*
 * Write a single byte to a register.
 * addr  -- 7-bit I2C device address
 * reg   -- register index to write
 * value -- byte to write
 * Returns 0 on success, -1 on error.
 */
int i2c_write_reg(uint8_t addr, uint8_t reg, uint8_t value);

/*
 * Burst read from a register.
 * Sends a repeated START between the write and read phases so the
 * internal register pointer is set before reading.
 * addr  -- 7-bit I2C device address
 * reg   -- first register to read from
 * buf   -- output buffer, must be at least 'len' bytes
 * len   -- number of bytes to read
 * Returns 0 on success, -1 on error.
 */
int i2c_read_regs(uint8_t addr, uint8_t reg, uint8_t *buf, int len);

#endif /* I2C_HAL_H */
