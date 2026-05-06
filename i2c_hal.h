#ifndef I2C_HAL_H
#define I2C_HAL_H

#include <stdint.h>

int  i2c_hal_open(const char *dev_path);

void i2c_hal_close(void);

int i2c_write_reg(uint8_t addr, uint8_t reg, uint8_t value);

int i2c_read_regs(uint8_t addr, uint8_t reg, uint8_t *buf, int len);

#endif /* I2C_HAL_H */
