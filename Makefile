# Makefile for MPU-6050 -> FPGA SRAM pipeline on DE1-SoC HPS
#
# Cross-compile on host:
#   make CROSS=arm-linux-gnueabihf-
# Native compile on HPS:
#   make

CC      = $(CROSS)gcc
CFLAGS  = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

TARGET = imu_reader
SRCS   = main.c \
         i2c_hal.c \
         mpu6050.c \
         imu_calibrate.c \
         imu_fixedpoint.c \
         imu_angles.c \
         fpga_sram.c

OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
