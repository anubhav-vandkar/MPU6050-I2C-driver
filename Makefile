# Makefile for MPU-6050 -> Kalman filter (FPGA) pipeline on DE1-SoC HPS
#
# Cross-compile: make CROSS=arm-linux-gnueabihf-
# Native:        make

CC      = $(CROSS)gcc
CFLAGS  = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

TARGET = imu_reader.o
SRCS   = main.c \
         i2c_hal.c \
         mpu6050.c \
         imu_calibrate.c \
         imu_fixedpoint.c \
         imu_angles.c \
         fpga_avalon.c

OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
