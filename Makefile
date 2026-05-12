CC      = $(CROSS)gcc
CFLAGS  = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

TARGET = imu_reader.o
SRCS   = main.c \
         i2c_hal.c \
         mpu6050.c \
         imu_calibrate.c \
         imu_angles.c \
         fpga_avalon.c \
         vga/vga_new.c \

OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
