#define _GNU_SOURCE

#include "fpga_avalon.h"
#include "imu_angles.h"

#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

static int    mem_fd  = -1;
static void  *av_base = NULL;

static inline void reg_write(uint32_t offset, uint32_t value)
{
    volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)av_base + offset);
    *reg = value;
}

static inline uint32_t reg_read(uint32_t offset)
{
    volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)av_base + offset);
    return *reg;
}

int fpga_avalon_open(void)
{
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        perror("fpga_avalon_open: cannot open /dev/mem (need root?)");
        return -1;
    }

    av_base = mmap(NULL,
                   FPGA_AVALON_MAP_SIZE,
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED,
                   mem_fd,
                   FPGA_AVALON_BASE);

    if (av_base == MAP_FAILED) {
    perror("fpga_avalon_open: mmap failed");
    fprintf(stderr, "  base=0x%08lX size=0x%X fd=%d\n",
            (unsigned long)FPGA_AVALON_BASE,
            (unsigned)FPGA_AVALON_MAP_SIZE,
            mem_fd);
        close(mem_fd);
        mem_fd = -1;
        return -1;
    }

    printf("Avalon bridge mapped: phys=0x%08lX virt=%p\n", (unsigned long)FPGA_AVALON_BASE, av_base);

    reg_write(REG_DATA_READY,  0);
    reg_write(REG_DATA_STATUS, 0);
    return 0;
}


int fpga_avalon_write(const imu_angle_frame_t *frame)
{
    if (av_base == NULL) {
        fprintf(stderr, "fpga_avalon_write: not initialised\n");
        return -1;
    }

    reg_write(REG_DATA_STATUS, 1);

    reg_write(REG_ROLL, (uint32_t)frame->roll);
    reg_write(REG_PITCH, (uint32_t)frame->pitch);
    reg_write(REG_GX, (uint32_t)frame->gx);
    reg_write(REG_GY, (uint32_t)frame->gy);

    reg_write(REG_DATA_READY, 1);

    return 0;
}

int fpga_avalon_poll_read(kalman_result_t *result, uint32_t timeout_us)
{
    if (av_base == NULL) {
        fprintf(stderr, "fpga_avalon_poll_read: not initialised\n");
        return -1;
    }

    uint32_t elapsed_us = 0;
    const uint32_t poll_interval_us = 10;

    reg_write(REG_DATA_READY, 0);

    while (reg_read(REG_DATA_STATUS) != 1) {
        if (timeout_us > 0) {
            if (elapsed_us >= timeout_us) {
                fprintf(stderr,
                        "fpga_avalon_poll_read: timed out after %u us\n",
                        timeout_us);
                return -1;
            }
            elapsed_us += poll_interval_us;
        }
        usleep(poll_interval_us);
    }

    /* Kalman cleared data_ready -- read results */
    result->kalman_roll = reg_read(REG_RESULT_ROLL);
    result->kalman_pitch = reg_read(REG_RESULT_PITCH);

    return 0;
}

void fpga_avalon_close(void)
{
    if (av_base != NULL) {
        munmap(av_base, FPGA_AVALON_MAP_SIZE);
        av_base = NULL;
    }
    if (mem_fd >= 0) {
        close(mem_fd);
        mem_fd = -1;
    }
    printf("Avalon bridge unmapped\n");
}

void print_kalman_result(const kalman_result_t *res, const imu_angle_frame_t *angles)
{
    printf("roll=%f pitch=%f | Kalman -> roll=%f pitch=%f\n", 
            q39_to_float(angles->roll),
            q39_to_float(angles->pitch),
            q39_to_float(res->kalman_roll),
            q39_to_float(res->kalman_pitch));
}
