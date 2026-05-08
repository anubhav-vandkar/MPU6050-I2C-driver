#define _GNU_SOURCE

/*
 * fpga_avalon.c
 *
 * Sends imu_angle_frame_t to the Kalman filter's Avalon-MM slave
 * registers and polls for results -- no SRAM involved.
 *
 * Each register write is one 32-bit Avalon transaction.
 * The Kalman filter latches inputs on the write strobe and signals
 * completion by clearing REG_DATA_READY to 0.
 */

#include "fpga_avalon.h"

#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

static int    mem_fd  = -1;
static void  *av_base = NULL;   /* virtual base of mmap'd bridge window */

/* ── Register access helpers ────────────────────────────────────── */

/*
 * reg_write() / reg_read()
 *
 * Each goes through a volatile uint32_t pointer so the compiler
 * cannot cache or reorder the access -- required for MMIO.
 * offset is in bytes; we divide by 4 to index into the uint32 array.
 */
static inline void reg_write(uint32_t offset, uint32_t value)
{
    volatile uint32_t *reg = (volatile uint32_t *)
                             ((uint8_t *)av_base + offset);
    *reg = value;
}

static inline uint32_t reg_read(uint32_t offset)
{
    volatile uint32_t *reg = (volatile uint32_t *)
                             ((uint8_t *)av_base + offset);
    return *reg;
}

/* ─────────────────────────────────────────
 * fpga_avalon_open()
 * ───────────────────────────────────────── */
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
        close(mem_fd);
        mem_fd = -1;
        return -1;
    }

    printf("Avalon bridge mapped: phys=0x%08lX virt=%p\n",
           (unsigned long)FPGA_AVALON_BASE, av_base);
    return 0;
}

/* ─────────────────────────────────────────
 * fpga_avalon_write()
 *
 * Sends the two packed 32-bit words to the Kalman slave.
 *
 * Write order matters:
 *   1. roll_pitch first  -- Kalman latches but does not start yet
 *   2. gx_gy second      -- this write triggers the Kalman filter
 *
 * If your .sv uses a dedicated "start" register instead of treating
 * the gx_gy write as a trigger, add a third write here:
 *   reg_write(REG_START, 1);
 * ───────────────────────────────────────── */
int fpga_avalon_write(const imu_angle_frame_t *frame)
{
    if (av_base == NULL) {
        fprintf(stderr, "fpga_avalon_write: not initialised\n");
        return -1;
    }

    reg_write(REG_ROLL_PITCH, (uint32_t)frame->roll_pitch);
    reg_write(REG_GX_GY,      (uint32_t)frame->gx_gy);

    /*
     * No data_ready = 1 write from the HPS.
     * The Avalon write strobe on REG_GX_GY is the trigger.
     * The Kalman filter .sv should start when it sees a write
     * to address offset 0x04.
     */

    return 0;
}

/* ─────────────────────────────────────────
 * fpga_avalon_poll_read()
 *
 * Spins on REG_DATA_READY until the Kalman filter writes 0,
 * then reads result registers into *result.
 * ───────────────────────────────────────── */
int fpga_avalon_poll_read(kalman_result_t *result, uint32_t timeout_us)
{
    if (av_base == NULL) {
        fprintf(stderr, "fpga_avalon_poll_read: not initialised\n");
        return -1;
    }

    uint32_t elapsed_us = 0;
    const uint32_t poll_interval_us = 10;

    while (reg_read(REG_DATA_READY) != 0) {
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
    result->result_0 = reg_read(REG_RESULT_0);
    result->result_1 = reg_read(REG_RESULT_1);

    return 0;
}

/* ─────────────────────────────────────────
 * fpga_avalon_close()
 * ───────────────────────────────────────── */
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
