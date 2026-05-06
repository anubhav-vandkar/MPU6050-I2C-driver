#define _GNU_SOURCE
/*
 * fpga_sram.c
 *
 * Writes imu_angle_frame_t into FPGA on-chip SRAM via /dev/mem mmap.
 *
 * Write strategy:
 *   The Avalon-MM bus on the DE1-SoC is 32-bit wide.
 *   We copy the struct as a sequence of uint32_t words so every
 *   bus transaction is naturally aligned. The last word handles
 *   any leftover bytes (struct size not a multiple of 4).
 *
 * data_ready flag:
 *   Written LAST, after all other fields are in SRAM.
 *   The FPGA side polls this byte; writing it last ensures the
 *   FPGA never reads a partially-written frame.
 *   The FPGA should clear it to 0 after consuming the frame.
 */

#include "fpga_sram.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

/* mmap state -- private to this file */
static int    mem_fd   = -1;
static void  *lw_base  = NULL;   /* virtual base of the mmap'd bridge window */

/* ─────────────────────────────────────────
 * fpga_sram_open()
 * ───────────────────────────────────────── */
int fpga_sram_open(void)
{
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        perror("fpga_sram_open: cannot open /dev/mem (need root?)");
        return -1;
    }

    /*
     * mmap the lightweight bridge window into our virtual address space.
     *
     * MAP_SHARED  -- writes go straight through to the physical bus.
     * O_SYNC      -- disables write buffering so writes are not reordered.
     * FPGA_SRAM_BASE must be page-aligned (0xFF200000 is page-aligned).
     */
    lw_base = mmap(NULL,
                   FPGA_SRAM_MAP_SIZE,
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED,
                   mem_fd,
                   FPGA_SRAM_BASE);

    if (lw_base == MAP_FAILED) {
        perror("fpga_sram_open: mmap failed");
        close(mem_fd);
        mem_fd = -1;
        return -1;
    }

    printf("FPGA SRAM mapped: phys=0x%08lX virt=%p size=0x%X\n",
           (unsigned long)FPGA_SRAM_BASE, lw_base, FPGA_SRAM_MAP_SIZE);

    return 0;
}

/* ─────────────────────────────────────────
 * fpga_sram_write()
 *
 * Writes the frame into SRAM in two passes:
 *   Pass 1: write every field EXCEPT data_ready
 *   Pass 2: write data_ready last (acts as a valid/commit flag)
 *
 * This guarantees the FPGA never sees data_ready=1 on a frame
 * that hasn't been fully written yet.
 * ───────────────────────────────────────── */
int fpga_sram_write(const imu_angle_frame_t *frame)
{
    if (lw_base == NULL) {
        fprintf(stderr, "fpga_sram_write: not initialised, call fpga_sram_open() first\n");
        return -1;
    }

    /*
     * Work out where in the mmap'd window our struct lives.
     * volatile uint32_t* tells the compiler not to cache or reorder
     * these writes -- important for memory-mapped I/O.
     */
    volatile uint32_t *dest = (volatile uint32_t *)
                              ((uint8_t *)lw_base + FPGA_SRAM_STRUCT_OFFSET);

    /*
     * Build a local copy of the frame with data_ready cleared.
     * We will set it explicitly in the final write.
     */
    imu_angle_frame_t tmp;
    memcpy(&tmp, frame, sizeof(tmp));
    tmp.data_ready = 0;   /* hold back the ready flag */

    /*
     * Pass 1: write all fields (data_ready still 0).
     *
     * We write word by word (32-bit) to match the Avalon-MM bus width.
     * memcpy into a uint8_t staging buffer first, then write 4 bytes
     * at a time to the volatile pointer.
     *
     * sizeof(imu_angle_frame_t) = 13 bytes (+ 1 pad = 14).
     * 14 / 4 = 3 full words + 2 leftover bytes.
     */
    uint8_t  staging[sizeof(imu_angle_frame_t)];
    memcpy(staging, &tmp, sizeof(tmp));

    int num_words    = sizeof(tmp) / 4;
    int leftover     = sizeof(tmp) % 4;
    uint32_t word;
    int i;

    for (i = 0; i < num_words; i++) {
        memcpy(&word, &staging[i * 4], 4);
        dest[i] = word;
    }

    /* handle leftover bytes (zero-padded into the last word) */
    if (leftover > 0) {
        word = 0;
        memcpy(&word, &staging[num_words * 4], leftover);
        dest[num_words] = word;
    }

    /*
     * Pass 2: write data_ready = 1 into its exact byte position.
     *
     * data_ready is a uint8_t at the end of the struct.
     * We update only its word to avoid touching neighbouring bytes.
     *
     * offsetof() gives the byte offset of data_ready in the struct.
     * We re-read the word currently in SRAM, patch the byte, write back.
     */
    uint32_t dr_offset = (uint32_t)__builtin_offsetof(imu_angle_frame_t, data_ready);
    uint32_t word_idx  = dr_offset / 4;
    uint32_t byte_pos  = dr_offset % 4;   /* which byte within the word */

    /* read the word we already wrote, patch the data_ready byte */
    word = dest[word_idx];
    ((uint8_t *)&word)[byte_pos] = 1;
    dest[word_idx] = word;               /* atomic 32-bit write to the bus */

    return 0;
}

/* ─────────────────────────────────────────
 * fpga_sram_poll_read()
 *
 * Spins watching the data_ready byte in SRAM.
 * The FPGA sets data_ready=1 receipt, processes the frame
 * (Kalman filter etc.), writes its results back into the same
 * SRAM region, then clears data_ready=0.
 *
 * We detect the 0 and copy the whole struct out into *result.
 *
 * Polling loop uses usleep(10) between reads -- 10 µs granularity
 * is fine given Kalman filter latency will be >> 10 µs.
 * Tight spinning without sleep would peg the HPS core for no gain.
 * ───────────────────────────────────────── */
int fpga_sram_poll_read(imu_angle_frame_t *result, uint32_t timeout_us)
{
    if (lw_base == NULL) {
        fprintf(stderr, "fpga_sram_poll_read: not initialised\n");
        return -1;
    }

    volatile uint8_t *sram_base = (volatile uint8_t *)lw_base
                                  + FPGA_SRAM_STRUCT_OFFSET;

    /*
     * Locate data_ready byte inside the mmap'd window.
     * Same offset calculation used in fpga_sram_write().
     */
    uint32_t dr_offset = (uint32_t)__builtin_offsetof(imu_angle_frame_t,
                                                       data_ready);
    volatile uint8_t *dr_ptr = sram_base + dr_offset;

    uint32_t elapsed_us = 0;
    const uint32_t poll_interval_us = 10;   /* sleep between polls */

    /*
     * Spin until data_ready == 0.
     * The FPGA clears this byte after writing its results back.
     */
    while (*dr_ptr != 0) {
        if (timeout_us > 0) {
            if (elapsed_us >= timeout_us) {
                fprintf(stderr,
                        "fpga_sram_poll_read: timed out after %u us\n",
                        timeout_us);
                return -1;
            }
            elapsed_us += poll_interval_us;
        }
        usleep(poll_interval_us);
    }

    /*
     * data_ready is now 0 -- FPGA has finished writing.
     * Read the full struct back out of SRAM word by word,
     * then memcpy into the caller's result struct.
     *
     * We read into a staging buffer first for the same reason
     * we write from one: keeps all bus accesses 32-bit aligned.
     */
    uint8_t  staging[sizeof(imu_angle_frame_t)];
    volatile uint32_t *src = (volatile uint32_t *)sram_base;

    int num_words = sizeof(imu_angle_frame_t) / 4;
    int leftover  = sizeof(imu_angle_frame_t) % 4;
    uint32_t word;
    int i;

    for (i = 0; i < num_words; i++) {
        word = src[i];
        memcpy(&staging[i * 4], &word, 4);
    }
    if (leftover > 0) {
        word = src[num_words];
        memcpy(&staging[num_words * 4], &word, leftover);
    }

    memcpy(result, staging, sizeof(imu_angle_frame_t));

    return 0;
}

/* ─────────────────────────────────────────
 * fpga_sram_close()
 * ───────────────────────────────────────── */
void fpga_sram_close(void)
{
    if (lw_base != NULL) {
        munmap(lw_base, FPGA_SRAM_MAP_SIZE);
        lw_base = NULL;
    }
    if (mem_fd >= 0) {
        close(mem_fd);
        mem_fd = -1;
    }
    printf("FPGA SRAM unmapped\n");
}
