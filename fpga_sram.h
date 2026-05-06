#ifndef FPGA_SRAM_H
#define FPGA_SRAM_H

/*
 * fpga_sram.h
 *
 * Writes imu_angle_frame_t structs into FPGA on-chip SRAM via the
 * HPS-to-FPGA Lightweight Avalon-MM bridge.
 *
 * ── How this works on the DE1-SoC ──────────────────────────────
 *
 * The HPS exposes the Lightweight HPS-to-FPGA bridge at physical
 * address 0xFF200000. In Qsys (Platform Designer) you instantiate
 * an on-chip SRAM (or any Avalon-MM slave) and give it an offset
 * from that base. For example, if Qsys assigned your SRAM the
 * address range 0x0000_0000 to 0x0000_FFFF (offset 0 on the bridge),
 * then the HPS sees it at 0xFF20_0000.
 *
 * We cannot touch physical addresses directly from userspace.
 * Instead:
 *   1. Open /dev/mem
 *   2. mmap() a window covering the bridge region into our process
 *   3. Write to the virtual address -- the kernel MMU translates it
 *      to the correct physical bus transaction on the AXI→Avalon bridge
 *
 * ── Address you need to check ───────────────────────────────────
 *
 * Open your Qsys project, find your SRAM component, and read the
 * "Base" address shown in the address map. That offset goes into
 * FPGA_SRAM_BASE below.
 *
 * Common default on DE1-SoC tutorial projects: 0x00000000 offset
 * (i.e., starts right at the bridge base 0xFF200000).
 * If your SRAM is at Qsys offset 0x08000, set:
 *   #define FPGA_SRAM_BASE  0xFF208000
 */

#include <stdint.h>
#include "imu_angles.h"

/* ── Physical address of the Lightweight HPS-to-FPGA bridge ──── */
#define HPS_TO_FPGA_LW_BASE   0xFF200000UL

/*
 * Your SRAM's base address as seen from the HPS.
 * = HPS_TO_FPGA_LW_BASE + (Qsys offset of your SRAM component)
 *
 * !! CHANGE THIS to match your Qsys address map !!
 */
#define FPGA_SRAM_BASE        0xFF200000UL

/*
 * How many bytes of the bridge we mmap.
 * The lightweight bridge window is 2 MB total (0xFF200000-0xFF3FFFFF).
 * We only need enough to cover our struct -- 4 KB (one page) is fine.
 */
#define FPGA_SRAM_MAP_SIZE    0x1000      /* 4 KB */

/*
 * Byte offset within the mmap'd window where the struct is written.
 * If your SRAM starts at FPGA_SRAM_BASE, offset 0 means write at
 * the very first word of the SRAM.
 */
#define FPGA_SRAM_STRUCT_OFFSET  0x0000

/* ── Function prototypes ─────────────────────────────────────── */

/*
 * fpga_sram_open()
 *
 * Opens /dev/mem and mmap's the bridge window.
 * Must be called once before fpga_sram_write().
 * Returns 0 on success, -1 on error.
 * Requires root (or CAP_SYS_RAWIO) to open /dev/mem.
 */
int fpga_sram_open(void);

/*
 * fpga_sram_write()
 *
 * Copies one imu_angle_frame_t into FPGA SRAM via the mmap'd window.
 * Word-by-word writes ensure each Avalon-MM transaction is aligned.
 * Returns 0 on success, -1 if not initialised.
 */
int fpga_sram_write(const imu_angle_frame_t *frame);

/*
 * fpga_sram_poll_read()
 *
 * Spins on the data_ready byte in SRAM until the FPGA clears it to 0,
 * then reads the entire frame back into *result.
 *
 * Call this immediately after fpga_sram_write(). The FPGA is expected
 * to:
 *   1. Detect data_ready == 1
 *   2. Run the Kalman filter (or any other processing)
 *   3. Write its output values back into the same SRAM region
 *   4. Clear data_ready to 0 to signal completion
 *
 * timeout_us: maximum microseconds to wait before giving up.
 *             Pass 0 to wait forever (not recommended in production).
 *
 * Returns  0 -- FPGA cleared the flag, *result is valid
 *         -1 -- timed out or not initialised
 */
int fpga_sram_poll_read(imu_angle_frame_t *result, uint32_t timeout_us);

/*
 * fpga_sram_close()
 *
 * Unmaps the bridge window and closes /dev/mem.
 */
void fpga_sram_close(void);

#endif /* FPGA_SRAM_H */
