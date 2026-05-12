#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
    VGA_WIDTH       = 640,
    VGA_HEIGHT      = 480,
    MAX_SEGMENTS    = 4,

    VGA_BUF0_WORD   = 0x0000,
    VGA_BUF1_WORD   = 0x0800,
    VGA_CTRL_WORD   = 0x1000,

    SEG_SKY         = 0,
    SEG_GROUND      = 1,
    SEG_HORIZON     = 2,
    SEG_MARKER      = 3,

    KALMAN_ROLL_WORD  = 0x0006,
    KALMAN_PITCH_WORD = 0x0007,
};

/* Physical addresses.
 *
 * On Cyclone V the HPS lightweight HPS-to-FPGA bridge starts at
 * 0xFF200000. Add the offset assigned to each peripheral in Platform
 * Designer. Adjust to match your project. */
#define LWH2F_BASE        0xFF200000u
#define VGA_PHYS_BASE     (LWH2F_BASE + 0x0000u)
#define KALMAN_PHYS_BASE  (LWH2F_BASE + 0x0040u)

#define VGA_MAP_SIZE      0x8000u   /* covers up to word 0x1000 */
#define KALMAN_MAP_SIZE   0x0100u

/* Tuning */
static const float PIXELS_PER_DEGREE = 4.0f;
static const float LINE_THICKNESS_PX = 5.0f;
static const float ROLL_SIGN  = -1.0f;
static const float PITCH_SIGN =  1.0f;

/* Print the current angles to stderr once per N frames; useful when
 * bringing up the Kalman link. Set to 0 to disable. */
#define PRINT_EVERY_N_FRAMES  30

/* --------------------------------------------------------------- */
static inline int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline int16_t sign_extend_12(uint16_t v) {
    v &= 0x0FFF;
    if (v & 0x0800) v |= 0xF000;
    return (int16_t)v;
}

static inline float q3_9_to_float_deg(uint32_t word) {
    int16_t raw = sign_extend_12((uint16_t)word);
    return (float)raw / 512.0f;
}

/* --------------------------------------------------------------- */
static void *map_peripheral(int fd, off_t phys_base, size_t length) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) { perror("sysconf"); exit(1); }

    off_t page_base = phys_base & ~((off_t)page_size - 1);
    off_t page_off  = phys_base - page_base;
    size_t map_len  = (size_t)page_off + length;

    void *m = mmap(NULL, map_len, PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, page_base);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }
    return (uint8_t *)m + page_off;
}

/* --------------------------------------------------------------- */
static void build_segments(
    float roll_deg, float pitch_deg,
    uint16_t seg_xstart[VGA_HEIGHT][MAX_SEGMENTS],
    uint16_t seg_xstop [VGA_HEIGHT][MAX_SEGMENTS])
{
    const float cx  = (VGA_WIDTH  - 1) * 0.5f;
    const float cy0 = (VGA_HEIGHT - 1) * 0.5f;

    const float theta = ROLL_SIGN * roll_deg * (float)M_PI / 180.0f;
    const float s = sinf(theta);
    const float c = cosf(theta);

    const float cy = cy0 + (PITCH_SIGN * pitch_deg * PIXELS_PER_DEGREE);
    const float half = LINE_THICKNESS_PX * 0.5f;

    const int sky_side_positive = (c < 0.0f);

    memset(seg_xstart, 0, sizeof(uint16_t) * VGA_HEIGHT * MAX_SEGMENTS);
    memset(seg_xstop, 0, sizeof(uint16_t) * VGA_HEIGHT * MAX_SEGMENTS);

    for (int y = 0; y < VGA_HEIGHT; ++y) {
        int line_xs = 0, line_xe = 0;
        int has_line = 0;

        if (fabsf(s) < 1e-6f) {
            if (fabsf((float)y - cy) <= half) {
                line_xs = 0;
                line_xe = VGA_WIDTH - 1;
                has_line = 1;
            }
        } else {
            float x1 = cx + ((((float)y - cy) * c) - half) / s;
            float x2 = cx + ((((float)y - cy) * c) + half) / s;
            if (x1 > x2) { float t = x1; x1 = x2; x2 = t; }

            int xs = clamp_int((int)ceilf(x1),  0, VGA_WIDTH - 1);
            int xe = clamp_int((int)floorf(x2), 0, VGA_WIDTH - 1);

            if (xe > xs) {
                line_xs = xs;
                line_xe = xe;
                has_line = 1;
            }
        }

        if (has_line) {
            int left_has  = (line_xs > 0);
            int right_has = (line_xe < VGA_WIDTH - 1);

            int left_is_sky  = 0;
            int right_is_sky = 0;

            if (fabsf(s) < 1e-6f) {
                left_is_sky  = ((float)y < cy);
                right_is_sky = left_is_sky;
            } else {
                if (left_has) {
                    float xp = (float)line_xs * 0.5f;
                    float lhs = ((float)y - cy) * c - (xp - cx) * s;
                    left_is_sky = ((lhs > 0.0f) == sky_side_positive);
                }
                if (right_has) {
                    float xp = ((float)line_xe + (float)(VGA_WIDTH - 1)) * 0.5f;
                    float lhs = ((float)y - cy) * c - (xp - cx) * s;
                    right_is_sky = ((lhs > 0.0f) == sky_side_positive);
                }
            }

            if (left_has) {
                if (left_is_sky) {
                    seg_xstart[y][SEG_SKY] = 0;
                    seg_xstop [y][SEG_SKY] = (uint16_t)(line_xs - 1);
                } else {
                    seg_xstart[y][SEG_GROUND] = 0;
                    seg_xstop [y][SEG_GROUND] = (uint16_t)(line_xs - 1);
                }
            }
            if (right_has) {
                if (right_is_sky) {
                    seg_xstart[y][SEG_SKY] = (uint16_t)(line_xe + 1);
                    seg_xstop [y][SEG_SKY] = (uint16_t)(VGA_WIDTH - 1);
                } else {
                    seg_xstart[y][SEG_GROUND] = (uint16_t)(line_xe + 1);
                    seg_xstop [y][SEG_GROUND] = (uint16_t)(VGA_WIDTH - 1);
                }
            }

            seg_xstart[y][SEG_HORIZON] = (uint16_t)line_xs;
            seg_xstop [y][SEG_HORIZON] = (uint16_t)line_xe;
        } else {
            int row_is_sky;
            if (fabsf(s) < 1e-6f) {
                row_is_sky = ((float)y < cy);
            } else {
                float lhs = ((float)y - cy) * c;
                row_is_sky = ((lhs > 0.0f) == sky_side_positive);
            }
            if (row_is_sky) {
                seg_xstart[y][SEG_SKY] = 0;
                seg_xstop [y][SEG_SKY] = (uint16_t)(VGA_WIDTH - 1);
            } else {
                seg_xstart[y][SEG_GROUND] = 0;
                seg_xstop [y][SEG_GROUND] = (uint16_t)(VGA_WIDTH - 1);
            }
        }
    }
}

/* --------------------------------------------------------------- */
static inline uint32_t pack_pos(uint16_t xstart, uint16_t xstop) {
    return ((uint32_t)xstop << 16) | (uint32_t)xstart;
}

static void write_buffer(
    volatile uint32_t *vga_words,
    uint32_t base_word,
    const uint16_t seg_xstart[VGA_HEIGHT][MAX_SEGMENTS],
    const uint16_t seg_xstop [VGA_HEIGHT][MAX_SEGMENTS])
{
    for (int y = 0; y < VGA_HEIGHT; ++y) {
        for (int seg = 0; seg < MAX_SEGMENTS; ++seg) {
            uint32_t idx = (uint32_t)y * MAX_SEGMENTS + (uint32_t)seg;
            vga_words[base_word + idx] = pack_pos(
                seg_xstart[y][seg], seg_xstop[y][seg]);
        }
    }
}

static void request_swap(volatile uint32_t *vga_words, int next_buffer) {
    vga_words[VGA_CTRL_WORD] = (uint32_t)(next_buffer & 0x1);
}

/* --------------------------------------------------------------- */
int main(void) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open(/dev/mem)"); return 1; }

    volatile uint32_t *vga_words = (volatile uint32_t *)map_peripheral(
        fd, (off_t)VGA_PHYS_BASE, VGA_MAP_SIZE);

    volatile uint32_t *kalman_words = (volatile uint32_t *)map_peripheral(
        fd, (off_t)KALMAN_PHYS_BASE, KALMAN_MAP_SIZE);

    static uint16_t seg_xstart[VGA_HEIGHT][MAX_SEGMENTS];
    static uint16_t seg_xstop [VGA_HEIGHT][MAX_SEGMENTS];

    int next_buffer = 0;
    int frame = 0;

    struct timespec next_tick;
    clock_gettime(CLOCK_MONOTONIC, &next_tick);

    for (;;) {
        /* Read latest estimates from the Kalman filter */
        uint32_t roll_word  = kalman_words[KALMAN_ROLL_WORD];
        uint32_t pitch_word = kalman_words[KALMAN_PITCH_WORD];

        float roll_deg  = q3_9_to_float_deg(roll_word);
        float pitch_deg = q3_9_to_float_deg(pitch_word);

#if PRINT_EVERY_N_FRAMES > 0
        if ((frame % PRINT_EVERY_N_FRAMES) == 0) {
            fprintf(stderr, "roll = %+7.2f  pitch = %+7.2f  "
                    "(raw 0x%03X 0x%03X)\n",
                    roll_deg, pitch_deg,
                    roll_word & 0xFFF, pitch_word & 0xFFF);
        }
#endif

        build_segments(roll_deg, pitch_deg, seg_xstart, seg_xstop);

        uint32_t base = (next_buffer == 0) ? VGA_BUF0_WORD : VGA_BUF1_WORD;
        write_buffer(vga_words, base, seg_xstart, seg_xstop);

        request_swap(vga_words, next_buffer);
        next_buffer ^= 1;
        frame++;

        next_tick.tv_nsec += 16666667L;
        while (next_tick.tv_nsec >= 1000000000L) {
            next_tick.tv_nsec -= 1000000000L;
            next_tick.tv_sec  += 1;
        }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_tick, NULL);
    }

    close(fd);
    return 0;
}