#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
    VGA_WIDTH     = 640,
    VGA_HEIGHT    = 480,
    WORDS_PER_ROW = 16,

    /* Word addresses in the VGA peripheral (32-bit registers)  */
    VGA_BUF0_WORD = 0x0000,
    VGA_BUF1_WORD = 0x2000,
    VGA_CTRL_WORD = 0x4000,
};

#define VGA_PHYS_BASE  0xff200000u
#define VGA_MAP_LEN    0x10004u

static const float PIXELS_PER_DEGREE    =  4.0f;
static const float LINE_THICKNESS_PX    =  5.0f;
static const float LADDER_THICKNESS_PX  =  3.0f;
static const float LADDER_HALF_LEN_LONG = 90.0f;  /* ±20 °, ±40 °, ±60 ° marks */
static const float LADDER_HALF_LEN_SHORT= 45.0f;  /* ±10 °, ±30 °, ±50 ° marks */
static const float ROLL_SIGN            = -1.0f;
static const float PITCH_SIGN           =  1.0f;
static const int   INFO_BAR_HEIGHT      = 24;

/* Fixed boresight dot at the screen centre */
#define CENTER_X 319
#define CENTER_Y 239
#define DOT_HALF   3   /* half-size in pixels → 7 × 7 yellow block */

enum {
    COLOR_SKY    = 0x5B,   /* blue             */
    COLOR_GRASS  = 0x10,   /* green            */
    COLOR_HORIZON= 0xFF,   /* white            */
    COLOR_LADDER = 0x92,   /* gray             */
    COLOR_TEXT   = 0xFF,   /* white            */
    COLOR_YELLOW = 0xFC,   /* yellow           */
    COLOR_BLACK  = 0x00    /* black            */
};

#define FONT_W 5
#define FONT_H 7
#define FONT_ADV 6

static volatile uint32_t *g_vga_words = NULL;
static int g_display_buffer = 0;

/* ------------------------------------------------------------------ */
/* Utilities                                                           */
/* ------------------------------------------------------------------ */

static inline int clamp_int(int v, int lo, int hi) {
    if(v < lo) return lo;
    if(v > hi) return hi;
    return v;
}

typedef struct { float x; float y; } point_t;

static inline float q3_9_to_float_roll(uint16_t raw) {
    raw &= 0x0FFFu;
    if(raw > 750u && raw < 3400u) raw = 0u;
    int16_t signed_raw = (raw & 0x0800u) ? (int16_t)(raw | 0xF000u) : (int16_t)raw;
    return (float)signed_raw / 512.0f;
}

static inline float q3_9_to_float_pitch(uint16_t raw) {
    raw &= 0x0FFFu;
    if(raw > 600u && raw < 3400u) raw = 0u;
    int16_t signed_raw = (raw & 0x0800u) ? (int16_t)(raw | 0xF000u) : (int16_t)raw;
    return (float)signed_raw / 512.0f;
}

static inline float q3_9_to_degrees_roll(uint16_t raw) {
    return q3_9_to_float_roll(raw) * 180.0f / (float)M_PI;
}

static inline float q3_9_to_degrees_pitch(uint16_t raw) {
    return q3_9_to_float_pitch(raw) * 180.0f / (float)M_PI;
}

static inline uint32_t pack_segment_word(int xs, int xe, uint8_t color) {
    xs = clamp_int(xs, 0, 0x0FFF);
    xe = clamp_int(xe, 0, 0x0FFF);
    if(xe <= xs) return 0;
    return ((uint32_t)(color & 0xFFu)   << 24) |
           ((uint32_t)(xe    & 0x0FFFu) << 12) |
           ((uint32_t)(xs    & 0x0FFFu) <<  0);
}

static inline void append_segment(uint32_t row_words[WORDS_PER_ROW],
                                  int *seg_count, int xs, int xe, uint8_t color) {
    if(*seg_count >= WORDS_PER_ROW) return;
    uint32_t packed = pack_segment_word(xs, xe, color);
    if(packed == 0) return;
    row_words[(*seg_count)++] = packed;
}

/* ------------------------------------------------------------------ */
/* Font glyph table                                                    */
/* ------------------------------------------------------------------ */

static const uint8_t *glyph_rows(char ch) {
    static const uint8_t SPACE[FONT_H]= {0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static const uint8_t P[FONT_H]    = {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10};
    static const uint8_t I[FONT_H]    = {0x1F,0x04,0x04,0x04,0x04,0x04,0x1F};
    static const uint8_t T_[FONT_H]   = {0x1F,0x04,0x04,0x04,0x04,0x04,0x04};
    static const uint8_t C[FONT_H]    = {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E};
    static const uint8_t H[FONT_H]    = {0x11,0x11,0x11,0x1F,0x11,0x11,0x11};
    static const uint8_t R[FONT_H]    = {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11};
    static const uint8_t O[FONT_H]    = {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E};
    static const uint8_t L[FONT_H]    = {0x10,0x10,0x10,0x10,0x10,0x10,0x1F};
    static const uint8_t PLUS[FONT_H] = {0x00,0x04,0x04,0x1F,0x04,0x04,0x00};
    static const uint8_t MINUS[FONT_H]= {0x00,0x00,0x00,0x1F,0x00,0x00,0x00};
    static const uint8_t DOT[FONT_H]  = {0x00,0x00,0x00,0x00,0x00,0x04,0x04};
    static const uint8_t COLON[FONT_H]= {0x00,0x04,0x04,0x00,0x04,0x04,0x00};
    static const uint8_t D0[FONT_H]   = {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E};
    static const uint8_t D1[FONT_H]   = {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E};
    static const uint8_t D2[FONT_H]   = {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F};
    static const uint8_t D3[FONT_H]   = {0x1E,0x01,0x01,0x0E,0x01,0x01,0x1E};
    static const uint8_t D4[FONT_H]   = {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02};
    static const uint8_t D5[FONT_H]   = {0x1F,0x10,0x10,0x1E,0x01,0x01,0x1E};
    static const uint8_t D6[FONT_H]   = {0x0E,0x10,0x10,0x1E,0x11,0x11,0x0E};
    static const uint8_t D7[FONT_H]   = {0x1F,0x01,0x02,0x04,0x08,0x08,0x08};
    static const uint8_t D8[FONT_H]   = {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E};
    static const uint8_t D9[FONT_H]   = {0x0E,0x11,0x11,0x0F,0x01,0x01,0x0E};

    switch(ch) {
        case 'P': return P;    case 'I': return I;
        case 'T': return T_;   case 'C': return C;
        case 'H': return H;    case 'R': return R;
        case 'O': return O;    case 'L': return L;
        case '+': return PLUS; case '-': return MINUS;
        case '.': return DOT;  case ':': return COLON;
        case '0': return D0;   case '1': return D1;
        case '2': return D2;   case '3': return D3;
        case '4': return D4;   case '5': return D5;
        case '6': return D6;   case '7': return D7;
        case '8': return D8;   case '9': return D9;
        default:  return SPACE;
    }
}

/* ------------------------------------------------------------------ */
/* Text rendering                                                      */
/* ------------------------------------------------------------------ */

/* Normal weight: one segment per contiguous run of lit pixels. */
static void draw_text_row(uint32_t row_words[WORDS_PER_ROW],
                          int *seg_count, int row,
                          int start_x, int start_y,
                          const char *text, uint8_t color) {
    if(row < start_y || row >= start_y + FONT_H) return;

    const int glyph_row = row - start_y;
    int x = start_x;

    for(const char *p = text; *p != '\0'; ++p) {
        const uint8_t *g = glyph_rows(*p);
        uint8_t bits = g[glyph_row] & 0x1Fu;
        int col = 0;
        while(col < FONT_W) {
            if(!(bits & (1u << (FONT_W - 1 - col)))) { ++col; continue; }
            int run_start = col;
            while(col < FONT_W && (bits & (1u << (FONT_W - 1 - col)))) ++col;
            append_segment(row_words, seg_count, x + run_start, x + col - 1, color);
            if(*seg_count >= WORDS_PER_ROW) return;
        }
        x += FONT_ADV;
        if(x >= VGA_WIDTH) return;
    }
}

/*
 * Bold weight: each lit pixel run is extended one pixel to the right.
 * Same segment count as the normal variant — no extra budget consumed.
 */
static void draw_text_row_bold(uint32_t row_words[WORDS_PER_ROW],
                               int *seg_count, int row,
                               int start_x, int start_y,
                               const char *text, uint8_t color) {
    if(row < start_y || row >= start_y + FONT_H) return;

    const int glyph_row = row - start_y;
    int x = start_x;

    for(const char *p = text; *p != '\0'; ++p) {
        const uint8_t *g = glyph_rows(*p);
        uint8_t bits = g[glyph_row] & 0x1Fu;
        int col = 0;
        while(col < FONT_W) {
            if(!(bits & (1u << (FONT_W - 1 - col)))) { ++col; continue; }
            int run_start = col;
            while(col < FONT_W && (bits & (1u << (FONT_W - 1 - col)))) ++col;
            /* +1 on the right edge for the bold stroke */
            append_segment(row_words, seg_count, x + run_start, x + col, color);
            if(*seg_count >= WORDS_PER_ROW) return;
        }
        x += FONT_ADV;
        if(x >= VGA_WIDTH) return;
    }
}

/* ------------------------------------------------------------------ */
/* Scanline polygon helpers                                            */
/* ------------------------------------------------------------------ */

static int gather_scanline_intersections(float scan_y, const point_t p[4],
                                         float xs_out[4]) {
    int n = 0;
    for(int i = 0; i < 4; ++i) {
        point_t a = p[i];
        point_t b = p[(i + 1) & 3];
        if(fabsf(a.y - b.y) < 1e-6f) continue;
        float ymin = fminf(a.y, b.y);
        float ymax = fmaxf(a.y, b.y);
        if(scan_y >= ymin && scan_y < ymax) {
            float t = (scan_y - a.y) / (b.y - a.y);
            float x = a.x + t * (b.x - a.x);
            int duplicate = 0;
            for(int j = 0; j < n; ++j) {
                if(fabsf(xs_out[j] - x) < 0.5f) { duplicate = 1; break; }
            }
            if(!duplicate && n < 4) xs_out[n++] = x;
        }
    }
    if(n < 2) return 0;
    /* bubble sort the two (usually) intersections */
    for(int i = 0; i < n - 1; ++i)
        for(int j = i + 1; j < n; ++j)
            if(xs_out[j] < xs_out[i]) {
                float tmp = xs_out[i]; xs_out[i] = xs_out[j]; xs_out[j] = tmp;
            }
    return n;
}

static void add_thick_segment_row(uint32_t row_words[WORDS_PER_ROW],
                                  int *seg_count, int row,
                                  float x0, float y0, float x1, float y1,
                                  float thickness_px, uint8_t color) {
    if(*seg_count >= WORDS_PER_ROW) return;

    float half_t  = thickness_px * 0.5f;
    float scan_y  = (float)row;

    /* Horizontal fast-path */
    if(fabsf(y1 - y0) < 1e-6f) {
        if(fabsf(scan_y - y0) > half_t) return;
        int xs = (int)ceilf(fminf(x0, x1));
        int xe = (int)floorf(fmaxf(x0, x1));
        append_segment(row_words, seg_count, xs, xe, color);
        return;
    }

    float dx = x1 - x0, dy = y1 - y0;
    float len = sqrtf(dx * dx + dy * dy);
    if(len < 1e-6f) return;

    float nx = -dy / len, ny = dx / len;
    point_t poly[4];
    poly[0].x = x0 + nx * half_t;  poly[0].y = y0 + ny * half_t;
    poly[1].x = x1 + nx * half_t;  poly[1].y = y1 + ny * half_t;
    poly[2].x = x1 - nx * half_t;  poly[2].y = y1 - ny * half_t;
    poly[3].x = x0 - nx * half_t;  poly[3].y = y0 - ny * half_t;

    float xs[4];
    int n = gather_scanline_intersections(scan_y, poly, xs);
    if(n < 2) return;

    float xmin = xs[0], xmax = xs[0];
    for(int i = 1; i < n; ++i) {
        if(xs[i] < xmin) xmin = xs[i];
        if(xs[i] > xmax) xmax = xs[i];
    }
    append_segment(row_words, seg_count,
                   (int)ceilf(xmin), (int)floorf(xmax), color);
}

/* ------------------------------------------------------------------ */
/* Horizon fill                                                        */
/* ------------------------------------------------------------------ */

static void add_horizon_row(uint32_t row_words[WORDS_PER_ROW],
                            int *seg_count, int y,
                            float cx, float cy, float s, float c) {
    const float half_thickness = LINE_THICKNESS_PX * 0.5f;
    const float fy = (float)y;

    if(fabsf(s) < 1e-6f) {
        if     (fabsf(fy - cy) <= half_thickness)
            append_segment(row_words, seg_count, 0, VGA_WIDTH - 1, COLOR_HORIZON);
        else if(fy < cy)
            append_segment(row_words, seg_count, 0, VGA_WIDTH - 1, COLOR_SKY);
        else
            append_segment(row_words, seg_count, 0, VGA_WIDTH - 1, COLOR_GRASS);
        return;
    }

    float x1 = cx + ((fy - cy) * c - half_thickness) / s;
    float x2 = cx + ((fy - cy) * c + half_thickness) / s;
    if(x1 > x2) { float tmp = x1; x1 = x2; x2 = tmp; }

    if(x2 < 0.0f) {
        append_segment(row_words, seg_count, 0, VGA_WIDTH - 1,
                       (s > 0.0f) ? COLOR_SKY : COLOR_GRASS);
        return;
    }
    if(x1 > (float)(VGA_WIDTH - 1)) {
        append_segment(row_words, seg_count, 0, VGA_WIDTH - 1,
                       (s > 0.0f) ? COLOR_GRASS : COLOR_SKY);
        return;
    }

    int xs = clamp_int((int)ceilf(x1),  0, VGA_WIDTH - 1);
    int xe = clamp_int((int)floorf(x2), 0, VGA_WIDTH - 1);
    if(xe <= xs) {
        append_segment(row_words, seg_count, 0, VGA_WIDTH - 1,
                       (s > 0.0f) ? COLOR_GRASS : COLOR_SKY);
        return;
    }

    uint8_t before = (s > 0.0f) ? COLOR_GRASS : COLOR_SKY;
    uint8_t after  = (s > 0.0f) ? COLOR_SKY   : COLOR_GRASS;
    if(xs > 0)             append_segment(row_words, seg_count, 0,        xs - 1,       before);
                           append_segment(row_words, seg_count, xs,       xe,            COLOR_HORIZON);
    if(xe < VGA_WIDTH - 1) append_segment(row_words, seg_count, xe + 1,  VGA_WIDTH - 1, after);
}

/* ------------------------------------------------------------------ */
/* Pitch ladder                                                        */
/* ------------------------------------------------------------------ */

/*
 * Ladder mark table.  12 marks: six above and six below the horizon.
 *
 * ladder_offsets_deg[i]:  screen-space y offset = offset * PPD.
 *   Negative → above screen centre (higher pitch shown in sky).
 *   Label shown = −offset (e.g. offset=−10 → "+10").
 *
 * ladder_half_lens[i]:  half-length of horizontal bar in pixels.
 *   Long  bars for ±20 °, ±40 °, ±60 ° (multiples of 20).
 *   Short bars for ±10 °, ±30 °, ±50 °.
 */
static const float ladder_offsets_deg[] = {
    -60.0f, -50.0f, -40.0f, -30.0f, -20.0f, -10.0f,
     10.0f,  20.0f,  30.0f,  40.0f,  50.0f,  60.0f
};

#define LADDER_COUNT \
    ((int)(sizeof(ladder_offsets_deg) / sizeof(ladder_offsets_deg[0])))

/* Long for multiples of 20, short otherwise (matches label magnitude). */
static const float ladder_half_lens[] = {
    /* -60  -50   -40   -30   -20   -10  */
    90.0f, 45.0f, 90.0f, 45.0f, 90.0f, 45.0f,
    /* +10  +20   +30   +40   +50   +60  */
    45.0f, 90.0f, 45.0f, 90.0f, 45.0f, 90.0f
};

/* Inertial pitch each mark represents (label = −offset). */
static const char * const ladder_labels[] = {
    "+60", "+50", "+40", "+30", "+20", "+10",
    "-10", "-20", "-30", "-40", "-50", "-60"
};

static void add_pitch_ladder_rows(uint32_t row_words[WORDS_PER_ROW],
                                  int *seg_count, int y,
                                  float cx, float cy0,
                                  float pitch_deg, float s, float c) {
    for(int i = 0; i < LADDER_COUNT; ++i) {
        if(*seg_count >= WORDS_PER_ROW) return;

        float ladder_pitch = pitch_deg + ladder_offsets_deg[i];
        float line_cy      = cy0 + (PITCH_SIGN * ladder_pitch * PIXELS_PER_DEGREE);
        float half_len     = ladder_half_lens[i];

        /* Rotated endpoints of the ladder bar. */
        float dx = half_len * c;
        float dy = half_len * s;
        float x0 = cx - dx,  y0 = line_cy - dy;
        float x1 = cx + dx,  y1 = line_cy + dy;

        /* Draw the ladder bar itself. */
        add_thick_segment_row(row_words, seg_count, y,
                              x0, y0, x1, y1,
                              LADDER_THICKNESS_PX, COLOR_LADDER);

        /*
         * Draw pitch-angle label to the left and right of the bar.
         * The glyph window is centred on line_cy.
         * Only render on scanlines that fall inside that window.
         */
        int label_top = (int)line_cy - FONT_H / 2;
        if(y >= label_top && y < label_top + FONT_H) {
            const char *lbl = ladder_labels[i];
            int nchars      = (int)strlen(lbl);

            /* Right label: gap of 5px past the far endpoint. */
            int right_x = (int)(cx + half_len + 5.0f);

            /* Left label: right-align it 5px before the near endpoint. */
            int left_x  = (int)(cx - half_len - 5.0f) - nchars * FONT_ADV;

            draw_text_row(row_words, seg_count, y,
                          right_x, label_top, lbl, COLOR_LADDER);
            draw_text_row(row_words, seg_count, y,
                          left_x,  label_top, lbl, COLOR_LADDER);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Frame builder                                                       */
/* ------------------------------------------------------------------ */

void ahrs_display_build_frame(float roll_deg, float pitch_deg,
                              uint32_t frame[VGA_HEIGHT][WORDS_PER_ROW]) {
    const float cx  = (VGA_WIDTH  - 1) * 0.5f;
    const float cy0 = (VGA_HEIGHT - 1) * 0.5f;

    const float theta = ROLL_SIGN * roll_deg * (float)M_PI / 180.0f;
    const float s = sinf(theta);
    const float c = cosf(theta);

    const float cy = cy0 + (PITCH_SIGN * pitch_deg * PIXELS_PER_DEGREE);

    /*
     * Top-bar text strings.
     *
     * Layout (INFO_BAR_HEIGHT = 24 px, FONT_H = 7 px):
     *   Rows  3-9 : bold label  "ROLL"  (left)  |  "PITCH"  (right)
     *   Rows 13-19: bold value  "+45.2" (left)  |  "+10.5"  (right)
     */
    char roll_val[16], pitch_val[16];
    snprintf(roll_val,  sizeof(roll_val),  "%+.1f", roll_deg);
    snprintf(pitch_val, sizeof(pitch_val), "%+.1f", pitch_deg);

    /* Right-edge x for right-aligned text (8 px margin from right edge). */
    const int right_margin = 8;
    int pitch_label_x = VGA_WIDTH - right_margin - (int)(5 * FONT_ADV);       /* "PITCH" */
    int pitch_val_x   = VGA_WIDTH - right_margin
                        - (int)(strlen(pitch_val) * FONT_ADV);

    for(int y = 0; y < VGA_HEIGHT; ++y) {
        /* Clear row. */
        for(int i = 0; i < WORDS_PER_ROW; ++i) frame[y][i] = 0;
        int seg_count = 0;

        /* ---- Top information bar ---- */
        if(y < INFO_BAR_HEIGHT) {
            /*
             * Segments are priority-ordered (index 0 = highest).
             * Append text first so it overwrites the black background.
             */

            /* ROLL label and value on the left, bold white. */
            draw_text_row_bold(frame[y], &seg_count, y,  8,             3,  "ROLL",    COLOR_TEXT);
            draw_text_row_bold(frame[y], &seg_count, y,  8,             13, roll_val,  COLOR_TEXT);

            /* PITCH label and value on the right, bold white. */
            draw_text_row_bold(frame[y], &seg_count, y,  pitch_label_x, 3,  "PITCH",   COLOR_TEXT);
            draw_text_row_bold(frame[y], &seg_count, y,  pitch_val_x,   13, pitch_val, COLOR_TEXT);

            /* Full-width black background — lowest priority fills the rest. */
            append_segment(frame[y], &seg_count, 0, VGA_WIDTH - 1, COLOR_BLACK);
            continue;
        }

        /* ---- Main display area ---- */

        /*
         * 1. Boresight dot — highest priority; always on top.
         *    Fixed 7×7 yellow block at the screen centre regardless of
         *    aircraft attitude.
         */
        if(y >= CENTER_Y - DOT_HALF && y <= CENTER_Y + DOT_HALF) {
            append_segment(frame[y], &seg_count,
                           CENTER_X - DOT_HALF, CENTER_X + DOT_HALF,
                           COLOR_YELLOW);
        }

        /*
         * 2. Pitch ladder lines and their numeric labels.
         *    Drawn before the horizon fill so they sit above the
         *    sky/ground background.
         */
        add_pitch_ladder_rows(frame[y], &seg_count, y, cx, cy0,
                              pitch_deg, s, c);

        /*
         * 3. Horizon fill — lowest priority; fills every unlit pixel
         *    with the appropriate sky / horizon / ground colour.
         */
        add_horizon_row(frame[y], &seg_count, y, cx, cy, s, c);
    }
}

/* ------------------------------------------------------------------ */
/* Hardware interface                                                  */
/* ------------------------------------------------------------------ */

static void *map_peripheral(int fd, off_t phys_base, size_t length) {
    long page_size = sysconf(_SC_PAGESIZE);
    if(page_size <= 0) { perror("sysconf"); exit(1); }
    off_t page_base = phys_base & ~((off_t)page_size - 1);
    off_t page_off  = phys_base - page_base;
    void *map = mmap(NULL, (size_t)page_off + length,
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, page_base);
    if(map == MAP_FAILED) { perror("mmap"); exit(1); }
    return (uint8_t *)map + page_off;
}

static int read_current_display_buffer(volatile uint32_t *vga_words) {
    (void)vga_words;
    return g_display_buffer;
}

static void request_buffer_swap(volatile uint32_t *vga_words, int next_buffer) {
    vga_words[VGA_CTRL_WORD] = (uint32_t)(next_buffer & 0x1);
    g_display_buffer = next_buffer & 0x1;
}

static void write_table_to_vga(volatile uint32_t *vga_words,
                               uint32_t base_word,
                               const uint32_t frame[VGA_HEIGHT][WORDS_PER_ROW]) {
    for(int y = 0; y < VGA_HEIGHT; ++y)
        for(int seg = 0; seg < WORDS_PER_ROW; ++seg)
            vga_words[base_word + (uint32_t)y * WORDS_PER_ROW + (uint32_t)seg]
                = frame[y][seg];
}

void ahrs_display_init(void) {
    if(g_vga_words != NULL) return;
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if(fd < 0) { perror("open(/dev/mem)"); exit(1); }
    g_vga_words = (volatile uint32_t *)
        map_peripheral(fd, (off_t)VGA_PHYS_BASE, VGA_MAP_LEN);
    close(fd);
    g_display_buffer = 0;
}

void ahrs_display_render(uint16_t roll_raw, uint16_t pitch_raw) {
    if(g_vga_words == NULL) ahrs_display_init();

    float roll_deg  = q3_9_to_degrees_roll(roll_raw);
    float pitch_deg = q3_9_to_degrees_pitch(pitch_raw);

    static uint32_t frame[VGA_HEIGHT][WORDS_PER_ROW];

    int current_buffer = read_current_display_buffer(g_vga_words);
    int target_buffer  = current_buffer ^ 1;

    ahrs_display_build_frame(roll_deg, pitch_deg, frame);

    if(target_buffer == 0)
        write_table_to_vga(g_vga_words, VGA_BUF0_WORD, frame);
    else
        write_table_to_vga(g_vga_words, VGA_BUF1_WORD, frame);

    request_buffer_swap(g_vga_words, target_buffer);
}
