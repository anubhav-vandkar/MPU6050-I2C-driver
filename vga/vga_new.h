#ifndef VGA_NEW_H
#define VGA_NEW_H

#include <stdint.h>

enum {
    VGA_WIDTH      = 640,
    VGA_HEIGHT     = 480,
    WORDS_PER_ROW  = 16
};

void ahrs_display_init(void);

void ahrs_display_build_frame(float roll_deg, float pitch_deg, uint32_t frame[VGA_HEIGHT][WORDS_PER_ROW]);

void ahrs_display_render(uint16_t roll_raw, uint16_t pitch_raw);

#endif /* VGA_NEW_H */