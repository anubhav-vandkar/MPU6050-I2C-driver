#define _GNU_SOURCE

#include <stdio.h>
#include <math.h>
#include <unistd.h>

#include "fpga_vga.h"

#define DEG_TO_RAD (3.14159265f / 180.0f)
#define Q39_SCALE 512.0f

int main(void)
{
    if (fpga_vga_open() < 0)
        return 1;

    fpga_vga_set_background(0x00, 0x00, 0x80);

    printf("Sweeping roll 0->180->0 deg, pitch 0->90->0 deg\n");
    printf("Each step holds for 50ms. Ctrl+C to stop.\n\n");

    while (1) {
        /* roll: 0 -> 180 degrees */
        for (int roll_deg = 0; roll_deg <= 180; roll_deg += 2) {

            /* pitch mirrors roll but only goes 0->90 */
            int pitch_deg = (roll_deg <= 90) ? roll_deg : (180 - roll_deg);

            float roll_rad  =  roll_deg  * DEG_TO_RAD;
            float pitch_rad =  pitch_deg * DEG_TO_RAD;

            /* encode as Q3.9 */
            int16_t roll_q39  = (int16_t)((int16_t)(roll_rad  * Q39_SCALE) & 0x0FFF);
            int16_t pitch_q39 = (int16_t)((int16_t)(pitch_rad * Q39_SCALE) & 0x0FFF);

            kalman_result_t kalman_result = {
                .kalman_roll = roll_q39,
                .kalman_pitch = pitch_q39
            };

            fpga_vga_update(&kalman_result);

            printf("roll=%4d deg  pitch=%4d deg  "
                   "roll_q39=%5d  pitch_q39=%5d\n",
                   roll_deg, pitch_deg, roll_q39, pitch_q39);

            usleep(50000);   /* 50 ms per step */
        }

        /* roll: 180 -> 0 degrees (sweep back) */
        for (int roll_deg = 180; roll_deg >= 0; roll_deg -= 2) {

            int pitch_deg = (roll_deg <= 90) ? roll_deg : (180 - roll_deg);

            float roll_rad  = roll_deg  * DEG_TO_RAD;
            float pitch_rad = pitch_deg * DEG_TO_RAD;

            int16_t roll_q39  = (int16_t)((int16_t)(roll_rad  * Q39_SCALE) & 0x0FFF);
            int16_t pitch_q39 = (int16_t)((int16_t)(pitch_rad * Q39_SCALE) & 0x0FFF);

            kalman_result_t kalman_result = {
                .kalman_roll = roll_q39,
                .kalman_pitch = pitch_q39
            };

            fpga_vga_update(&kalman_result);

            printf("roll=%4d deg  pitch=%4d deg  "
                   "roll_q39=%5d  pitch_q39=%5d\n",
                   roll_deg, pitch_deg, roll_q39, pitch_q39);

            usleep(50000);
        }
    }

    fpga_vga_close();
    return 0;
}