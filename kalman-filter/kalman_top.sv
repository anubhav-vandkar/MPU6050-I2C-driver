// =============================================================================
// kalman_top.sv : Top-level dual Kalman filter (roll + pitch)
// -----------------------------------------------------------------------------
// Instantiates two kalman_2x2 cores -- one for roll/gyroX, one for pitch/gyroY.
// A single din_valid pulse triggers both simultaneously.
// dout_valid asserts when BOTH cores have completed (gated AND).
//
// Inputs/outputs are all Q7.9 (16-bit signed).
//
// Designed for DE1-SoC (Cyclone V).  Synthesizes cleanly to ~50 MHz with
// margin.  All arithmetic is saturating; all storage is in Q7.25 internally.
// =============================================================================
`default_nettype none

module kalman_top #(
    // Same parameters propagated to both cores.  Override at instantiation
    // if you want different tuning for roll vs pitch.
    parameter int W       = 32,
    parameter int FRAC    = 25,
    parameter int IO_W    = 16,
    parameter int IO_FRAC = 9,

    parameter signed [W-1:0] DT_Q725    = 32'sd33554,
    parameter signed [W-1:0] Q00_Q725   = 32'sd33554,
    parameter signed [W-1:0] Q11_Q725   = 32'sd100663,
    parameter signed [W-1:0] R00_Q725   = 32'sd1006633,
    parameter signed [W-1:0] R11_Q725   = 32'sd1006633,
    parameter signed [W-1:0] P0_00_Q725 = 32'sd33554432,
    parameter signed [W-1:0] P0_11_Q725 = 32'sd33554432
) (
    input  wire                       clk,
    input  wire                       rst_n,

    input  wire                       din_valid,
    input  wire signed [IO_W-1:0]     roll_in,        // Q7.9 accel-derived roll
    input  wire signed [IO_W-1:0]     pitch_in,       // Q7.9 accel-derived pitch
    input  wire signed [IO_W-1:0]     gx_in,          // Q7.9 gyro X (rad/s)
    input  wire signed [IO_W-1:0]     gy_in,          // Q7.9 gyro Y (rad/s)

    output wire signed [IO_W-1:0]     est_roll,
    output wire signed [IO_W-1:0]     est_pitch,
    output wire                       dout_valid,
    output wire                       busy
);

    wire roll_dvalid, pitch_dvalid;
    wire roll_busy,   pitch_busy;

    // Roll Kalman core
    kalman_2x2 #(
        .W(W), .FRAC(FRAC), .IO_W(IO_W), .IO_FRAC(IO_FRAC),
        .DT_Q725(DT_Q725),
        .Q00_Q725(Q00_Q725), .Q11_Q725(Q11_Q725),
        .R00_Q725(R00_Q725), .R11_Q725(R11_Q725),
        .P0_00_Q725(P0_00_Q725), .P0_11_Q725(P0_11_Q725)
    ) u_roll (
        .clk(clk), .rst_n(rst_n),
        .din_valid(din_valid & ~busy),       // ignore new starts while busy
        .ang_meas_q9 (roll_in),
        .gyro_meas_q9(gx_in),
        .est_angle_q9(est_roll),
        .dout_valid  (roll_dvalid),
        .busy        (roll_busy)
    );

    // Pitch Kalman core
    kalman_2x2 #(
        .W(W), .FRAC(FRAC), .IO_W(IO_W), .IO_FRAC(IO_FRAC),
        .DT_Q725(DT_Q725),
        .Q00_Q725(Q00_Q725), .Q11_Q725(Q11_Q725),
        .R00_Q725(R00_Q725), .R11_Q725(R11_Q725),
        .P0_00_Q725(P0_00_Q725), .P0_11_Q725(P0_11_Q725)
    ) u_pitch (
        .clk(clk), .rst_n(rst_n),
        .din_valid(din_valid & ~busy),
        .ang_meas_q9 (pitch_in),
        .gyro_meas_q9(gy_in),
        .est_angle_q9(est_pitch),
        .dout_valid  (pitch_dvalid),
        .busy        (pitch_busy)
    );

    assign busy       = roll_busy | pitch_busy;
    // Both cores run identical FSMs in lock-step, so dout_valid pulses align.
    // AND them anyway for robustness.
    assign dout_valid = roll_dvalid & pitch_dvalid;

endmodule

`default_nettype wire
