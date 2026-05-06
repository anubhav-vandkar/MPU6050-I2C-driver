// =============================================================================
// q_mult.sv : Q7.25 signed multiplier with saturation
// -----------------------------------------------------------------------------
// Inputs  : a, b -- 32-bit signed Q7.25
// Output  : p    -- 32-bit signed Q7.25, saturated on overflow
// Latency : combinational  (will infer a DSP slice; user may register externally)
// =============================================================================
`default_nettype none

module q_mult #(
    parameter int W       = 32,
    parameter int FRAC    = 25
) (
    input  wire signed [W-1:0] a,
    input  wire signed [W-1:0] b,
    output reg  signed [W-1:0] p
);

    // 64-bit full-precision product
    wire signed [2*W-1:0] full = a * b;
    // Shift right by FRAC to bring back to Q7.25
    wire signed [2*W-1:0] shifted = full >>> FRAC;

    // Saturation:
    //   The valid signed range for the W-bit result is
    //   [-2^(W-1) .. 2^(W-1)-1].  If shifted exceeds those bounds,
    //   clip to MAX_POS or MAX_NEG.
    localparam signed [2*W-1:0] MAX_POS = (1 <<< (W-1)) - 1;       //  0x7FFFFFFF
    localparam signed [2*W-1:0] MAX_NEG = -(1 <<< (W-1));          //  0x80000000

    always @* begin
        if (shifted > MAX_POS)
            p = MAX_POS[W-1:0];
        else if (shifted < MAX_NEG)
            p = MAX_NEG[W-1:0];
        else
            p = shifted[W-1:0];
    end

endmodule

`default_nettype wire
