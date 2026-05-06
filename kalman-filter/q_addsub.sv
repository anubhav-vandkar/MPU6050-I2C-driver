// =============================================================================
// q_addsub.sv : Q7.25 signed adder / subtractor with saturation
// -----------------------------------------------------------------------------
// sub = 0 -> y = a + b
// sub = 1 -> y = a - b
// Combinational.
// =============================================================================
`default_nettype none

module q_addsub #(
    parameter int W = 32
) (
    input  wire                sub,
    input  wire signed [W-1:0] a,
    input  wire signed [W-1:0] b,
    output reg  signed [W-1:0] y
);

    // Use one extra bit to detect overflow cleanly
    wire signed [W:0] a_ext = {a[W-1], a};
    wire signed [W:0] b_ext = {b[W-1], b};
    wire signed [W:0] sum   = sub ? (a_ext - b_ext) : (a_ext + b_ext);

    localparam signed [W:0] MAX_POS = {1'b0, {1'b0, {(W-1){1'b1}}}}; //  +max
    localparam signed [W:0] MAX_NEG = {1'b1, {1'b1, {(W-1){1'b0}}}}; //  -max-1 sign-extended

    always @* begin
        if (sum > $signed({2'b00, {(W-1){1'b1}}}))
            y = {1'b0, {(W-1){1'b1}}};            // 0x7FFF_FFFF
        else if (sum < $signed({2'b11, {(W-1){1'b0}}}))
            y = {1'b1, {(W-1){1'b0}}};            // 0x8000_0000
        else
            y = sum[W-1:0];
    end

endmodule

`default_nettype wire
