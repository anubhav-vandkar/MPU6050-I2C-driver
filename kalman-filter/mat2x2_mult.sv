// =============================================================================
// mat2x2_mult.sv : 2x2 * 2x2 matrix multiplier (Q7.25)
// -----------------------------------------------------------------------------
//   |p00 p01|   |m00 m01|   |n00 n01|
//   |p10 p11| = |m10 m11| * |n10 n11|
//
// p00 = m00*n00 + m01*n10
// p01 = m00*n01 + m01*n11
// p10 = m10*n00 + m11*n10
// p11 = m10*n01 + m11*n11
//
// Uses 8 q_mult and 4 q_addsub in parallel -> single-cycle combinational.
// =============================================================================
`default_nettype none

module mat2x2_mult #(
    parameter int W    = 32,
    parameter int FRAC = 25
) (
    input  wire signed [W-1:0] m00, m01, m10, m11,
    input  wire signed [W-1:0] n00, n01, n10, n11,
    output wire signed [W-1:0] p00, p01, p10, p11
);

    wire signed [W-1:0] m00n00, m01n10;
    wire signed [W-1:0] m00n01, m01n11;
    wire signed [W-1:0] m10n00, m11n10;
    wire signed [W-1:0] m10n01, m11n11;

    q_mult #(.W(W), .FRAC(FRAC)) u_m00n00 (.a(m00), .b(n00), .p(m00n00));
    q_mult #(.W(W), .FRAC(FRAC)) u_m01n10 (.a(m01), .b(n10), .p(m01n10));
    q_mult #(.W(W), .FRAC(FRAC)) u_m00n01 (.a(m00), .b(n01), .p(m00n01));
    q_mult #(.W(W), .FRAC(FRAC)) u_m01n11 (.a(m01), .b(n11), .p(m01n11));
    q_mult #(.W(W), .FRAC(FRAC)) u_m10n00 (.a(m10), .b(n00), .p(m10n00));
    q_mult #(.W(W), .FRAC(FRAC)) u_m11n10 (.a(m11), .b(n10), .p(m11n10));
    q_mult #(.W(W), .FRAC(FRAC)) u_m10n01 (.a(m10), .b(n01), .p(m10n01));
    q_mult #(.W(W), .FRAC(FRAC)) u_m11n11 (.a(m11), .b(n11), .p(m11n11));

    q_addsub #(.W(W)) u_p00 (.sub(1'b0), .a(m00n00), .b(m01n10), .y(p00));
    q_addsub #(.W(W)) u_p01 (.sub(1'b0), .a(m00n01), .b(m01n11), .y(p01));
    q_addsub #(.W(W)) u_p10 (.sub(1'b0), .a(m10n00), .b(m11n10), .y(p10));
    q_addsub #(.W(W)) u_p11 (.sub(1'b0), .a(m10n01), .b(m11n11), .y(p11));

endmodule

`default_nettype wire
