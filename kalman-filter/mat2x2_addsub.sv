// =============================================================================
// mat2x2_addsub.sv : 2x2 matrix add or subtract (Q7.25)
// sub=0 -> M + N ; sub=1 -> M - N
// Combinational.
// =============================================================================
`default_nettype none

module mat2x2_addsub #(
    parameter int W = 32
) (
    input  wire                sub,
    input  wire signed [W-1:0] m00, m01, m10, m11,
    input  wire signed [W-1:0] n00, n01, n10, n11,
    output wire signed [W-1:0] p00, p01, p10, p11
);
    q_addsub #(.W(W)) u_00 (.sub(sub), .a(m00), .b(n00), .y(p00));
    q_addsub #(.W(W)) u_01 (.sub(sub), .a(m01), .b(n01), .y(p01));
    q_addsub #(.W(W)) u_10 (.sub(sub), .a(m10), .b(n10), .y(p10));
    q_addsub #(.W(W)) u_11 (.sub(sub), .a(m11), .b(n11), .y(p11));
endmodule

`default_nettype wire
