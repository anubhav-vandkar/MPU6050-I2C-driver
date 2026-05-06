// =============================================================================
// q_recip.sv : Q7.25 signed reciprocal  y = 1 / d   (d > 0 expected)
// -----------------------------------------------------------------------------
// Algorithm
//   1. Take absolute value (d here is always positive in our use, but be safe).
//   2. Normalize: find shift `k` so that d_norm = d << k has its MSB-of-magnitude
//      at bit position FRAC-1 (i.e. d_norm in [0.5, 1.0) in Q7.25).
//      For Q7.25: target MSB position = 24 (bit weight 0.5).
//   3. Seed:  x0 = 48/17 - 32/17 * d_norm   (classical NR seed, error <= 6%)
//             implemented as constants in Q7.25.
//   4. Iterate x_{n+1} = x_n * (2 - d_norm * x_n)  three times.
//      With seed error 6%, three iterations reach ~ 1e-7 relative error -- plenty.
//   5. Denormalize: y = x_final << k    (because 1/(d>>k) = (1/d) << k effectively;
//      we shifted d *up* by k to normalize it, so the reciprocal must be shifted
//      *up* by k as well to undo the normalization on 1/d_norm = 1/(d<<k) = (1/d)>>k).
//      Wait -- careful.  Let me re-derive once more.
//
//      d_norm = d * 2^k    (we left-shifted d by k bits).
//      x_final ~= 1 / d_norm = 1 / (d * 2^k) = (1/d) * 2^(-k).
//      So  1/d = x_final * 2^k = x_final << k.
//      Correct.
//
// Pipeline: 7 cycles (1 normalize, 1 seed, 3*2 iterate, 1 denormalize).
// `start` pulses for one cycle when `d` is valid; `done` pulses for one cycle
// when `y` is valid.
// =============================================================================
`default_nettype none

module q_recip #(
    parameter int W    = 32,
    parameter int FRAC = 25
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                start,
    input  wire signed [W-1:0] d,
    output reg  signed [W-1:0] y,
    output reg                 done
);

    // ---- Constants in Q7.25 -----------------------------------------------
    // 48/17 = 2.82352941...  -> Q7.25 = round(2.82352941 * 2^25) = 94732479
    // 32/17 = 1.88235294...  -> Q7.25 = round(1.88235294 * 2^25) = 63154986
    // 2.0                    -> Q7.25 = 2 << 25 = 67108864
    localparam signed [W-1:0] C_48_17 = 32'sd94732479;
    localparam signed [W-1:0] C_32_17 = 32'sd63154986;
    localparam signed [W-1:0] C_TWO   = 32'sd67108864;

    // ---- Pipeline registers -----------------------------------------------
    reg                  v0, v1, v2, v3, v4, v5, v6;
    reg signed [W-1:0]   d_norm_r;
    reg [5:0]            shift_r;
    reg                  neg_r;        // d was negative (shouldn't happen but guard)
    reg signed [W-1:0]   x_r;          // current NR estimate

    // ---- Stage 0 : compute |d| and find normalization shift ---------------
    wire signed [W-1:0] d_abs = d[W-1] ? -d : d;

    // Find leading-one position of d_abs.  Result `lead_pos` is in [0..W-2].
    // For Q7.25 the value 1.0 has bit 25 set.  We want d_norm with leading
    // bit at position FRAC-1 = 24 (i.e. d_norm in [0.5,1.0) Q7.25).
    // Therefore   shift = 24 - lead_pos
    //   if lead_pos <= 24, shift left by  (24 - lead_pos)
    //   if lead_pos >  24, shift right by (lead_pos - 24)
    integer i;
    reg [5:0] lead_pos;
    always @* begin
        lead_pos = 6'd0;
        for (i = 0; i < W-1; i = i + 1)
            if (d_abs[i]) lead_pos = i[5:0];
    end

    wire        shift_left  = (lead_pos <= 6'd24);
    wire [5:0]  shift_amt   = shift_left ? (6'd24 - lead_pos) : (lead_pos - 6'd24);
    wire signed [W-1:0] d_norm_w = shift_left ? (d_abs <<< shift_amt)
                                              : (d_abs >>> shift_amt);

    // ---- NR iteration combinational helpers -------------------------------
    // one NR step: x_next = x * (2 - d_norm * x)
    // We instantiate q_mult/q_addsub to share the saturation logic.
    // To stay strictly pipelined and simple, we do ONE NR iteration per cycle.
    wire signed [W-1:0] dx;       // d_norm * x_current
    wire signed [W-1:0] two_dx;   // 2 - d_norm*x
    wire signed [W-1:0] x_next;   // x * (2 - d_norm*x)

    q_mult   #(.W(W), .FRAC(FRAC)) u_dx     (.a(d_norm_r), .b(x_r),     .p(dx));
    q_addsub #(.W(W))              u_two_dx (.sub(1'b1), .a(C_TWO), .b(dx), .y(two_dx));
    q_mult   #(.W(W), .FRAC(FRAC)) u_xn     (.a(x_r),     .b(two_dx),  .p(x_next));

    // Seed combinational: x0 = 48/17 - (32/17) * d_norm
    wire signed [W-1:0] seed_mul;
    wire signed [W-1:0] seed_val;
    q_mult   #(.W(W), .FRAC(FRAC)) u_seed_m (.a(C_32_17), .b(d_norm_w), .p(seed_mul));
    q_addsub #(.W(W))              u_seed_a (.sub(1'b1), .a(C_48_17), .b(seed_mul), .y(seed_val));

    // ---- Pipeline FSM : state counter --------------------------------------
    // Stages:
    //  s0: capture d, compute d_norm, seed x  (fires on start)
    //  s1: NR iter 1
    //  s2: NR iter 2
    //  s3: NR iter 3
    //  s4: denormalize, output
    reg [2:0] stage;
    reg       busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage    <= 3'd0;
            busy     <= 1'b0;
            done     <= 1'b0;
            d_norm_r <= '0;
            shift_r  <= 6'd0;
            neg_r    <= 1'b0;
            x_r      <= '0;
            y        <= '0;
        end
        else begin
            done <= 1'b0;

            if (!busy && start) begin
                d_norm_r <= d_norm_w;
                shift_r  <= shift_amt;
                neg_r    <= shift_left;   // remember which direction we shifted d
                x_r      <= seed_val;     // seed x0
                stage    <= 3'd1;
                busy     <= 1'b1;
            end
            else if (busy) begin
                case (stage)
                    3'd1, 3'd2, 3'd3: begin
                        x_r   <= x_next;     // NR iteration
                        stage <= stage + 3'd1;
                    end
                    3'd4: begin
                        // Denormalize: 1/d = x_final * 2^(shift)
                        // If we left-shifted d by shift_amt to normalize, then
                        // 1/d = x_final << shift_amt.  Else (right-shifted d),
                        // 1/d = x_final >> shift_amt.
                        // Saturate on left-shift overflow.
                        if (neg_r) begin
                            // x_r is positive (reciprocal of positive d).
                            // Check if shifting left would overflow:
                            //   overflow if any of the upper `shift_r+1` bits of x_r are set.
                            // Simple check: compare x_r against (MAX_POS >> shift_r).
                            if (x_r > ($signed({1'b0, {(W-1){1'b1}}}) >>> shift_r))
                                y <= {1'b0, {(W-1){1'b1}}};   // +max
                            else
                                y <= x_r <<< shift_r;
                        end
                        else
                            y <= x_r >>> shift_r;
                        done  <= 1'b1;
                        busy  <= 1'b0;
                        stage <= 3'd0;
                    end
                    default: begin
                        stage <= 3'd0;
                        busy  <= 1'b0;
                    end
                endcase
            end
        end
    end

endmodule

`default_nettype wire
