// =============================================================================
// mat2x2_inv.sv : 2x2 matrix inverse (Q7.25)
// -----------------------------------------------------------------------------
//   M = | a b |    M^-1 = (1/det) * |  d -b |
//       | c d |                     | -c  a |
//   det = a*d - b*c
//
// Multi-cycle: ~8 cycles (1 to compute det, 7 for reciprocal, 1 to multiply
// the cofactor matrix by 1/det).  Uses one q_recip and four q_mult/q_addsub.
//
// Interface:
//   start  : 1-cycle pulse to begin
//   done   : 1-cycle pulse when inv00..inv11 are valid
//   busy   : high while computing
// =============================================================================
`default_nettype none

module mat2x2_inv #(
    parameter int W    = 32,
    parameter int FRAC = 25
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                start,
    input  wire signed [W-1:0] m00, m01, m10, m11,
    output reg  signed [W-1:0] inv00, inv01, inv10, inv11,
    output reg                 done
);

    // ---- Stage 1: compute det = m00*m11 - m01*m10 ------------------------
    wire signed [W-1:0] ad, bc, det_w;
    q_mult   #(.W(W), .FRAC(FRAC)) u_ad  (.a(m00), .b(m11), .p(ad));
    q_mult   #(.W(W), .FRAC(FRAC)) u_bc  (.a(m01), .b(m10), .p(bc));
    q_addsub #(.W(W))              u_det (.sub(1'b1), .a(ad), .b(bc), .y(det_w));

    // ---- Reciprocal of det ------------------------------------------------
    // NOTE: pass det combinationally to q_recip so that on the same cycle
    // recip_start is asserted, the correct det value is sampled.  Latching
    // det into det_r and then feeding it would create a 1-cycle race where
    // q_recip would see the OLD det_r value.
    reg                 recip_start;
    wire signed [W-1:0] inv_det;
    wire                recip_done;

    q_recip #(.W(W), .FRAC(FRAC)) u_recip (
        .clk(clk), .rst_n(rst_n),
        .start(recip_start),
        .d(det_w),               // combinational from m* inputs
        .y(inv_det),
        .done(recip_done)
    );

    // ---- Stage 3: multiply cofactor matrix by inv_det ---------------------
    // Cofactor matrix: |  m11 -m01 |
    //                  | -m10  m00 |
    reg  signed [W-1:0] cof00, cof01, cof10, cof11;
    wire signed [W-1:0] r00, r01, r10, r11;
    q_mult #(.W(W), .FRAC(FRAC)) u_r00 (.a(cof00), .b(inv_det), .p(r00));
    q_mult #(.W(W), .FRAC(FRAC)) u_r01 (.a(cof01), .b(inv_det), .p(r01));
    q_mult #(.W(W), .FRAC(FRAC)) u_r10 (.a(cof10), .b(inv_det), .p(r10));
    q_mult #(.W(W), .FRAC(FRAC)) u_r11 (.a(cof11), .b(inv_det), .p(r11));

    // ---- FSM --------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE,
        S_RECIP,
        S_MULT,
        S_DONE
    } state_t;

    state_t state, nstate;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= nstate;
    end

    always @* begin
        nstate      = state;
        recip_start = 1'b0;
        case (state)
            S_IDLE : if (start)        nstate = S_RECIP;
            S_RECIP: if (recip_done)   nstate = S_MULT;
            S_MULT :                   nstate = S_DONE;
            S_DONE :                   nstate = S_IDLE;
            default:                   nstate = S_IDLE;
        endcase

        // Pulse recip_start in the first cycle of S_RECIP.
        // We achieve that by asserting it in S_IDLE while transitioning.
        if (state == S_IDLE && start) recip_start = 1'b1;
    end

    // ---- Datapath registers ------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cof00  <= '0; cof01 <= '0; cof10 <= '0; cof11 <= '0;
            inv00  <= '0; inv01 <= '0; inv10 <= '0; inv11 <= '0;
            done   <= 1'b0;
        end
        else begin
            done <= 1'b0;

            if (state == S_IDLE && start) begin
                // Latch cofactor matrix on start.
                // (det is fed combinationally to q_recip; no need to register it here.)
                cof00  <=  m11;
                cof01  <= -m01;
                cof10  <= -m10;
                cof11  <=  m00;
            end

            if (state == S_MULT) begin
                // Multiplications are combinational off cof* and inv_det,
                // both of which are stable now.  Capture results.
                inv00 <= r00;
                inv01 <= r01;
                inv10 <= r10;
                inv11 <= r11;
            end

            if (state == S_MULT && nstate == S_DONE) begin
                done <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
