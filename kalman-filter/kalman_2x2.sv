// =============================================================================
// kalman_2x2.sv : 2-state Kalman filter for one axis (roll OR pitch).
// -----------------------------------------------------------------------------
// State vector  X = [angle ; gyro_rate]
// A = [1 dt ; 0 1]                    (state transition)
// H = I                               (measurement matrix)
// Q = diag(Q00, Q11)                  (process noise)
// R = diag(R00, R11)                  (sensor noise)
// Y = [angle_meas ; gyro_meas]
//
// All I/O scalar values are Q7.9 (16 bits, signed).
// All internal storage is Q7.25 (32 bits, signed).
//
// Conversion:
//   Q7.9 -> Q7.25 :  sign-extend to 32, then left-shift by (25-9)=16.
//   Q7.25 -> Q7.9 :  arithmetic right-shift by 16, with saturation to 16-bit.
//
// Handshake:
//   din_valid  : 1-cycle pulse to start one Kalman iteration.
//                Inputs ang_meas_q9, gyro_meas_q9 must be stable when pulsed.
//   dout_valid : 1-cycle pulse when est_angle_q9 is updated.
//   busy       : high while computation is in progress (also gates din_valid).
//
// =============================================================================
`default_nettype none

module kalman_2x2 #(
    parameter int W      = 32,
    parameter int FRAC   = 25,
    parameter int IO_W   = 16,
    parameter int IO_FRAC= 9,

    // Process model
    parameter signed [W-1:0] DT_Q725   = 32'sd33554,    // 0.001 s in Q7.25 (=1e-3 * 2^25)

    // Process noise covariance Q (diagonal)
    parameter signed [W-1:0] Q00_Q725  = 32'sd33554,    // 0.001 in Q7.25
    parameter signed [W-1:0] Q11_Q725  = 32'sd100663,   // 0.003 in Q7.25

    // Sensor noise covariance R (diagonal)
    parameter signed [W-1:0] R00_Q725  = 32'sd1006633,  // 0.03 in Q7.25
    parameter signed [W-1:0] R11_Q725  = 32'sd1006633,  // 0.03 in Q7.25

    // Initial covariance P0 (diagonal)
    parameter signed [W-1:0] P0_00_Q725 = 32'sd33554432, // 1.0 in Q7.25
    parameter signed [W-1:0] P0_11_Q725 = 32'sd33554432  // 1.0 in Q7.25
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       din_valid,
    input  wire signed [IO_W-1:0]     ang_meas_q9,    // Q7.9 measured angle (radians)
    input  wire signed [IO_W-1:0]     gyro_meas_q9,   // Q7.9 measured gyro (rad/s)
    output reg  signed [IO_W-1:0]     est_angle_q9,
    output reg                        dout_valid,
    output wire                       busy
);

    // ---- Q7.9 <-> Q7.25 conversion helpers --------------------------------
    function automatic signed [W-1:0] q9_to_q25 (input signed [IO_W-1:0] x);
        // sign-extend then shift left by (FRAC - IO_FRAC) = 16
        q9_to_q25 = {{(W-IO_W){x[IO_W-1]}}, x} <<< (FRAC - IO_FRAC);
    endfunction

    function automatic signed [IO_W-1:0] q25_to_q9 (input signed [W-1:0] x);
        // arithmetic right shift, saturating to IO_W-bit signed
        logic signed [W-1:0] sh;
        logic signed [IO_W-1:0] MAX_P, MAX_N;
        begin
            sh = x >>> (FRAC - IO_FRAC);
            MAX_P = {1'b0, {(IO_W-1){1'b1}}};   // +max
            MAX_N = {1'b1, {(IO_W-1){1'b0}}};   // -max-1
            if (sh > $signed({{(W-IO_W){1'b0}}, MAX_P}))
                q25_to_q9 = MAX_P;
            else if (sh < $signed({{(W-IO_W){1'b1}}, MAX_N}))
                q25_to_q9 = MAX_N;
            else
                q25_to_q9 = sh[IO_W-1:0];
        end
    endfunction

    // ---- State storage ----------------------------------------------------
    reg signed [W-1:0] X0, X1;                 // X_prev (kept across iterations)
    reg signed [W-1:0] P00, P01, P10, P11;     // P_prev

    // Working registers (intermediate matrices)
    reg signed [W-1:0] Xkp0, Xkp1;             // predicted state
    reg signed [W-1:0] Pkp00, Pkp01, Pkp10, Pkp11;
    reg signed [W-1:0] S00, S01, S10, S11;     // innovation cov
    reg signed [W-1:0] Sinv00, Sinv01, Sinv10, Sinv11;
    reg signed [W-1:0] K00, K01, K10, K11;     // Kalman gain
    reg signed [W-1:0] Y0, Y1;                 // measurement (latched)
    reg signed [W-1:0] innov0, innov1;         // residual (Y - HX_kp)
    reg signed [W-1:0] Knew0, Knew1;           // K * innov
    reg signed [W-1:0] IK00, IK01, IK10, IK11; // (I - K)
    reg signed [W-1:0] Pnew00, Pnew01, Pnew10, Pnew11;

    // ---- Shared 2x2 matrix multiplier (combinational) ---------------------
    // Operands selected by FSM mux.
    reg  signed [W-1:0] mm_a00, mm_a01, mm_a10, mm_a11;
    reg  signed [W-1:0] mm_b00, mm_b01, mm_b10, mm_b11;
    wire signed [W-1:0] mm_p00, mm_p01, mm_p10, mm_p11;

    mat2x2_mult #(.W(W), .FRAC(FRAC)) u_mm (
        .m00(mm_a00), .m01(mm_a01), .m10(mm_a10), .m11(mm_a11),
        .n00(mm_b00), .n01(mm_b01), .n10(mm_b10), .n11(mm_b11),
        .p00(mm_p00), .p01(mm_p01), .p10(mm_p10), .p11(mm_p11)
    );

    // ---- Shared 2x2 add/sub (combinational) -------------------------------
    reg                  ms_sub;
    reg  signed [W-1:0]  ms_a00, ms_a01, ms_a10, ms_a11;
    reg  signed [W-1:0]  ms_b00, ms_b01, ms_b10, ms_b11;
    wire signed [W-1:0]  ms_p00, ms_p01, ms_p10, ms_p11;

    mat2x2_addsub #(.W(W)) u_ms (
        .sub(ms_sub),
        .m00(ms_a00), .m01(ms_a01), .m10(ms_a10), .m11(ms_a11),
        .n00(ms_b00), .n01(ms_b01), .n10(ms_b10), .n11(ms_b11),
        .p00(ms_p00), .p01(ms_p01), .p10(ms_p10), .p11(ms_p11)
    );

    // ---- 2x2 inverse engine -----------------------------------------------
    reg                  inv_start;
    reg  signed [W-1:0]  inv_in00, inv_in01, inv_in10, inv_in11;
    wire signed [W-1:0]  inv_out00, inv_out01, inv_out10, inv_out11;
    wire                 inv_done;

    mat2x2_inv #(.W(W), .FRAC(FRAC)) u_inv (
        .clk(clk), .rst_n(rst_n),
        .start(inv_start),
        .m00(inv_in00), .m01(inv_in01), .m10(inv_in10), .m11(inv_in11),
        .inv00(inv_out00), .inv01(inv_out01), .inv10(inv_out10), .inv11(inv_out11),
        .done(inv_done)
    );

    // ---- A and A^T constants (built from DT) ------------------------------
    // A   = | 1   DT |    A^T = | 1   0  |
    //       | 0   1  |          | DT  1  |
    localparam signed [W-1:0] ONE_Q725 = 32'sd33554432;   // 1.0 in Q7.25
    wire signed [W-1:0] A00 = ONE_Q725;
    wire signed [W-1:0] A01 = DT_Q725;
    wire signed [W-1:0] A10 = '0;
    wire signed [W-1:0] A11 = ONE_Q725;
    wire signed [W-1:0] AT00 = ONE_Q725;
    wire signed [W-1:0] AT01 = '0;
    wire signed [W-1:0] AT10 = DT_Q725;
    wire signed [W-1:0] AT11 = ONE_Q725;

    // ---- FSM --------------------------------------------------------------
    typedef enum logic [4:0] {
        S_IDLE,
        S_LATCH,        // capture measurements; convert Q7.9 -> Q7.25
        S_PRED_X,       // X_kp = A * X_prev   (treat X as 2x2 with col2=0; just compute angle = X0 + DT*X1, rate = X1)
        S_PRED_P1,      // tmp = A * P_prev
        S_PRED_P2,      // P_kp = tmp * A^T
        S_PRED_P3,      // P_kp = P_kp + Q
        S_INNOV_S,      // S = P_kp + R
        S_INV_START,    // pulse inv_start
        S_INV_WAIT,     // wait for inv_done
        S_GAIN,         // K = P_kp * Sinv
        S_RESID,        // innov = Y - X_kp
        S_KINNOV,       // Knew = K * innov   (treating innov as a col vector)
        S_UPD_X,        // X_new = X_kp + Knew
        S_IK,           // IK = I - K
        S_UPD_P,        // P_new = IK * P_kp
        S_COMMIT,       // X_prev <= X_new ; P_prev <= P_new ; output
        S_DONE
    } state_t;

    state_t state, nstate;

    assign busy = (state != S_IDLE);

    // ---- FSM transition ---------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= nstate;
    end

    always @* begin
        nstate = state;
        case (state)
            S_IDLE      : if (din_valid)  nstate = S_LATCH;
            S_LATCH     :                 nstate = S_PRED_X;
            S_PRED_X    :                 nstate = S_PRED_P1;
            S_PRED_P1   :                 nstate = S_PRED_P2;
            S_PRED_P2   :                 nstate = S_PRED_P3;
            S_PRED_P3   :                 nstate = S_INNOV_S;
            S_INNOV_S   :                 nstate = S_INV_START;
            S_INV_START :                 nstate = S_INV_WAIT;
            S_INV_WAIT  : if (inv_done)   nstate = S_GAIN;
            S_GAIN      :                 nstate = S_RESID;
            S_RESID     :                 nstate = S_KINNOV;
            S_KINNOV    :                 nstate = S_UPD_X;
            S_UPD_X     :                 nstate = S_IK;
            S_IK        :                 nstate = S_UPD_P;
            S_UPD_P     :                 nstate = S_COMMIT;
            S_COMMIT    :                 nstate = S_DONE;
            S_DONE      :                 nstate = S_IDLE;
            default     :                 nstate = S_IDLE;
        endcase
    end

    // ---- Operand muxing for shared mat2x2_mult / mat2x2_addsub ------------
    always @* begin
        // defaults: tie low to keep synthesis happy
        mm_a00='0; mm_a01='0; mm_a10='0; mm_a11='0;
        mm_b00='0; mm_b01='0; mm_b10='0; mm_b11='0;
        ms_sub = 1'b0;
        ms_a00='0; ms_a01='0; ms_a10='0; ms_a11='0;
        ms_b00='0; ms_b01='0; ms_b10='0; ms_b11='0;

        case (state)
            S_PRED_P1: begin
                // tmp = A * P_prev   (will be captured into Pkp)
                mm_a00=A00; mm_a01=A01; mm_a10=A10; mm_a11=A11;
                mm_b00=P00; mm_b01=P01; mm_b10=P10; mm_b11=P11;
            end
            S_PRED_P2: begin
                // Pkp = Pkp_tmp * A^T
                mm_a00=Pkp00; mm_a01=Pkp01; mm_a10=Pkp10; mm_a11=Pkp11;
                mm_b00=AT00;  mm_b01=AT01;  mm_b10=AT10;  mm_b11=AT11;
            end
            S_PRED_P3: begin
                // Pkp = Pkp + Q   (Q diagonal)
                ms_sub = 1'b0;
                ms_a00=Pkp00; ms_a01=Pkp01; ms_a10=Pkp10; ms_a11=Pkp11;
                ms_b00=Q00_Q725; ms_b01='0; ms_b10='0; ms_b11=Q11_Q725;
            end
            S_INNOV_S: begin
                // S = Pkp + R
                ms_sub = 1'b0;
                ms_a00=Pkp00; ms_a01=Pkp01; ms_a10=Pkp10; ms_a11=Pkp11;
                ms_b00=R00_Q725; ms_b01='0; ms_b10='0; ms_b11=R11_Q725;
            end
            S_GAIN: begin
                // K = Pkp * Sinv     (H = I)
                mm_a00=Pkp00; mm_a01=Pkp01; mm_a10=Pkp10; mm_a11=Pkp11;
                mm_b00=Sinv00; mm_b01=Sinv01; mm_b10=Sinv10; mm_b11=Sinv11;
            end
            S_RESID: begin
                // innov = Y - X_kp   (column vector treated via 2x2 with second col = 0)
                ms_sub = 1'b1;
                ms_a00=Y0;   ms_a01='0; ms_a10=Y1;   ms_a11='0;
                ms_b00=Xkp0; ms_b01='0; ms_b10=Xkp1; ms_b11='0;
            end
            S_KINNOV: begin
                // Knew = K * innov   (innov as 2x1 -> pack into col 0 of N, col 1 = 0)
                mm_a00=K00; mm_a01=K01; mm_a10=K10; mm_a11=K11;
                mm_b00=innov0; mm_b01='0; mm_b10=innov1; mm_b11='0;
            end
            S_UPD_X: begin
                // X_new (col 0) = X_kp + Knew
                ms_sub = 1'b0;
                ms_a00=Xkp0; ms_a01='0; ms_a10=Xkp1; ms_a11='0;
                ms_b00=Knew0; ms_b01='0; ms_b10=Knew1; ms_b11='0;
            end
            S_IK: begin
                // IK = I - K
                ms_sub = 1'b1;
                ms_a00=ONE_Q725; ms_a01='0;       ms_a10='0;       ms_a11=ONE_Q725;
                ms_b00=K00;      ms_b01=K01;      ms_b10=K10;      ms_b11=K11;
            end
            S_UPD_P: begin
                // P_new = IK * Pkp
                mm_a00=IK00; mm_a01=IK01; mm_a10=IK10; mm_a11=IK11;
                mm_b00=Pkp00; mm_b01=Pkp01; mm_b10=Pkp10; mm_b11=Pkp11;
            end
            default: ; // unused
        endcase
    end

    // ---- Datapath sequential block ----------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all internal state
            X0 <= '0;  X1 <= '0;
            P00 <= P0_00_Q725; P01 <= '0;
            P10 <= '0;          P11 <= P0_11_Q725;

            Xkp0 <= '0; Xkp1 <= '0;
            Pkp00 <= '0; Pkp01 <= '0; Pkp10 <= '0; Pkp11 <= '0;
            S00 <= '0; S01 <= '0; S10 <= '0; S11 <= '0;
            Sinv00 <= '0; Sinv01 <= '0; Sinv10 <= '0; Sinv11 <= '0;
            K00 <= '0; K01 <= '0; K10 <= '0; K11 <= '0;
            Y0 <= '0; Y1 <= '0;
            innov0 <= '0; innov1 <= '0;
            Knew0 <= '0; Knew1 <= '0;
            IK00 <= '0; IK01 <= '0; IK10 <= '0; IK11 <= '0;
            Pnew00 <= '0; Pnew01 <= '0; Pnew10 <= '0; Pnew11 <= '0;

            est_angle_q9 <= '0;
            dout_valid   <= 1'b0;
            inv_start    <= 1'b0;
            inv_in00 <= '0; inv_in01 <= '0; inv_in10 <= '0; inv_in11 <= '0;
        end
        else begin
            dout_valid <= 1'b0;
            inv_start  <= 1'b0;

            case (state)
                S_LATCH: begin
                    Y0 <= q9_to_q25(ang_meas_q9);
                    Y1 <= q9_to_q25(gyro_meas_q9);
                end

                S_PRED_X: begin
                    // X_kp = A * X_prev :
                    //   Xkp0 = X0 + DT * X1
                    //   Xkp1 = X1
                    // Use the shared multiplier?  We can't easily multiplex it for a
                    // scalar product without contention.  Instead, instantiate a tiny
                    // dedicated q_mult here -- it's just one extra multiplier and
                    // keeps the FSM clean.
                    Xkp0 <= pred_x0;
                    Xkp1 <= X1;
                end

                S_PRED_P1: begin
                    // capture A*P_prev into Pkp (to be re-used in next stage)
                    Pkp00 <= mm_p00; Pkp01 <= mm_p01;
                    Pkp10 <= mm_p10; Pkp11 <= mm_p11;
                end

                S_PRED_P2: begin
                    Pkp00 <= mm_p00; Pkp01 <= mm_p01;
                    Pkp10 <= mm_p10; Pkp11 <= mm_p11;
                end

                S_PRED_P3: begin
                    Pkp00 <= ms_p00; Pkp01 <= ms_p01;
                    Pkp10 <= ms_p10; Pkp11 <= ms_p11;
                end

                S_INNOV_S: begin
                    S00 <= ms_p00; S01 <= ms_p01;
                    S10 <= ms_p10; S11 <= ms_p11;
                end

                S_INV_START: begin
                    inv_in00 <= S00; inv_in01 <= S01;
                    inv_in10 <= S10; inv_in11 <= S11;
                    inv_start <= 1'b1;
                end

                S_INV_WAIT: begin
                    if (inv_done) begin
                        Sinv00 <= inv_out00; Sinv01 <= inv_out01;
                        Sinv10 <= inv_out10; Sinv11 <= inv_out11;
                    end
                end

                S_GAIN: begin
                    K00 <= mm_p00; K01 <= mm_p01;
                    K10 <= mm_p10; K11 <= mm_p11;
                end

                S_RESID: begin
                    innov0 <= ms_p00;
                    innov1 <= ms_p10;
                end

                S_KINNOV: begin
                    Knew0 <= mm_p00;
                    Knew1 <= mm_p10;
                end

                S_UPD_X: begin
                    // Capture into temporary; commit happens in S_COMMIT
                    Xkp0 <= ms_p00;   // reuse Xkp0 register to hold X_new col0 temporarily
                    Xkp1 <= ms_p10;
                end

                S_IK: begin
                    IK00 <= ms_p00; IK01 <= ms_p01;
                    IK10 <= ms_p10; IK11 <= ms_p11;
                end

                S_UPD_P: begin
                    Pnew00 <= mm_p00; Pnew01 <= mm_p01;
                    Pnew10 <= mm_p10; Pnew11 <= mm_p11;
                end

                S_COMMIT: begin
                    X0  <= Xkp0;
                    X1  <= Xkp1;
                    P00 <= Pnew00; P01 <= Pnew01;
                    P10 <= Pnew10; P11 <= Pnew11;
                    est_angle_q9 <= q25_to_q9(Xkp0);
                    dout_valid   <= 1'b1;
                end

                default: ;
            endcase
        end
    end

    // ---- Tiny helper: predicted angle = X0 + DT * X1 ---------------------
    // (Single scalar mult + add.  Combinational off X0, X1, DT.)
    wire signed [W-1:0] dt_x1;
    wire signed [W-1:0] pred_x0;
    q_mult   #(.W(W), .FRAC(FRAC)) u_predmul (.a(DT_Q725), .b(X1),    .p(dt_x1));
    q_addsub #(.W(W))              u_predadd (.sub(1'b0),  .a(X0),    .b(dt_x1), .y(pred_x0));

endmodule

`default_nettype wire
