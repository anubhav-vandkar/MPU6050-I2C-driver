// =============================================================================
// kalman_avalon.sv : Avalon-MM slave wrapper around kalman_top
// -----------------------------------------------------------------------------
// Register map (32-bit word addressing, byteenable ignored for simplicity):
//
//   addr  | name        | access | bits used | description
//   ------+-------------+--------+-----------+---------------------------------
//    0x0  | ROLL_IN     |   W    |  [15:0]   | Q7.9 accel roll
//    0x1  | PITCH_IN    |   W    |  [15:0]   | Q7.9 accel pitch
//    0x2  | GX_IN       |   W    |  [15:0]   | Q7.9 gyro X
//    0x3  | GY_IN       |   W    |  [15:0]   | Q7.9 gyro Y
//    0x4  | START       |   W    |  [0]      | write 1 to launch one Kalman iter
//    0x5  | STATUS      |   R    |  [0]=done | done=1 means new estimates ready
//    0x5  | STATUS_CLR  |   W    |  [0]      | write 1 to clear `done`
//    0x6  | EST_ROLL    |   R    |  [15:0]   | Q7.9 estimated roll
//    0x7  | EST_PITCH   |   R    |  [15:0]   | Q7.9 estimated pitch
//
// Recommended HPS sequence:
//   1. write ROLL_IN, PITCH_IN, GX_IN, GY_IN
//   2. write 1 to START
//   3. poll STATUS until done==1   (or use IRQ -- irq is asserted while done==1)
//   4. read EST_ROLL, EST_PITCH
//   5. write 1 to STATUS_CLR  (clears done and irq)
// =============================================================================
`default_nettype none

module kalman_avalon #(
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
    input  wire        clk,
    input  wire        rst_n,

    // Avalon-MM slave
    input  wire [2:0]  avs_address,
    input  wire        avs_write,
    input  wire [31:0] avs_writedata,
    input  wire        avs_read,
    output reg  [31:0] avs_readdata,
    // (no waitrequest -- single-cycle access)

    output wire        irq                  // optional interrupt
);

    // ---- Input registers ---------------------------------------------------
    reg signed [IO_W-1:0] roll_in_r, pitch_in_r, gx_in_r, gy_in_r;
    reg                   start_pulse;
    reg                   done_flag;

    // ---- Kalman top --------------------------------------------------------
    wire signed [IO_W-1:0] est_roll, est_pitch;
    wire                   dout_valid, busy;

    kalman_top #(
        .W(W), .FRAC(FRAC), .IO_W(IO_W), .IO_FRAC(IO_FRAC),
        .DT_Q725(DT_Q725),
        .Q00_Q725(Q00_Q725), .Q11_Q725(Q11_Q725),
        .R00_Q725(R00_Q725), .R11_Q725(R11_Q725),
        .P0_00_Q725(P0_00_Q725), .P0_11_Q725(P0_11_Q725)
    ) u_kalman (
        .clk(clk), .rst_n(rst_n),
        .din_valid(start_pulse),
        .roll_in (roll_in_r),
        .pitch_in(pitch_in_r),
        .gx_in   (gx_in_r),
        .gy_in   (gy_in_r),
        .est_roll (est_roll),
        .est_pitch(est_pitch),
        .dout_valid(dout_valid),
        .busy(busy)
    );

    // ---- Output registers (latch on dout_valid) ----------------------------
    reg signed [IO_W-1:0] est_roll_r, est_pitch_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            est_roll_r  <= '0;
            est_pitch_r <= '0;
        end
        else if (dout_valid) begin
            est_roll_r  <= est_roll;
            est_pitch_r <= est_pitch;
        end
    end

    // ---- Write decoding ----------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            roll_in_r   <= '0;
            pitch_in_r  <= '0;
            gx_in_r     <= '0;
            gy_in_r     <= '0;
            start_pulse <= 1'b0;
            done_flag   <= 1'b0;
        end
        else begin
            // start_pulse is one-shot
            start_pulse <= 1'b0;

            if (avs_write) begin
                case (avs_address)
                    3'd0: roll_in_r  <= avs_writedata[IO_W-1:0];
                    3'd1: pitch_in_r <= avs_writedata[IO_W-1:0];
                    3'd2: gx_in_r    <= avs_writedata[IO_W-1:0];
                    3'd3: gy_in_r    <= avs_writedata[IO_W-1:0];
                    3'd4: if (avs_writedata[0] && !busy) start_pulse <= 1'b1;
                    3'd5: if (avs_writedata[0]) done_flag <= 1'b0;   // STATUS_CLR
                    default: ;
                endcase
            end

            // Set done_flag when computation completes
            if (dout_valid) done_flag <= 1'b1;
        end
    end

    // ---- Read decoding -----------------------------------------------------
    always @* begin
        avs_readdata = 32'd0;
        case (avs_address)
            3'd5: avs_readdata = {31'd0, done_flag};
            3'd6: avs_readdata = {{(32-IO_W){est_roll_r[IO_W-1]}},  est_roll_r};   // sign-extend
            3'd7: avs_readdata = {{(32-IO_W){est_pitch_r[IO_W-1]}}, est_pitch_r};
            default: avs_readdata = 32'd0;
        endcase
    end

    assign irq = done_flag;

endmodule

`default_nettype wire
