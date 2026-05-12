/*
 * AHRS VGA display peripheral
 *
 * Storage-block / M10K version with default draft display.
 *
 * Fixed for proper M10K inference:
 *   - RAM read/write is in a pure synchronous always_ff (NO reset).
 *   - No conditional clearing of RAM output registers.
 *   - Control registers (display_buffer etc.) are in a separate always_ff
 *     that DOES have async reset.
 *
 * Address map and protocol unchanged.
 */
module ahrs_vga (
    input  logic        clk,
    input  logic        reset,
 
    input  logic [31:0] writedata,
    output logic [31:0] readdata,
    input  logic [15:0] address,
    input  logic        write,
    input  logic        chipselect,
 
    output logic [7:0]  VGA_R,
    output logic [7:0]  VGA_G,
    output logic [7:0]  VGA_B,
    output logic        VGA_CLK,
    output logic        VGA_HS,
    output logic        VGA_VS,
    output logic        VGA_BLANK_n,
    output logic        VGA_SYNC_n
);
    // ------------------------------------------------------------
    localparam int VGA_WIDTH      = 640;
    localparam int VGA_HEIGHT     = 480;
    localparam int MAX_SEGMENTS   = 4;
    localparam int ROWS_STORED    = 512;
 
    localparam logic [15:0] POS_BUF0_BASE = 16'h0000;
    localparam logic [15:0] POS_BUF1_BASE = 16'h0800;
    localparam logic [15:0] CTRL_ADDR     = 16'h1000;
 
    // ------------------------------------------------------------
    // VGA timing
    // ------------------------------------------------------------
    logic [10:0] hcount;
    logic [9:0]  vcount;
    logic [9:0] x, y;
    assign x = hcount[10:1];
    assign y = vcount;
 
    vga_counters counters (
        .clk50       (clk),
        .reset       (reset),
        .hcount      (hcount),
        .vcount      (vcount),
        .VGA_CLK     (VGA_CLK),
        .VGA_HS      (VGA_HS),
        .VGA_VS      (VGA_VS),
        .VGA_BLANK_n (VGA_BLANK_n),
        .VGA_SYNC_n  (VGA_SYNC_n)
    );
 
    // Storage blocks
    (* ramstyle = "M10K" *) logic [31:0] posbuf0_seg0 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf0_seg1 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf0_seg2 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf0_seg3 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf1_seg0 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf1_seg1 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf1_seg2 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf1_seg3 [0:ROWS_STORED-1];
 
    // RAM synchronous-read outputs (one per RAM)
    logic [31:0] q_pos0_b0, q_pos1_b0, q_pos2_b0, q_pos3_b0;
    logic [31:0] q_pos0_b1, q_pos1_b1, q_pos2_b1, q_pos3_b1;
 
    // Double-buffer control (in normal logic, async reset is fine)
    logic display_buffer;
    logic requested_buffer;
    logic swap_pending;
    logic display_buffer_q;   // 1 cycle delayed to align with RAM read
 
    logic start_of_frame;
    assign start_of_frame = (hcount == 11'd0) && (vcount == 10'd0);
 
    // Avalon address decode
    logic pos_access_buf0, pos_access_buf1, control_access;
    logic [10:0] table_index;
    logic [8:0]  write_row;
    logic [1:0]  write_seg;
 
    assign pos_access_buf0 =
        (address >= POS_BUF0_BASE) && (address < POS_BUF0_BASE + 16'd2048);
    assign pos_access_buf1 =
        (address >= POS_BUF1_BASE) && (address < POS_BUF1_BASE + 16'd2048);
    assign control_access  = (address == CTRL_ADDR);
 
    always_comb begin
        table_index = 11'd0;
        if (pos_access_buf0)      table_index = address - POS_BUF0_BASE;
        else if (pos_access_buf1) table_index = address - POS_BUF1_BASE;
 
        write_row = table_index[10:2];
        write_seg = table_index[1:0];
    end
 
    // Per-RAM write enables
    logic w_b0_s0, w_b0_s1, w_b0_s2, w_b0_s3;
    logic w_b1_s0, w_b1_s1, w_b1_s2, w_b1_s3;
 
    logic wr_valid;
    assign wr_valid = chipselect && write && (write_row < VGA_HEIGHT);
 
    assign w_b0_s0 = wr_valid && pos_access_buf0 && (write_seg == 2'd0);
    assign w_b0_s1 = wr_valid && pos_access_buf0 && (write_seg == 2'd1);
    assign w_b0_s2 = wr_valid && pos_access_buf0 && (write_seg == 2'd2);
    assign w_b0_s3 = wr_valid && pos_access_buf0 && (write_seg == 2'd3);
 
    assign w_b1_s0 = wr_valid && pos_access_buf1 && (write_seg == 2'd0);
    assign w_b1_s1 = wr_valid && pos_access_buf1 && (write_seg == 2'd1);
    assign w_b1_s2 = wr_valid && pos_access_buf1 && (write_seg == 2'd2);
    assign w_b1_s3 = wr_valid && pos_access_buf1 && (write_seg == 2'd3);
 
    // Read address
    logic [8:0] read_addr;
    assign read_addr = y[8:0];
 
    // RAM block: pure synchronous, NO reset.

    always_ff @(posedge clk) begin
        // ---- buffer 0 ----
        if (w_b0_s0) posbuf0_seg0[write_row] <= writedata;
        if (w_b0_s1) posbuf0_seg1[write_row] <= writedata;
        if (w_b0_s2) posbuf0_seg2[write_row] <= writedata;
        if (w_b0_s3) posbuf0_seg3[write_row] <= writedata;
 
        q_pos0_b0 <= posbuf0_seg0[read_addr];
        q_pos1_b0 <= posbuf0_seg1[read_addr];
        q_pos2_b0 <= posbuf0_seg2[read_addr];
        q_pos3_b0 <= posbuf0_seg3[read_addr];
 
        // ---- buffer 1 ----
        if (w_b1_s0) posbuf1_seg0[write_row] <= writedata;
        if (w_b1_s1) posbuf1_seg1[write_row] <= writedata;
        if (w_b1_s2) posbuf1_seg2[write_row] <= writedata;
        if (w_b1_s3) posbuf1_seg3[write_row] <= writedata;
 
        q_pos0_b1 <= posbuf1_seg0[read_addr];
        q_pos1_b1 <= posbuf1_seg1[read_addr];
        q_pos2_b1 <= posbuf1_seg2[read_addr];
        q_pos3_b1 <= posbuf1_seg3[read_addr];
    end
 
    // Pipeline registers for x, y, blank to align with RAM output
    // (RAM output is 1 cycle delayed)
    logic [9:0] x_d, y_d;
    logic       blank_d;
 
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            x_d     <= 10'd0;
            y_d     <= 10'd0;
            blank_d <= 1'b0;
        end else begin
            x_d     <= x;
            y_d     <= y;
            blank_d <= VGA_BLANK_n;
        end
    end
 

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            display_buffer   <= 1'b0;
            requested_buffer <= 1'b0;
            swap_pending     <= 1'b0;
            display_buffer_q <= 1'b0;
        end else begin
            display_buffer_q <= display_buffer;
 
            if (start_of_frame && swap_pending) begin
                display_buffer <= requested_buffer;
                swap_pending   <= 1'b0;
            end
 
            if (chipselect && write && control_access) begin
                requested_buffer <= writedata[0];
                swap_pending     <= 1'b1;
            end
        end
    end
 
    // Buffer mux: select displayed buffer AFTER synchronous read
    logic [31:0] cur_pos0, cur_pos1, cur_pos2, cur_pos3;
    assign cur_pos0 = display_buffer_q ? q_pos0_b1 : q_pos0_b0;
    assign cur_pos1 = display_buffer_q ? q_pos1_b1 : q_pos1_b0;
    assign cur_pos2 = display_buffer_q ? q_pos2_b1 : q_pos2_b0;
    assign cur_pos3 = display_buffer_q ? q_pos3_b1 : q_pos3_b0;
 
    assign readdata = 32'd0;
 
    // Segment hit + pixel rendering (unchanged)
    function automatic logic seg_hit(
        input logic [31:0] pos_word,
        input logic [9:0]  px
    );
        logic [15:0] xs, xe;
        begin
            xs = pos_word[15:0];
            xe = pos_word[31:16];
            seg_hit = (xe > xs) &&
                      ({6'd0, px} >= xs) && ({6'd0, px} <= xe);
        end
    endfunction
 
    logic [23:0] selected_rgb;
 
    always_comb begin
        selected_rgb = 24'h000000;
 
        if (blank_d && y_d < VGA_HEIGHT) begin
            // Default draft display
            if (y_d < 10'd238)       selected_rgb = 24'h0040FF;
            else if (y_d > 10'd242)  selected_rgb = 24'h008000;
            else                     selected_rgb = 24'hFFFFFF;
 
            // Overlay from memory
            if (seg_hit(cur_pos0, x_d)) selected_rgb = 24'h0040FF;
            if (seg_hit(cur_pos1, x_d)) selected_rgb = 24'h008000;
            if (seg_hit(cur_pos2, x_d)) selected_rgb = 24'hFFFFFF;
            if (seg_hit(cur_pos3, x_d)) selected_rgb = 24'hFFFF00;
        end
 
        VGA_R = selected_rgb[23:16];
        VGA_G = selected_rgb[15:8];
        VGA_B = selected_rgb[7:0];
    end
 
endmodule
 
 
module vga_counters (
    input  logic        clk50,
    input  logic        reset,
    output logic [10:0] hcount,
    output logic [9:0]  vcount,
    output logic        VGA_CLK,
    output logic        VGA_HS,
    output logic        VGA_VS,
    output logic        VGA_BLANK_n,
    output logic        VGA_SYNC_n
);
    localparam logic [10:0] H_ACTIVE      = 11'd1280;
    localparam logic [10:0] H_FRONT_PORCH = 11'd32;
    localparam logic [10:0] H_SYNC        = 11'd192;
    localparam logic [10:0] H_BACK_PORCH  = 11'd96;
    localparam logic [10:0] H_TOTAL       =
        H_ACTIVE + H_FRONT_PORCH + H_SYNC + H_BACK_PORCH;
 
    localparam logic [9:0] V_ACTIVE      = 10'd480;
    localparam logic [9:0] V_FRONT_PORCH = 10'd10;
    localparam logic [9:0] V_SYNC        = 10'd2;
    localparam logic [9:0] V_BACK_PORCH  = 10'd33;
    localparam logic [9:0] V_TOTAL       =
        V_ACTIVE + V_FRONT_PORCH + V_SYNC + V_BACK_PORCH;
 
    logic end_of_line, end_of_frame;
    assign end_of_line  = (hcount == H_TOTAL - 11'd1);
    assign end_of_frame = (vcount == V_TOTAL - 10'd1);
 
    always_ff @(posedge clk50 or posedge reset) begin
        if (reset)            hcount <= 11'd0;
        else if (end_of_line) hcount <= 11'd0;
        else                  hcount <= hcount + 11'd1;
    end
 
    always_ff @(posedge clk50 or posedge reset) begin
        if (reset) vcount <= 10'd0;
        else if (end_of_line) begin
            if (end_of_frame) vcount <= 10'd0;
            else              vcount <= vcount + 10'd1;
        end
    end
 
    assign VGA_HS = ~((hcount >= H_ACTIVE + H_FRONT_PORCH) &&
                      (hcount <  H_ACTIVE + H_FRONT_PORCH + H_SYNC));
    assign VGA_VS = ~((vcount >= V_ACTIVE + V_FRONT_PORCH) &&
                      (vcount <  V_ACTIVE + V_FRONT_PORCH + V_SYNC));
    assign VGA_BLANK_n = (hcount < H_ACTIVE) && (vcount < V_ACTIVE);
    assign VGA_CLK    = hcount[0];
    assign VGA_SYNC_n = 1'b0;
endmodule
 

/*
 * AHRS VGA display peripheral
 *
 * Storage-block / M10K version with default draft display.
 *
 * Fixed for proper M10K inference:
 *   - RAM read/write is in a pure synchronous always_ff (NO reset).
 *   - No conditional clearing of RAM output registers.
 *   - Control registers (display_buffer etc.) are in a separate always_ff
 *     that DOES have async reset.
 *
 * Address map and protocol unchanged.
 */
module ahrs_vga (
    input  logic        clk,
    input  logic        reset,
 
    input  logic [31:0] writedata,
    output logic [31:0] readdata,
    input  logic [15:0] address,
    input  logic        write,
    input  logic        chipselect,
 
    output logic [7:0]  VGA_R,
    output logic [7:0]  VGA_G,
    output logic [7:0]  VGA_B,
    output logic        VGA_CLK,
    output logic        VGA_HS,
    output logic        VGA_VS,
    output logic        VGA_BLANK_n,
    output logic        VGA_SYNC_n
);
    // ------------------------------------------------------------
    localparam int VGA_WIDTH      = 640;
    localparam int VGA_HEIGHT     = 480;
    localparam int MAX_SEGMENTS   = 4;
    localparam int ROWS_STORED    = 512;
 
    localparam logic [15:0] POS_BUF0_BASE = 16'h0000;
    localparam logic [15:0] POS_BUF1_BASE = 16'h0800;
    localparam logic [15:0] CTRL_ADDR     = 16'h1000;
 
    // ------------------------------------------------------------
    // VGA timing
    // ------------------------------------------------------------
    logic [10:0] hcount;
    logic [9:0]  vcount;
    logic [9:0] x, y;
    assign x = hcount[10:1];
    assign y = vcount;
 
    vga_counters counters (
        .clk50       (clk),
        .reset       (reset),
        .hcount      (hcount),
        .vcount      (vcount),
        .VGA_CLK     (VGA_CLK),
        .VGA_HS      (VGA_HS),
        .VGA_VS      (VGA_VS),
        .VGA_BLANK_n (VGA_BLANK_n),
        .VGA_SYNC_n  (VGA_SYNC_n)
    );
 
    // Storage blocks
    (* ramstyle = "M10K" *) logic [31:0] posbuf0_seg0 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf0_seg1 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf0_seg2 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf0_seg3 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf1_seg0 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf1_seg1 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf1_seg2 [0:ROWS_STORED-1];
    (* ramstyle = "M10K" *) logic [31:0] posbuf1_seg3 [0:ROWS_STORED-1];
 
    // RAM synchronous-read outputs (one per RAM)
    logic [31:0] q_pos0_b0, q_pos1_b0, q_pos2_b0, q_pos3_b0;
    logic [31:0] q_pos0_b1, q_pos1_b1, q_pos2_b1, q_pos3_b1;
 
    // Double-buffer control (in normal logic, async reset is fine)
    logic display_buffer;
    logic requested_buffer;
    logic swap_pending;
    logic display_buffer_q;   // 1 cycle delayed to align with RAM read
 
    logic start_of_frame;
    assign start_of_frame = (hcount == 11'd0) && (vcount == 10'd0);
 
    // Avalon address decode
    logic pos_access_buf0, pos_access_buf1, control_access;
    logic [10:0] table_index;
    logic [8:0]  write_row;
    logic [1:0]  write_seg;
 
    assign pos_access_buf0 =
        (address >= POS_BUF0_BASE) && (address < POS_BUF0_BASE + 16'd2048);
    assign pos_access_buf1 =
        (address >= POS_BUF1_BASE) && (address < POS_BUF1_BASE + 16'd2048);
    assign control_access  = (address == CTRL_ADDR);
 
    always_comb begin
        table_index = 11'd0;
        if (pos_access_buf0)      table_index = address - POS_BUF0_BASE;
        else if (pos_access_buf1) table_index = address - POS_BUF1_BASE;
 
        write_row = table_index[10:2];
        write_seg = table_index[1:0];
    end
 
    // Per-RAM write enables
    logic w_b0_s0, w_b0_s1, w_b0_s2, w_b0_s3;
    logic w_b1_s0, w_b1_s1, w_b1_s2, w_b1_s3;
 
    logic wr_valid;
    assign wr_valid = chipselect && write && (write_row < VGA_HEIGHT);
 
    assign w_b0_s0 = wr_valid && pos_access_buf0 && (write_seg == 2'd0);
    assign w_b0_s1 = wr_valid && pos_access_buf0 && (write_seg == 2'd1);
    assign w_b0_s2 = wr_valid && pos_access_buf0 && (write_seg == 2'd2);
    assign w_b0_s3 = wr_valid && pos_access_buf0 && (write_seg == 2'd3);
 
    assign w_b1_s0 = wr_valid && pos_access_buf1 && (write_seg == 2'd0);
    assign w_b1_s1 = wr_valid && pos_access_buf1 && (write_seg == 2'd1);
    assign w_b1_s2 = wr_valid && pos_access_buf1 && (write_seg == 2'd2);
    assign w_b1_s3 = wr_valid && pos_access_buf1 && (write_seg == 2'd3);
 
    // Read address
    logic [8:0] read_addr;
    assign read_addr = y[8:0];
 
    // RAM block: pure synchronous, NO reset.

    always_ff @(posedge clk) begin
        // ---- buffer 0 ----
        if (w_b0_s0) posbuf0_seg0[write_row] <= writedata;
        if (w_b0_s1) posbuf0_seg1[write_row] <= writedata;
        if (w_b0_s2) posbuf0_seg2[write_row] <= writedata;
        if (w_b0_s3) posbuf0_seg3[write_row] <= writedata;
 
        q_pos0_b0 <= posbuf0_seg0[read_addr];
        q_pos1_b0 <= posbuf0_seg1[read_addr];
        q_pos2_b0 <= posbuf0_seg2[read_addr];
        q_pos3_b0 <= posbuf0_seg3[read_addr];
 
        // ---- buffer 1 ----
        if (w_b1_s0) posbuf1_seg0[write_row] <= writedata;
        if (w_b1_s1) posbuf1_seg1[write_row] <= writedata;
        if (w_b1_s2) posbuf1_seg2[write_row] <= writedata;
        if (w_b1_s3) posbuf1_seg3[write_row] <= writedata;
 
        q_pos0_b1 <= posbuf1_seg0[read_addr];
        q_pos1_b1 <= posbuf1_seg1[read_addr];
        q_pos2_b1 <= posbuf1_seg2[read_addr];
        q_pos3_b1 <= posbuf1_seg3[read_addr];
    end
 
    // Pipeline registers for x, y, blank to align with RAM output
    // (RAM output is 1 cycle delayed)
    logic [9:0] x_d, y_d;
    logic       blank_d;
 
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            x_d     <= 10'd0;
            y_d     <= 10'd0;
            blank_d <= 1'b0;
        end else begin
            x_d     <= x;
            y_d     <= y;
            blank_d <= VGA_BLANK_n;
        end
    end
 

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            display_buffer   <= 1'b0;
            requested_buffer <= 1'b0;
            swap_pending     <= 1'b0;
            display_buffer_q <= 1'b0;
        end else begin
            display_buffer_q <= display_buffer;
 
            if (start_of_frame && swap_pending) begin
                display_buffer <= requested_buffer;
                swap_pending   <= 1'b0;
            end
 
            if (chipselect && write && control_access) begin
                requested_buffer <= writedata[0];
                swap_pending     <= 1'b1;
            end
        end
    end
 
    // Buffer mux: select displayed buffer AFTER synchronous read
    logic [31:0] cur_pos0, cur_pos1, cur_pos2, cur_pos3;
    assign cur_pos0 = display_buffer_q ? q_pos0_b1 : q_pos0_b0;
    assign cur_pos1 = display_buffer_q ? q_pos1_b1 : q_pos1_b0;
    assign cur_pos2 = display_buffer_q ? q_pos2_b1 : q_pos2_b0;
    assign cur_pos3 = display_buffer_q ? q_pos3_b1 : q_pos3_b0;
 
    assign readdata = 32'd0;
 
    // Segment hit + pixel rendering (unchanged)
    function automatic logic seg_hit(
        input logic [31:0] pos_word,
        input logic [9:0]  px
    );
        logic [15:0] xs, xe;
        begin
            xs = pos_word[15:0];
            xe = pos_word[31:16];
            seg_hit = (xe > xs) &&
                      ({6'd0, px} >= xs) && ({6'd0, px} <= xe);
        end
    endfunction
 
    logic [23:0] selected_rgb;
 
    always_comb begin
        selected_rgb = 24'h000000;
 
        if (blank_d && y_d < VGA_HEIGHT) begin
            // Default draft display
            if (y_d < 10'd238)       selected_rgb = 24'h0040FF;
            else if (y_d > 10'd242)  selected_rgb = 24'h008000;
            else                     selected_rgb = 24'hFFFFFF;
 
            // Overlay from memory
            if (seg_hit(cur_pos0, x_d)) selected_rgb = 24'h0040FF;
            if (seg_hit(cur_pos1, x_d)) selected_rgb = 24'h008000;
            if (seg_hit(cur_pos2, x_d)) selected_rgb = 24'hFFFFFF;
            if (seg_hit(cur_pos3, x_d)) selected_rgb = 24'hFFFF00;
        end
 
        VGA_R = selected_rgb[23:16];
        VGA_G = selected_rgb[15:8];
        VGA_B = selected_rgb[7:0];
    end
 
endmodule
 
 
module vga_counters (
    input  logic        clk50,
    input  logic        reset,
    output logic [10:0] hcount,
    output logic [9:0]  vcount,
    output logic        VGA_CLK,
    output logic        VGA_HS,
    output logic        VGA_VS,
    output logic        VGA_BLANK_n,
    output logic        VGA_SYNC_n
);
    localparam logic [10:0] H_ACTIVE      = 11'd1280;
    localparam logic [10:0] H_FRONT_PORCH = 11'd32;
    localparam logic [10:0] H_SYNC        = 11'd192;
    localparam logic [10:0] H_BACK_PORCH  = 11'd96;
    localparam logic [10:0] H_TOTAL       =
        H_ACTIVE + H_FRONT_PORCH + H_SYNC + H_BACK_PORCH;
 
    localparam logic [9:0] V_ACTIVE      = 10'd480;
    localparam logic [9:0] V_FRONT_PORCH = 10'd10;
    localparam logic [9:0] V_SYNC        = 10'd2;
    localparam logic [9:0] V_BACK_PORCH  = 10'd33;
    localparam logic [9:0] V_TOTAL       =
        V_ACTIVE + V_FRONT_PORCH + V_SYNC + V_BACK_PORCH;
 
    logic end_of_line, end_of_frame;
    assign end_of_line  = (hcount == H_TOTAL - 11'd1);
    assign end_of_frame = (vcount == V_TOTAL - 10'd1);
 
    always_ff @(posedge clk50 or posedge reset) begin
        if (reset)            hcount <= 11'd0;
        else if (end_of_line) hcount <= 11'd0;
        else                  hcount <= hcount + 11'd1;
    end
 
    always_ff @(posedge clk50 or posedge reset) begin
        if (reset) vcount <= 10'd0;
        else if (end_of_line) begin
            if (end_of_frame) vcount <= 10'd0;
            else              vcount <= vcount + 10'd1;
        end
    end
 
    assign VGA_HS = ~((hcount >= H_ACTIVE + H_FRONT_PORCH) &&
                      (hcount <  H_ACTIVE + H_FRONT_PORCH + H_SYNC));
    assign VGA_VS = ~((vcount >= V_ACTIVE + V_FRONT_PORCH) &&
                      (vcount <  V_ACTIVE + V_FRONT_PORCH + V_SYNC));
    assign VGA_BLANK_n = (hcount < H_ACTIVE) && (vcount < V_ACTIVE);
    assign VGA_CLK    = hcount[0];
    assign VGA_SYNC_n = 1'b0;
endmodule