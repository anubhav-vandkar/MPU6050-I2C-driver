module vga_ball(input  logic        clk,
                input  logic        reset,
                input  logic [31:0] writedata,
                input  logic        write,
                input  logic        chipselect,
                input  logic [2:0]  address,
                output logic [7:0]  VGA_R, VGA_G, VGA_B,
                output logic        VGA_CLK, VGA_HS, VGA_VS,
                                    VGA_BLANK_n,
                output logic        VGA_SYNC_n);

  logic [10:0] hcount;
  logic [9:0] vcount;

  localparam [9:0] CENTRE_X  = 10'd320;
  localparam [20:0] RADIUS_SQ = 21'd5625;

  logic signed [11:0] pitch_q39;
  logic signed [15:0] slope_q39;

  logic signed [31:0] pitch_scaled;
  logic [8:0]  centre_y;

  assign pitch_scaled = ($signed(pitch_q39) * 32'sd120) / 32'sd804;
  assign centre_y     = 9'd240 - pitch_scaled[8:0];

  logic [9:0] dx_u;
  logic [8:0] dy_u;
  logic [19:0] dx_sq;
  logic [17:0] dy_sq;
  logic [20:0] distance_sq;
  logic in_circle;

  assign dx_u = (hcount[10:1] >= CENTRE_X) ? (hcount[10:1]  - CENTRE_X) : (CENTRE_X - hcount[10:1]);
  assign dy_u = (vcount[8:0] >= centre_y) ? (vcount[8:0]  - centre_y) : (centre_y - vcount[8:0]);
  assign dx_sq = dx_u * dx_u;
  assign dy_sq = dy_u * dy_u;
  assign distance_sq = {1'b0, dx_sq} + {2'b0, dy_sq};
  assign in_circle = (distance_sq <= RADIUS_SQ);

  logic signed [9:0] dx_line;
  logic signed [9:0] dy_line;
  logic signed [25:0] lhs;
  logic signed [25:0] rhs;
  logic signed [25:0] line_err;
  logic on_line;

  assign dx_line  = $signed({1'b0, hcount[10:1]}) - $signed({1'b0, CENTRE_X});
  assign dy_line  = $signed({1'b0, vcount[9:0]})  - $signed({1'b0, {1'b0, centre_y}});

  assign lhs      = dy_line * 26'sd512;
  assign rhs      = $signed(slope_q39) * $signed({{16{dx_line[9]}}, dx_line});
  assign line_err = lhs - rhs;

  assign on_line  = (line_err <= 26'sd512) && (line_err >= -26'sd512);

  /* ── Background: sky blue top, ground green bottom ──────────── */
  /*
   * Split is fixed at y=240 regardless of pitch.
   * The circle and line move over this static background.
   */
  logic [7:0] bg_r, bg_g, bg_b;

  always_comb begin
    if (vcount < 10'd240) begin
      /* sky -- blue */
      bg_r = 8'h00;
      bg_g = 8'h3A;
      bg_b = 8'hCC;
    end else begin
      /* ground -- brown-green */
      bg_r = 8'h4B;
      bg_g = 8'h6F;
      bg_b = 8'h00;
    end
  end

  always_ff @(posedge clk) begin
    if (reset) begin
      pitch_q39 <= 12'h0;
      slope_q39 <= 16'h0;
    end else if (chipselect && write)
      case (address)
        3'h3 : begin
          slope_q39 <= writedata[31:16];
          pitch_q39 <= writedata[11:0];
        end
      endcase
  end

  always_comb begin
    {VGA_R, VGA_G, VGA_B} = {8'h0, 8'h0, 8'h0};
    if (VGA_BLANK_n) begin
      if (in_circle && on_line)
        {VGA_R, VGA_G, VGA_B} = {8'h00, 8'hff, 8'hff};
      else if (in_circle)
        /* circle body -- white */
        {VGA_R, VGA_G, VGA_B} = {8'hFF, 8'hFF, 8'hFF};
      else
        /* static sky/ground background */
        {VGA_R, VGA_G, VGA_B} = {bg_r, bg_g, bg_b};
    end
  end

  vga_counters counters(.clk50(clk), .*);

endmodule


module vga_counters(
 input  logic         clk50, reset,
 output logic [10:0]  hcount,
 output logic [9:0]   vcount,
 output logic         VGA_CLK, VGA_HS, VGA_VS, VGA_BLANK_n, VGA_SYNC_n);

   parameter HACTIVE      = 11'd 1280,
             HFRONT_PORCH = 11'd 32,
             HSYNC        = 11'd 192,
             HBACK_PORCH  = 11'd 96,
             HTOTAL       = HACTIVE + HFRONT_PORCH + HSYNC + HBACK_PORCH;

   parameter VACTIVE      = 10'd 480,
             VFRONT_PORCH = 10'd 10,
             VSYNC        = 10'd 2,
             VBACK_PORCH  = 10'd 33,
             VTOTAL       = VACTIVE + VFRONT_PORCH + VSYNC + VBACK_PORCH;

   logic endOfLine;

   always_ff @(posedge clk50 or posedge reset)
     if (reset)          hcount <= 0;
     else if (endOfLine) hcount <= 0;
     else                hcount <= hcount + 11'd1;

   assign endOfLine = hcount == HTOTAL - 1;

   logic endOfField;

   always_ff @(posedge clk50 or posedge reset)
     if (reset)          vcount <= 0;
     else if (endOfLine)
       if (endOfField)   vcount <= 0;
       else              vcount <= vcount + 10'd1;

   assign endOfField = vcount == VTOTAL - 1;

   assign VGA_HS      = !( (hcount[10:8] == 3'b101) &
                           !(hcount[7:5] == 3'b111));
   assign VGA_VS      = !( vcount[9:1] == (VACTIVE + VFRONT_PORCH) / 2);
   assign VGA_SYNC_n  = 1'b0;
   assign VGA_BLANK_n = !( hcount[10] & (hcount[9] | hcount[8]) ) &
                        !( vcount[9] | (vcount[8:5] == 4'b1111) );
   assign VGA_CLK     = hcount[0];

endmodule