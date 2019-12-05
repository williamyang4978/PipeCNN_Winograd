module mult_add_fix8bx16bx4 (
		input  wire        clock0,  //  clock0.clock0
		input  wire [15:0] dataa_0, // dataa_0.dataa_0
		input  wire [15:0] dataa_1, // dataa_1.dataa_1
		input  wire [15:0] dataa_2, // dataa_2.dataa_2
		input  wire [15:0] dataa_3, // dataa_3.dataa_3
		input  wire [7:0]  datab_0, // datab_0.datab_0
		input  wire [7:0]  datab_1, // datab_1.datab_1
		input  wire [7:0]  datab_2, // datab_2.datab_2
		input  wire [7:0]  datab_3, // datab_3.datab_3
		output wire [25:0] result   //  result.result
	);
endmodule

