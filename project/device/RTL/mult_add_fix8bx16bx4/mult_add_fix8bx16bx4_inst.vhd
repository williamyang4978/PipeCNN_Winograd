	component mult_add_fix8bx16bx4 is
		port (
			clock0  : in  std_logic                     := 'X';             -- clock0
			dataa_0 : in  std_logic_vector(15 downto 0) := (others => 'X'); -- dataa_0
			dataa_1 : in  std_logic_vector(15 downto 0) := (others => 'X'); -- dataa_1
			dataa_2 : in  std_logic_vector(15 downto 0) := (others => 'X'); -- dataa_2
			dataa_3 : in  std_logic_vector(15 downto 0) := (others => 'X'); -- dataa_3
			datab_0 : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- datab_0
			datab_1 : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- datab_1
			datab_2 : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- datab_2
			datab_3 : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- datab_3
			result  : out std_logic_vector(25 downto 0)                     -- result
		);
	end component mult_add_fix8bx16bx4;

	u0 : component mult_add_fix8bx16bx4
		port map (
			clock0  => CONNECTED_TO_clock0,  --  clock0.clock0
			dataa_0 => CONNECTED_TO_dataa_0, -- dataa_0.dataa_0
			dataa_1 => CONNECTED_TO_dataa_1, -- dataa_1.dataa_1
			dataa_2 => CONNECTED_TO_dataa_2, -- dataa_2.dataa_2
			dataa_3 => CONNECTED_TO_dataa_3, -- dataa_3.dataa_3
			datab_0 => CONNECTED_TO_datab_0, -- datab_0.datab_0
			datab_1 => CONNECTED_TO_datab_1, -- datab_1.datab_1
			datab_2 => CONNECTED_TO_datab_2, -- datab_2.datab_2
			datab_3 => CONNECTED_TO_datab_3, -- datab_3.datab_3
			result  => CONNECTED_TO_result   --  result.result
		);

