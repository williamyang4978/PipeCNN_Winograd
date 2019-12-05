source /home/share/init_aocl_a10gx_18_0
make clean
aoc -seed=6 -report -I device/RTL -L device/RTL -l rtl_lib.aoclib  -g  ./device/conv_pipe.cl
