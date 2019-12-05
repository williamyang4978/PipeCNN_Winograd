source /home/share/init_aocl_a10gx_19_1
make clean
aoc -report -I device/RTL -L device/RTL -l rtl_lib.aoclib  -g  -rtl ./device/conv_pipe.cl
