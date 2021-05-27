source ../init_aocl_a10gx_18_0
rm -rf conv_result.log
make clean
ls bin
rm -rf bin
make host
aoc -march=emulator -report -I device/RTL -L device/RTL -l rtl_lib.aoclib  -g  ./device/conv_pipe.cl

export CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 
#gdb ./run.exe 
#./run.exe conv_pipe.aocx 
./run.exe ../data/picture/picture.jpg 

