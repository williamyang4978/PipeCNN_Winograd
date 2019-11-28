
# PipeCNN_Winograd

## About 

**PipeCNN_Winograd** is an OpenCL-based FPGA accelerator which can run the YOLOv2 model very efficiently. It was originally forked from **PipeCNN** (https://github.com/doonny/PipeCNN) and has been modified significantly ever since. Winnograd algorithm is employed to accelerate convolution calculation in FPGA. The design solves the unaligned memory access challenge caused by Winnograd algorithm, makes full use of the available memory access bandwidth and utilizes all available DSP resources. Parallelism is exploited in various dimensions for optimal performance. PipeCNN_Winograd only supports the YOLOv2 model at present. To our knowledge, this is the first open source project to implement Winograd algorithm in FPGA with OpenCL language.

## How to Use

Clone PipeCNN_Winograd project from [github](https://github.com/PipeCNN_Winograd). 
1) Put testing pictures into ./data/picture.

2) Pre-process model weights with the weight_gen.sh script in ./data/weight;

3) Compile design with the run_fpga.sh script in ./project, it will take around one hour to finish all the compilations. Finally, there will generate file "conv_pipe.aocx"

4) Test the design with FPGA;
        ./project/fpga_test.sh

This project is tested with Intel Arrial 10 FPGA and Intel OpenCL SDK v18.0/v19.1.

## Citation
Please kindly cite our work of PipeCNN-Winograd if it helps your research:
```
Anrong Yang, Yuanhui Li, Hongqiao Shu, Jianlin Deng, Chuanzhao Ma, Zheng Li and Qigang Wang, "An OpenCL-Based FPGA Accelerator for Compressed YOLOv2", FPT 2019.
```

