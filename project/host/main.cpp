//////////////////////////////////////////
//
// OpenCL host program template for multiple
// FPGA boards.
//
// Created by dongwang@2016.01.10
//
/////////////////////////////////////////
//
/*
 * ----------------------------------------------------------
 *
 * PipeCNN_Winograd: Pipecnn accelerated by Winograd
 *  
 * --------------------------------------------------------------------------
 * 
 * Author(s):
 *   -Anrong Yang, yangar@lenovo.com
 *
 * --------------------------------------------------------------------------
 *
 * Copyright (C) 2019 Lenovo
 * Licensed under BSD-3, see COPYING.BSD file for details
 *
 *--------------------------------------------------------------------------
 *
 * Since the PipeCNN on device is modified and implemented by Winograd 
 * algorithm, on the basis of PipeCNN host program, we modify the port 
 * parameters when the device program is called. In addition, we add image 
 * preprocessing, quantization and bounding box of image detection results.
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <assert.h>
#include <list> 


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include<cmath>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <semaphore.h>

#include <iomanip>

#include <CL/opencl.h>

// user defined library
#include "ocl_util.h"
#include "timer.h"

// CNN network configuration file
#include "../device/hw_param.cl"
#include "layer_config.h"

#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
#endif

using namespace std;
using namespace ocl_util;

typedef  signed char  DTYPE;

#ifdef XILINX
//#define USE_SDX_1DDR  // reserved for v7-690T device, DO NOT USE
//#define USE_SDX_4DDR  // reverved for 4-bank DDR BSP, DO NOT USE
#endif

//----------- Design Parameters --------------//
// select what platform is used
#ifdef XILINX
const char *vendor_name = "Xilinx";
#else
const char *vendor_name = "Intel";
#endif
#define DEVICE_TYPE CL_DEVICE_TYPE_ACCELERATOR

// SW System parameters
#define DMA_ALIGNMENT   64
#define MAX_LAYER_NUM   23
#define MAX_BATCH_SIZE  16

#define IN_BUF_SIZE    544*544*32  // Note: the buffer size should be large enough to hold all temperary results
#define OUT_BUF_SIZE   544*544*32
//#define FC_BUF_SIZE    32768*MAX_BATCH_SIZE
#define CONCAT_BUF_SIZE 544*544*32
//#define CONV_RES_BUF_SIZE 34*34*512
#define CONV_RES_BUF_SIZE 544*544*32

#define MEAN_DATA_WIDTH   256
#define MEAN_DATA_HEIGHT  256
#define MEAN_DATA_CHANNEl 3
#define PICTURE_NUM 8000
#define MAX_PIC_NUM 50000
const char *mean_data_file_path    = "./data/imagenet/mean_data.dat";
//const char *synset_word_file_path  = "./data/imagenet/synset_words.txt";
const char *LabelPath		       = "./data/imagenet/val.txt";
char  picture_file_path_head[100]  = "/home/dwang/Work/imagenet/ilsvrc2012/ILSVRC2012_img_val/ILSVRC2012_val_";
char  picture_file_path[100];
int   label[MAX_PIC_NUM]={0};
char  label_buf[MAX_PIC_NUM][1024]={0};
char  synset_buf[1000][1024]={0};
DTYPE searchTop[1024];
float accuracy1=0;
float accuracy5=0;
#define RESULT_OUT_NUM 30*17*17

// AlexNet
// Original problem size
// File size is in num of DTYPE numbers
//#define IMAGE_FILE_SIZE   544*544*3
#define IMAGE_FILE_SIZE   548*544*3
//#define IMAGE_FILE_SIZE   416*416*3
//#define IMAGE_FILE_SIZE   320*320*3
//#define WEIGHTS_FILE_SIZE 60965224 //fc8-1000
//#define WEIGHTS_FILE_SIZE 50547678  // weights+bias
//#define WEIGHTS_FILE_SIZE 142344320  // weights+bias
//#define WEIGHTS_FILE_SIZE 34034752  // weights+bias lanenum=32 wvec=8
#define WEIGHTS_FILE_SIZE 25521216  // weights+bias lanenum=32 wvec=6
//#define WEIGHTS_FILE_SIZE 35742240  // weights+bias lanenum=48 wvec=8
//#define WEIGHTS_FILE_SIZE 34186432  // weights+bias lanenum=64 wvec=8
//#define WEIGHTS_FILE_SIZE 25641152  // weights+bias lanenum=64 wvec=6
#define LAYER_NUM         22 
#define CONV_NUM          22
const char *weight_file_path = "../data/weight/weight.dat";
const char *input_file_path = "./data_yolo9000/image.dat";
const char *ref_file_path = "./data/data_alex/fc8.dat";
const char *dump_file_path = "./result_dump.txt";


/*
// VGG16
// Original problem size
// File size is in num of DTYPE numbers
#define IMAGE_FILE_SIZE   (224*224*3)
#define WEIGHTS_FILE_SIZE 138455872  //fc8-1024
#define LAYER_NUM         16
#define CONV_NUM          13

const char *weight_file_path = "./data/data_vgg/weights.dat";
const char *input_file_path = "./data/data_vgg/image.dat";
const char *ref_file_path = "./data/data_vgg/fc8.dat";
const char *dump_file_path = "./result_dump.txt";
*/

// Configuration file instructions
enum config_item{
layer_type, // "0" -> conv, "1" -> fc

data_w, data_h, data_n, weight_w, weight_h, weight_n, weight_m, bias_size, //memRd Parameters

memrd_src, //"0"-> data_buf  "1"-> output_buf  "2"->"concat_buffer"  "3"->"conv_res_buffer"

conv_x, conv_y, conv_z, conv_stride, conv_padding, conv_split, conv_relu, conv_res_save,  //Conv Parameters

pool_on, pool_x, pool_y, pool_z, pool_size, pool_stride, // Pooling Parameters


memwr_dst, reorg_en, addr_offset//"0"-> data_buf  "1"-> output_buf  "2"->"concat_buf"  "3"->"conv_res_buf"

};

enum input_item{

image_w, image_h, image_n, // original image size

batch_size

};

enum output_item{

output_w, output_h, output_n

};

enum precision_item{

frac_w, frac_b, frac_din, frac_dout

};


// Define the kernel names used
const char *knl_name_memRd = "memRead";
const char *knl_name_conv  = "coreConv";
const char *knl_name_Pool  = "maxPool";
const char *knl_name_memWr = "memWrite";
const char *knl_name_memWriteReorg = "memWriteReorg";
//const char *knl_name_lrn   = "lrn";


//------------ Global Functions & Variables ------------//
cl_uint num_devices = 0;
cl_platform_id platform_id = NULL;
cl_context context = NULL;
cl_program program = NULL;
scoped_array<cl_device_id> device;
scoped_array<cl_kernel> knl_memRd;
//scoped_array<cl_kernel> knl_conv;
scoped_array<cl_kernel> knl_memWr;
//scoped_array<cl_kernel> knl_memWrReorg;
//scoped_array<cl_kernel> knl_pool;
//scoped_array<cl_kernel> knl_lrn;
scoped_array<cl_command_queue> que_memRd;
//scoped_array<cl_command_queue> que_conv;
scoped_array<cl_command_queue> que_memWr;
//scoped_array<cl_command_queue> que_memWrReorg;
//scoped_array<cl_command_queue> que_pool;
scoped_array<cl_mem> data_buf;
scoped_array<cl_mem> output_buf;
scoped_array<cl_mem> weights_buf;
scoped_array<cl_mem> bias_buf;
scoped_array<cl_mem> concat_buf;
scoped_array<cl_mem> conv_res_buf;

#ifdef XILINX
	scoped_array<cl_event> write_event(num_devices);
#else
//DO NOTHING
#endif

DTYPE *weights;
DTYPE *image;
DTYPE *data_init;
DTYPE *weight_conv[MAX_LAYER_NUM];
DTYPE *bias_conv[MAX_LAYER_NUM];
DTYPE *output;
DTYPE *output_one_item;
DTYPE *output_reorder;
DTYPE *golden_ref;

unsigned layer_config_original[LAYER_NUM][NUM_CONFIG_ITEM];

#ifdef USE_OPENCV
int  load_picture(DTYPE *image);
void getAccuracy(DTYPE *output_reorder,int num);
void labelNum();
void numtochar(int num,char *end);
#endif
void loadImageToBuffer(DTYPE *imageDeal);
int  prepare();
void readDataBack();
void verifyResult(int num);
void printResult(int offset, int num);
void filePrintResult();
void dumpResult();
void reorderWeights(DTYPE *weights, DTYPE *weight_buf, unsigned dim1, unsigned dim2, unsigned dim3, unsigned dim4, unsigned dim3_original, unsigned dim4_original, unsigned offset, unsigned padding_offset, unsigned vecSize, unsigned w_vecSize,unsigned laneNum);
void reorderBias(DTYPE *dataIn, DTYPE *bias, unsigned offset, unsigned padding_offset, unsigned dim4, unsigned dim4_original, unsigned laneNum);
void reorderOutput(DTYPE *output, DTYPE *output_reorder, unsigned dim1, unsigned dim2, unsigned dim3);
void extractOutput(DTYPE *output, DTYPE *output_one_item, unsigned item_num, unsigned batch_size, unsigned dim1, unsigned dim2, unsigned dim3);
void softmax(DTYPE *output_reorder , DTYPE *output);
int getProb(DTYPE *output);
void cleanup();





int fpga_object_init();
int fpga_object_main();

using namespace cv;


const float mean_val[]={104, 117, 123};

#define DEL_IMAGE_WEIGHT 544 
#define DEL_IMAGE_HEIGHT 544
#define DEL_IMAGE_CHANNEL 3
#define DEL_IMAGE_SIZE DEL_IMAGE_WEIGHT*DEL_IMAGE_HEIGHT*DEL_IMAGE_CHANNEL



typedef struct{
	float x;
	float y;
	float w;
	float h;
	float score;
	int class_index;
} yolobox;

typedef struct
{
	int index;
	float probvalue;
}sortable_element;

typedef struct
{
	float w;
	float h;
}anchor_unit;


std::vector<yolobox> results;



bool frameReadStatus;
bool isLastFrame;


static pthread_mutex_t mutex;


struct layer_config_Arg
{
	unsigned short memWr_dim1;
	unsigned short memWr_dim2;
	unsigned short memWr_dim3;
	unsigned char  memwr_scal;
	unsigned char  memwr_scal_rem_z;
	unsigned int   memwr_dim1xdim2;
	unsigned char  q_vec;
	unsigned char  rem_size_x;
	unsigned char  memWr_dim1_div_q_vec;
	unsigned short memWr_dim3_div_lane_numxscal;
	unsigned char  scalxq_vec;
	unsigned char  scalxrem_size_x;
	unsigned char  scal_rem_zxq_vec;
	unsigned char  scal_rem_zxrem_size_x;
	unsigned int   dim_z_edge_num;
	

};

float data_str[30*17*17]={0};


void dealImagetoRGBandInt8(const Mat& orig_image, DTYPE *image) 
{
	//int max=0;
	int value=0;
	//int min = 0;
	Timer t;  // Timer used for performance measurement
	float time;
	t.start();
	register int bitoffset =0;
	//double mulParamValue=0 ;

    Mat resized_image(orig_image);
    if (DEL_IMAGE_WEIGHT != orig_image.size().width || DEL_IMAGE_HEIGHT!= orig_image.size().height) {
        resize(orig_image, resized_image, cv::Size(DEL_IMAGE_WEIGHT, DEL_IMAGE_HEIGHT));
    }


    for (register int c = 0; c < DEL_IMAGE_CHANNEL; c++) {
        for (register int  h = 0; h < DEL_IMAGE_HEIGHT; h++) {
			bitoffset = bitoffset+1;
            for (register int w = 0; w < DEL_IMAGE_WEIGHT; w++) {
				
				value = resized_image.at<cv::Vec3b>(h, w)[c]-mean_val[c];
				
                image[bitoffset] = value>>1;
                
				bitoffset++;
			
            }
			
			bitoffset = bitoffset+3;
			
        }
    }


	t.stop();
	time = t.get_time_s();
	
}


template <typename Dtype>
inline Dtype sigmoidfunc(Dtype x)
{
	return 1. / (1. + exp(-x));
}

//static float anchors_array[10] = {1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52};
	
static float anchors_array[10] = {1.32, 1.73, 3.19, 4.01, 5.06, 8.10, 9.47, 4.84, 11.24, 10.01};

template <typename Dtype>
Dtype softmax_region(Dtype* input, int classes)
{
	Dtype sum = 0;
	Dtype large = input[0];
	for (int i = 0; i < classes; ++i)
	{
		//printf("class prob is %f\n", input[i]);
		if (input[i] > large)
			large = input[i];
	}
	for (int i = 0; i < classes; ++i){
		Dtype e = exp(input[i] - large);
		sum += e;
		input[i] = e;
	}
	for (int i = 0; i < classes; ++i){
		input[i] = input[i] / sum;
	}
	return 0;
}

float Boxoverlap(float x1, float w1, float x2, float w2)
{
	float left = std::max(x1 - w1 / 2, x2 - w2 / 2);
	float right = std::min(x1 + w1 / 2, x2 + w2 / 2);
	return right - left;
}

float Calc_iou(const yolobox& box1, const yolobox& box2)
{
	float w = Boxoverlap(box1.x, box1.w, box2.x, box2.w);
	float h = Boxoverlap(box1.y, box1.h, box2.y, box2.h);
	if (w < 0 || h < 0) return 0;
	float inter_area = w * h;
	float union_area = box1.w * box1.h + box2.w * box2.h - inter_area;
	return inter_area / union_area;
}

bool CompareYoloBBox(const sortable_element & a, const sortable_element & b) {
	if (a.probvalue == b.probvalue)
		return a.index > b.index;
	else
		return a.probvalue > b.probvalue;
}

yolobox get_region_box(float *x, vector<anchor_unit> biases, int n, int index, int i, int j, int w, int h)
{

	yolobox result_vec;
	result_vec.x = ((float)j + sigmoidfunc<float>(x[index + 0])) / w;
	result_vec.y = ((float)i + sigmoidfunc<float>(x[index + 1])) / h;
	result_vec.w = (exp(x[index + 2])*biases[n].w) / w;
	result_vec.h = (exp(x[index + 3])*biases[n].h) / h;
	return result_vec;
}

std::vector<yolobox> Do_NMSMultiLabel(std::vector<yolobox>& regboxes, std::vector<std::vector<float> >& probs, int num_class_,float thresh, float threshiou)
{
	std::vector<yolobox> bboxes_nms;
	std::vector<sortable_element> sortelements;
	int totalnum = regboxes.size();
	assert(totalnum == probs.size());
	sortelements.resize(totalnum);
	//init
	for (int c = 0; c < num_class_; c++)
	{
		for (int k = 0; k < totalnum; k++)
		{
			sortelements[k].index = k;
			sortelements[k].probvalue = probs[k][c];
			if (probs[k][c] > 0.0f)
			{
				//printf("probs is %f\n", probs[k][c]);
			}
		}
		std::sort(sortelements.begin(), sortelements.end(), CompareYoloBBox);
		for (int i = 0; i < sortelements.size(); ++i)
		{
			if (probs[sortelements[i].index][c] < thresh)
			{
				probs[sortelements[i].index][c] = 0.0f;
				continue;
			}
			yolobox a = regboxes[sortelements[i].index];
			for (int j = i + 1; j < totalnum; ++j)
			{
				yolobox b = regboxes[sortelements[j].index];
				float calced_iou = Calc_iou(a, b);
				if (calced_iou > threshiou)
				{
					probs[sortelements[j].index][c] = 0.0f;
				}
			}
		}
	}
	for (int c = 0; c < num_class_; c++)
	{
		for (int k = 0; k < totalnum; k++)
		{
			if (probs[k][c] == 0.0f)
				continue;
			regboxes[k].class_index = c;
			bboxes_nms.push_back(regboxes[k]);
		}
	}
	//float calced_iou111 = Calc_iou(bboxes_nms[0], bboxes_nms[1]);
	return bboxes_nms;
}

std::vector<yolobox> YoloV2BoxDetect(float*output_data,int batchnum,int num_class_,int sidenum_,int num_objects_,float thresh_score_,float thresh_iou_)
{
        
    std::vector<yolobox> total_boxes_;
	int owidth = sidenum_;
	int oheight = sidenum_;
    int feature_step = num_class_ + 1 + 4;
	int ochannels = num_objects_*feature_step;
    vector<anchor_unit> anchor_vec;
    static float *tempclassarray = (float *)malloc(num_class_*sizeof(float));

	std::vector<std::vector<float> > fprobs_;

    static float * class_prob_ = (float *)malloc(num_class_*sidenum_*sidenum_*num_objects_*sizeof(float));
	
	fprobs_.resize(num_objects_*sidenum_*sidenum_);

    int anchor_size = sizeof(anchors_array)/sizeof(float);
	anchor_vec.resize(anchor_size/2);
	for (int i = 0; i < fprobs_.size(); i++)
	{
		fprobs_[i].resize(num_class_);
	}
	for (int i = 0; i < anchor_size/2; i++)
	{
		anchor_vec[i].w = anchors_array[2*i];
		anchor_vec[i].h = anchors_array[2*i+1];
	}


        static float *out_rerange = (float *)malloc(batchnum*owidth*oheight*ochannels*sizeof(float));
	int simagestep = owidth*oheight;
        int reimagestep = oheight*owidth*ochannels;
	int colstep = owidth*ochannels;
	for (int nc = 0; nc < batchnum; nc++)
	{
		for (int h = 0; h < oheight; h++)
		{
			for (int w = 0; w < owidth; w++)
			{
				for (int c = 0; c < ochannels; c++)
				{
					out_rerange[nc*reimagestep + h*colstep + w*ochannels + c] = output_data[nc*reimagestep + c*simagestep + h*owidth + w];
				}
			}
		}
	}
	//rerange the output data from (n,c,h,w) -> (n, h, w, c)
	int blob_num = batchnum;
	int blob_insidecount = batchnum*sidenum_*sidenum_*ochannels;
	int class_step = num_class_;
	for (int k = 0; k < blob_num; ++k)
	{
		std::vector<yolobox> result_boxes;
		int index = k * blob_insidecount;
		float *temp_ptr = &out_rerange[index];
		for (int i = 0; i < sidenum_; ++i)
		{
			for (int j = 0; j < sidenum_; ++j)
			{
				for (int n = 0; n < num_objects_; n++)
				{
					int index = i*sidenum_*feature_step*num_objects_ + j*num_objects_*feature_step + n*feature_step;
					int c_index = class_step*(i*sidenum_*num_objects_ + j*num_objects_ + n);
					int p_index = i*sidenum_*num_objects_ + j*num_objects_ + n;
					yolobox tempbox = get_region_box(temp_ptr, anchor_vec, n, index, i, j, sidenum_, sidenum_);
					tempbox.score = temp_ptr[index + 4];
					result_boxes.push_back(tempbox);
					tempclassarray = &temp_ptr[index + 5];
					float temp_probeclass = softmax_region(tempclassarray, num_class_);
					for (int c = 0; c < num_class_; c++)
					{
						class_prob_[c_index + c] = sigmoidfunc(tempbox.score)*tempclassarray[c];
						if (class_prob_[c_index + c] >= thresh_score_)
						{
							//printf("above thresh!\n");
						}
						fprobs_[p_index][c] = (class_prob_[c_index + c] < thresh_score_) ? 0.0f : class_prob_[c_index + c];
					}
				}
			}
		}
		std::vector<yolobox> bboxes_nms = Do_NMSMultiLabel(result_boxes, fprobs_, num_class_,thresh_score_, thresh_iou_);
		if (bboxes_nms.size() > 0) {
			total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
		}
	}
	return total_boxes_;
}

int main(int argc, char* argv[])
{

	
	Mat frame=cv::Mat();
	Mat prev_frame=cv::Mat();
	Mat showFrame;


	if (argc != 2){
	    printf("Error: wrong commad format, usage:\n");
	    printf("%s <binaryfile>\n", argv[0]);
	    return 0;
	}
	
	char *image_name=argv[1];
	
	
	
	fpga_object_init();

	frame = cv::imread(image_name);

	
	dealImagetoRGBandInt8(frame,image);
	
	loadImageToBuffer(image);
	
	
	prev_frame = frame.clone();

	/*fpga process*/
	fpga_object_main();

	
	if(!frame.empty())
	{
		results = YoloV2BoxDetect(data_str,1,1, 17, 5, 0.8, 0.45);

		auto prev_detection_results = results;
		
		for (auto &result : prev_detection_results) {
			
			auto genderColor = cv::Scalar(0, 0, 255); 
			int x,y,w,h;
			x = (int)((result.x - result.w / 2)*frame.cols);
			y = (int)((result.y - result.h / 2)*frame.rows);
			w = (int)((result.w)*frame.cols);
			h = (int)((result.h)*frame.rows);
			cv::rectangle(frame, cv::Rect(x, y, w, h), genderColor, 2);
			
        }
		
		flip(frame, showFrame, 1);
		
		
		cv::imwrite("./result/result.png", showFrame);
		
	}



	
	cleanup();
	
	return 0;
	
}


int fpga_object_init()
{	
	cl_int status;
	unsigned int weight_buf_size;
	

	printf("***************************************************\n");
	printf("PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs \n");
	printf("***************************************************\n");

	// Connect to the desired platform
	platform_id = findPlatform(vendor_name);
	if(platform_id == NULL) {
		printf("ERROR: Unable to find the desired OpenCL platform.\n");
		return false;
	}

	// Query the available OpenCL device
	device.reset(getDevices(platform_id, DEVICE_TYPE, &num_devices));
	printf("\nPlatform: %s\n", getPlatformName(platform_id).c_str());
	printf("Using %d device(s)\n", num_devices);
	for(unsigned i = 0; i < num_devices; ++i) {
		printf("  Device %d: %s\n", i, getDeviceName(device[i]).c_str());
		displayDeviceInfo(device[i]);
	}


	// Create the context.
	context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
	checkError(status, "Failed to create context");

	// Create Program Objects
	char *kernel_file_name="conv_pipe.aocx";

	// Create the program for all device. All devices execute the same kernel.
	program = createProgramFromFile(context, (const char *) kernel_file_name, device, num_devices);

	// Create per-device objects.
	que_memRd.reset(num_devices);
	//que_conv.reset(num_devices);
	que_memWr.reset(num_devices);
	//que_memWrReorg.reset(num_devices);
	//que_pool.reset(num_devices);
	knl_memRd.reset(num_devices);
	//knl_conv.reset(num_devices);
	knl_memWr.reset(num_devices);
//	knl_memWrReorg.reset(num_devices);
//	knl_pool.reset(num_devices);
//	knl_lrn.reset(num_devices);
	// For each layer a group of buffers are created to store the weights and bias
	weights_buf.reset(num_devices*LAYER_NUM);
	bias_buf.reset(num_devices*LAYER_NUM);
	// Two buffers (data and output) are used as ping-pong buffers for conv layers
	data_buf.reset(num_devices*MAX_BATCH_SIZE);
	output_buf.reset(num_devices*MAX_BATCH_SIZE);
	// Two buffers are used as ping-pong buffers for fc layers
	concat_buf.reset(num_devices);
	conv_res_buf.reset(num_devices);


	// Prepare compute data
	status = prepare();
	if(status == 1){
		printf("Allocate memory for data and weights failed !!!\n");
		return false;
	}

	// Create qeues, kernels and mem objs
	for(unsigned i = 0; i < num_devices; ++i) {
		// Command queue
		que_memRd[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 0");
		//que_conv[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
		//checkError(status, "Failed to create command queue 1");
		que_memWr[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 2");
		//que_memWrReorg[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
		//checkError(status, "Failed to create command queue 3");
		//que_pool[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
		//checkError(status, "Failed to create command queue 4");

		// Kernel
		knl_memRd[i] = clCreateKernel(program, knl_name_memRd, &status);
		checkError(status, "Failed to create memRd kernel");

		//knl_conv[i] = clCreateKernel(program, knl_name_conv, &status);
		//checkError(status, "Failed to create conv kernel");

		//knl_pool[i] = clCreateKernel(program, knl_name_Pool, &status);
		//checkError(status, "Failed to create pooling kernel");

		knl_memWr[i] = clCreateKernel(program, knl_name_memWr, &status);
		checkError(status, "Failed to create memWr kernel");

		//knl_memWrReorg[i] = clCreateKernel(program, knl_name_memWriteReorg, &status);
		//checkError(status, "Failed to create memWr kernel");

		//knl_lrn[i] = clCreateKernel(program, knl_name_lrn, &status);
		//checkError(status, "Failed to create lrn kernel");

		// Mems
		// Create weight and bias buffers for each layer
		for(unsigned j = 0; j < LAYER_NUM; ++j){

			//weight_buf_size = layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m];
			//weight_buf_size = layer_config[j][weight_w]*2*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m];
			weight_buf_size = 6*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m];
			//printf("weights_buf zise:weight_buf_size=%d,weight_w=%d,weight_h=%d,weight_n=%d,weight_m=%d\n",weight_buf_size,layer_config[j][weight_w],layer_config[j][weight_h],layer_config[j][weight_n],layer_config[j][weight_m]);
#if defined(USE_SDX_1DDR)
			// Weights buffers for each layer
			weights_buf[i*LAYER_NUM+j] = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,weight_buf_size* sizeof(DTYPE),weight_conv[j],&status);
			checkError(status, "Failed to create buffer for weights in layer");

			// Bias buffers for each layer
			bias_buf[i*LAYER_NUM+j] = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,layer_config[j][bias_size] * sizeof(DTYPE),bias_conv[j],&status);
			checkError(status, "Failed to create buffer for bias in layer");

			clEnqueueMigrateMemObjects(que_memRd[i],1, &weights_buf[i*LAYER_NUM+j],0 /* flags, 0 means from host*/,0, NULL,&write_event[i]);

			clEnqueueMigrateMemObjects(que_memRd[i],1, &bias_buf[i*LAYER_NUM+j],0 /* flags, 0 means from host*/,0, NULL,&write_event[i]);
#elif defined(USE_SDX_4DDR)
			// Weights buffers for each layer
			cl_mem_ext_ptr_t weights_buf_ext , bias_buf_ext;
			weights_buf_ext.flags= XCL_MEM_DDR_BANK1;//Select DDR1
			weights_buf_ext.obj = NULL;
			weights_buf_ext.param = 0;
			weights_buf[i*LAYER_NUM+j] = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,weight_buf_size* sizeof(DTYPE),&weights_buf_ext,&status);
			checkError(status, "Failed to create buffer for weights in layer");

			// Bias buffers for each layer
			bias_buf_ext.flags= XCL_MEM_DDR_BANK2;//Select DDR2
			bias_buf_ext.obj = NULL;
			bias_buf_ext.param = 0;
			bias_buf[i*LAYER_NUM+j] = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,layer_config[j][bias_size] * sizeof(DTYPE),&bias_buf_ext,&status);
			checkError(status, "Failed to create buffer for bias in layer");

			// Initializing all weights buffers, blocking write is used
			status = clEnqueueWriteBuffer(que_memRd[i], weights_buf[i*LAYER_NUM+j], CL_TRUE,0, weight_buf_size*sizeof(DTYPE), weight_conv[j], 0, NULL, NULL);
			checkError(status, "Failed to transfer weight");

			status = clEnqueueWriteBuffer(que_memRd[i], bias_buf[i*LAYER_NUM+j], CL_TRUE,0, layer_config[j][bias_size] * sizeof(DTYPE), bias_conv[j], 0, NULL, NULL);
			checkError(status, "Failed to transfer bias");
#else // IntelFPGA
			// Weights buffers for each layer
			weights_buf[i*LAYER_NUM+j] = clCreateBuffer(context, CL_MEM_READ_ONLY, weight_buf_size* sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for weights in layer");

			// Bias buffers for each layer
			bias_buf[i*LAYER_NUM+j] = clCreateBuffer(context, CL_MEM_READ_ONLY, layer_config[j][bias_size] * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for bias in layer");

			// Initializing all weights buffers, blocking write is used
			status = clEnqueueWriteBuffer(que_memRd[i], weights_buf[i*LAYER_NUM+j], CL_TRUE, 0, weight_buf_size*sizeof(DTYPE), weight_conv[j], 0, NULL, NULL);
			checkError(status, "Failed to transfer weight");

			status = clEnqueueWriteBuffer(que_memRd[i], bias_buf[i*LAYER_NUM+j], CL_TRUE, 0, layer_config[j][bias_size] * sizeof(DTYPE), bias_conv[j], 0, NULL, NULL);
			checkError(status, "Failed to transfer bias");
#endif
		}

		// Create data buffers for each batch item
		for(unsigned j = 0; j < input_config[batch_size]; ++j){
#if defined(USE_SDX_1DDR)
			// Input data buffers
			data_buf[i*input_config[batch_size]+j] = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,(layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]) * sizeof(DTYPE), data_init, &status);
			checkError(status, "Failed to create buffer for data in layer");

			// Output results buffers
			output_buf[i*input_config[batch_size]+j] = clCreateBuffer(context,CL_MEM_READ_WRITE,OUT_BUF_SIZE * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for output");

			clEnqueueMigrateMemObjects(que_memRd[i],1, &data_buf[i*input_config[batch_size]+j], 0 /* flags, 0 means from host*/,0, NULL,&write_event[i]);

#elif defined(USE_SDX_4DDR)
			// Input data buffers
			cl_mem_ext_ptr_t data_buf_ext,output_buf_ext;
			data_buf_ext.flags = XCL_MEM_DDR_BANK0;//Select DDR0
			data_buf_ext.obj = NULL;
			data_buf_ext.param = 0;
			data_buf[i*input_config[batch_size]+j] = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,IN_BUF_SIZE * sizeof(DTYPE),&data_buf_ext,&status);
			checkError(status, "Failed to create buffer for data in layer");

			// Output results buffers
			output_buf_ext.flags = XCL_MEM_DDR_BANK0;//Select DDR0
			output_buf_ext.obj = NULL;
			output_buf_ext.param = 0;
			output_buf[i*input_config[batch_size]+j] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,OUT_BUF_SIZE * sizeof(DTYPE),&output_buf_ext,&status);
			checkError(status, "Failed to create buffer for output");
#else // IntelFPGA
			// Input data buffers
			data_buf[i*input_config[batch_size]+j] = clCreateBuffer(context, CL_MEM_READ_WRITE, IN_BUF_SIZE * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for data in layer");

			// Output results buffers
			output_buf[i*input_config[batch_size]+j] = clCreateBuffer(context, CL_MEM_READ_WRITE, OUT_BUF_SIZE * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for output");

			// Concat buffers
			concat_buf[i*input_config[batch_size]+j] = clCreateBuffer(context,	CL_MEM_READ_WRITE, CONCAT_BUF_SIZE * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for concat layer");

			//Conv result buffers
			conv_res_buf[i*input_config[batch_size]+j] = clCreateBuffer(context,  CL_MEM_READ_WRITE, CONV_RES_BUF_SIZE * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for save layer17-conv result");
#endif
		}
#ifdef USE_SDX_4DDR
		cl_mem_ext_ptr_t concat_buf_ext,conv_res_buf_ext;
		concat_buf_ext.flags = XCL_MEM_DDR_BANK0; //Select DDR0
		concat_buf_ext.obj = NULL;
		concat_buf_ext.param = 0;
		// Allocate fc buffers
		concat_buf[i] = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,FC_BUF_SIZE * sizeof(DTYPE),&concat_buf_ext,&status);
		checkError(status, "Failed to create buffer for data in fc layer");

		conv_res_buf_ext.flags = XCL_MEM_DDR_BANK0;//Select DDR0
		conv_res_buf_ext.obj = NULL;
		conv_res_buf_ext.param = 0;
		conv_res_buf[i] = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,FC_BUF_SIZE * sizeof(DTYPE),&conv_res_buf_ext,&status);
		checkError(status, "Failed to create buffer for data in fc layer");
#else // IntelFPGA
		// Allocate fc buffers
		//concat_buf[i] = clCreateBuffer(context,  CL_MEM_READ_WRITE, CONCAT_BUF_SIZE * sizeof(DTYPE), NULL, &status);
		//checkError(status, "Failed to create buffer for concat layer");

		//conv_res_buf[i] = clCreateBuffer(context,  CL_MEM_READ_WRITE, CONV_RES_BUF_SIZE * sizeof(DTYPE), NULL, &status);
		//checkError(status, "Failed to create buffer for save layer17-conv result");
#endif
		}

	return 0;

}

int fpga_object_main()
{
	cl_int status;

	unsigned int conv_output_num;
	unsigned int conv_loop_cnt;
	unsigned char conv_control;
	unsigned char conv_fc_en;
	//unsigned int pool_input_num;
	unsigned short pool_line_size;
	unsigned short pool_col_size;
	unsigned char pool_bypass;


	unsigned int memwr_output_num;
	unsigned int conv_out_loopnum;
    //unsigned short conv_row_num;
    unsigned char conv_row_rem;
    unsigned char conv_row_rem_flag;

	Timer t;  // Timer used for performance measurement
	float time;

	// Execute the kernel
	scoped_array<cl_event> memRd_event(num_devices);
	//scoped_array<cl_event> conv_event(num_devices);
	//scoped_array<cl_event> pool_event(num_devices);
	scoped_array<cl_event> memWr_event(num_devices);
	//scoped_array<cl_event> ReorgMemWr_event(num_devices);
	//scoped_array<cl_event> lrn_event(num_devices);

	// Recorde the excution time of each operation for each layer

	unsigned       iter_num;
	//unsigned short out_dim1xbatch;
	//unsigned int   out_dim1x2xbatch;
	//unsigned char  padding_offset;
	unsigned char  conv_group_num_dim1;
	unsigned int   conv_group_num_dim2;
	unsigned char  conv_win_size_dim1, conv_win_size_dim2;
	unsigned int   conv_win_size_dim1x2x3;
	unsigned char  conv_group_rem_dim1, conv_group_rem_dim2;
	unsigned int   conv_group_rem_dim1x2x3;
	unsigned short data_dim1;
	unsigned int   data_dim1x2;
	unsigned char  weight_dim1x2;
	unsigned int   weight_dim1x2x3;
	unsigned short weight_dim4_div_LaneNum;
	unsigned short conv_win_size;
	unsigned int    group_num_mul_win_size;
	unsigned int   pic_num =1;
	unsigned char  group_size_x;

	// Kernel excutions main loops
	for(unsigned i = 0; i < num_devices; ++i) {
		// Recorde the start time
		t.start();

		// Each iteration excutes one layer convolution
		// MemRd -> Conv(Relu) -> (MaxPool) -> MemWr -> (Lrn)
		for(unsigned char j = 0; j < LAYER_NUM; ++j){

			if(j<CONV_NUM)
				iter_num = input_config[batch_size]; // for conv layers, process by batch_size time
			else
				iter_num = 1; // for FC layers, process only one time

			// Each iteration process one item in batch
			for(unsigned k = 0; k < iter_num; ++k){
			// Set Arguments
			//
			// Set knl_memRd arguments.
			unsigned argi = 0;

			// Convolution tasks (conv_x,conv_y) are divided into multiple groups
            //if (layer_config[j][weight_w]==3) {
                group_size_x = CONV_GP_SIZE_X;
            //} else {
            //    group_size_x = CONV_GP_SIZE_X + 2;
            //}
			//conv_group_num_dim1   = ceil((float)layer_config[j][conv_x]/CONV_GP_SIZE_X);
			//conv_group_num_dim1   = ceil((float)layer_config[j][conv_x]/group_size_x);
			
            conv_row_rem = (layer_config[j][conv_x]+2*layer_config[j][conv_padding])%group_size_x;
            conv_row_rem_flag = layer_config[j][weight_w] == 3 && (conv_row_rem==1 || conv_row_rem==2);
			conv_group_num_dim1   = ceil((float)(layer_config[j][conv_x]+2*layer_config[j][conv_padding])/group_size_x);
			conv_group_num_dim2   = ceil((float)layer_config[j][conv_y]/CONV_GP_SIZE_Y);

			if(layer_config[j][conv_x]==1){
				conv_win_size_dim1  = layer_config[j][weight_w];
				conv_group_rem_dim1   = layer_config[j][weight_w];
			}
			else{
				conv_win_size_dim1  = layer_config[j][weight_w]+(CONV_GP_SIZE_X-1)*layer_config[j][conv_stride];
				if(layer_config[j][conv_x]%CONV_GP_SIZE_X==0)
					conv_group_rem_dim1   = CONV_GP_SIZE_X*layer_config[j][weight_w];
				else
					conv_group_rem_dim1   = layer_config[j][conv_x]%CONV_GP_SIZE_X*layer_config[j][weight_w];
			}
			conv_win_size_dim2    = layer_config[j][weight_h];
			conv_group_rem_dim2   = layer_config[j][weight_h];
			conv_win_size_dim1x2x3  = conv_win_size_dim1*conv_win_size_dim2*layer_config[j][weight_n];
			conv_group_rem_dim1x2x3 = conv_group_rem_dim1*conv_group_rem_dim2*layer_config[j][weight_n];

            //data_dim1 = ceil((float)(layer_config[j][data_w] + 2*layer_config[j][conv_padding])/group_size_x) * group_size_x;

			weight_dim4_div_LaneNum = layer_config[j][weight_m]/LANE_NUM;
			data_dim1x2 = layer_config[j][data_w]*layer_config[j][data_h];
			//data_dim1x2 = data_dim1*layer_config[j][data_h];
			weight_dim1x2 = layer_config[j][weight_w]*layer_config[j][weight_h];
			weight_dim1x2x3 = layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n];

            //conv_win_size = layer_config[j][weight_w]*2*layer_config[j][weight_h]*layer_config[j][weight_n] / (VEC_SIZE*W_VEC_SIZE); 
            conv_win_size = layer_config[j][weight_h]*layer_config[j][weight_n]/VEC_SIZE; 

            //group_num_mul_win_size = (weight_dim4_div_LaneNum * conv_group_num_dim1 * conv_group_num_dim2) * conv_win_size;
            //group_num_mul_win_size = (weight_dim4_div_LaneNum * ceil((float)layer_config[j][conv_x]/group_size_x) * conv_group_num_dim2+1) * conv_win_size;
            group_num_mul_win_size = (weight_dim4_div_LaneNum*conv_group_num_dim1*conv_group_num_dim2+2)*conv_win_size;
              
            //printf("host:group_num_mul_win_size=%d\n",group_num_mul_win_size);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &j);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &conv_row_rem);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_ushort), &layer_config[j][data_w]);
			//status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_ushort), &data_dim1);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_ushort), &layer_config[j][data_h]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &data_dim1x2);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][weight_w]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][weight_h]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_ushort), &layer_config[j][weight_n]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_ushort), &weight_dim4_div_LaneNum);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &weight_dim1x2);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint),  &weight_dim1x2x3);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_ushort), &layer_config[j][conv_x]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &group_size_x);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][conv_stride]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][conv_padding]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][conv_split]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &conv_group_num_dim1);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &conv_group_num_dim2);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &conv_group_rem_dim1);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &group_num_mul_win_size);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &conv_group_rem_dim1x2x3);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_ushort), &conv_win_size);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &conv_win_size_dim1);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &conv_win_size_dim2);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &conv_win_size_dim1x2x3);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
			// Select the kernel input mem object source
			// data_buf -> conv1 -> output_buf -> lrn1 -> data_buf -> conv2 -> output_buf -> lrn2 -> data_buf
			// -> conv3 -> output_buf -> conv4 -> output_buf -> ...
			if(layer_config[j][memrd_src]==0){
				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &data_buf[i*input_config[batch_size]+k]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
			}
			else if(layer_config[j][memrd_src]==1)
			{
				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &output_buf[i*input_config[batch_size]+k]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
			}
			else if(layer_config[j][memrd_src]==2)
			{
				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &concat_buf[i*input_config[batch_size]+k]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
			}
			else // 3
			{
				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &conv_res_buf[i]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
			}

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &weights_buf[i*LAYER_NUM+j]);
			checkError(status, "Failed to set argument %d kernel memRd", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &bias_buf[i*LAYER_NUM+j]);
			checkError(status, "Failed to set argument %d kernel memRd", argi - 1);

			//  Set knl_conv arguments.
			//argi = 0;
			//if(k == 0&&pic_num==1)
			//	printf("\nSetting kernel conv arg Layer %d:\n", j+1);

            //if (layer_config[j][weight_w] == 1) {
			//    //conv_loop_cnt = layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n]/VEC_SIZE;
			//    conv_loop_cnt = layer_config[j][weight_w]*6*layer_config[j][weight_h]*layer_config[j][weight_n]/(VEC_SIZE*W_VEC_SIZE);
			//    conv_output_num = ceil((float)layer_config[j][conv_x]/W_VEC_SIZE)*layer_config[j][conv_y]*layer_config[j][weight_m]/LANE_NUM; // new weight_m is divisible by LANE_NUM
            //}else {
			    //conv_loop_cnt = (layer_config[j][weight_w]+(group_size_x-1)*layer_config[j][conv_stride])*layer_config[j][weight_h]*layer_config[j][weight_n]/(VEC_SIZE*W_VEC_SIZE);
			    conv_loop_cnt = layer_config[j][weight_h]*layer_config[j][weight_n]/VEC_SIZE;
			    conv_output_num = ceil((float)layer_config[j][conv_x]/group_size_x)*layer_config[j][conv_y]*layer_config[j][weight_m]/LANE_NUM; // new weight_m is divisible by LANE_NUM

                conv_out_loopnum = conv_output_num * conv_loop_cnt;

                //printf("host:conv_out_loopnum=%d\n",conv_out_loopnum);
            //}
			conv_control = (layer_config[j][conv_relu]&0x01)|(((~layer_config[j][pool_on])&0x01)<<1)|((layer_config[j][conv_res_save]&0x01)<<2);
            conv_fc_en = layer_config[j][weight_w] == 1;

			//status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uchar), &j);
			//checkError(status, "Failed to set argument %d of kernel conv", argi - 1);


			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &conv_out_loopnum);
			checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &conv_loop_cnt);
			checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &conv_control);
			checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &conv_fc_en);
			checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_char), &precision_config[j][frac_w]);
			checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_char), &precision_config[j][frac_b]);
			checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_char), &precision_config[j][frac_din]);
			checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_char), &precision_config[j][frac_dout]);
			checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

			//pool_input_num = ceil((float)layer_config[j][conv_x]/(W_VEC_SIZE-2))*layer_config[j][conv_y]*layer_config[j][weight_m]/LANE_NUM; // new weight_m is divisible by LANE_NUM
			pool_col_size = ceil((float)layer_config[j][conv_x]/(W_VEC_SIZE-2));
			pool_line_size = layer_config[j][conv_y];

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_ushort), &pool_line_size);
			checkError(status, "Failed to set argument %d of kernel pool", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_ushort), &pool_col_size);
			checkError(status, "Failed to set argument %d of kernel pool", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][pool_size]);
			checkError(status, "Failed to set argument %d of kernel pool", argi - 1);

			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][pool_stride]);
			checkError(status, "Failed to set argument %d of kernel pool", argi - 1);
			

			unsigned short memWr_dim1;
			unsigned short memWr_dim1_orig;
			unsigned short memWr_dim2;
			unsigned short memWr_dim3;
			//unsigned short ReorgMemWr_dim1;
			//unsigned short ReorgMemWr_dim2;
			//unsigned short ReorgMemWr_dim3;
            unsigned char  memwr_scal;
            unsigned char  memwr_scal_rem_z;
            unsigned short out_dim3_div_scalxlaneNum;
            unsigned int   memwr_dim1xdim2;
            unsigned char  q_vec;
            unsigned char  rem_size_x;
            unsigned char  start_size_x;
            unsigned char  memWr_dim1_div_q_vec;
            unsigned short memWr_dim3_div_lane_numxscal;
            unsigned char  scalxq_vec;
            unsigned char  scalxrem_size_x;
            unsigned char  scalxstart_size_x;
            unsigned char  scal_rem_zxq_vec;
            unsigned char  scal_rem_zxrem_size_x;
            unsigned char  scal_rem_zxstart_size_x;
            unsigned int   dim_z_edge_num;
            unsigned char  next_layer_padding;

            if (j<LAYER_NUM-1)
                next_layer_padding=layer_config[j+1][conv_padding];
            else 
                next_layer_padding=0;

			pool_bypass = (~layer_config[j][pool_on])&0x01;

			
            if(layer_config[j][pool_on]==1){
				//memWr_dim1 = layer_config[j][pool_x];
                
				memWr_dim1_orig = layer_config[j][pool_x];
				//memWr_dim1 = ceil(((float)layer_config[j][pool_x]+2*next_layer_padding)/4)*4;
				memWr_dim2 = layer_config[j][pool_y];
				memWr_dim3 = layer_config[j][pool_z];
                //memWr_dim1_div_q_vec=ceil((float)layer_config[j][pool_x]/q_vec);
			}
			else{
				//memWr_dim1 = layer_config[j][conv_x];
				memWr_dim1_orig = layer_config[j][conv_x];
				//memWr_dim1 = ceil(((float)layer_config[j][conv_x]+2*next_layer_padding)/4)*4;
				memWr_dim2 = layer_config[j][conv_y];
				memWr_dim3 = layer_config[j][conv_z];
                //memWr_dim1_div_q_vec=ceil((float)layer_config[j][conv_x]/q_vec);
			}
			if (j<LAYER_NUM-1)
                memWr_dim1 = ceil(((float)memWr_dim1_orig+2*next_layer_padding)/4)*4;
            else
                memWr_dim1 = memWr_dim1_orig;
            
            if (j==0) 
                memwr_scal = 2;
            else
                memwr_scal = LANE_NUM/VEC_SIZE;

            if((memWr_dim3%LANE_NUM)!=0)
                memwr_scal_rem_z = (memWr_dim3%LANE_NUM)/VEC_SIZE;
            else
                memwr_scal_rem_z = LANE_NUM/VEC_SIZE;

            memwr_dim1xdim2 = memWr_dim1*memWr_dim2;
            
            if(layer_config[j][pool_on]==1)
                q_vec=2;
            else if (layer_config[j][weight_w]==1)
                q_vec=4;
            else if (layer_config[j][weight_w]==3)
                q_vec=4;
            
            start_size_x=q_vec+next_layer_padding;
            rem_size_x=memWr_dim1_orig%q_vec==0 ? (q_vec+next_layer_padding) : (memWr_dim1_orig%q_vec+next_layer_padding);
            //memWr_dim1_div_q_vec=ceil((float)memWr_dim1/q_vec);
            memWr_dim1_div_q_vec=ceil((float)memWr_dim1_orig/q_vec);
            memWr_dim3_div_lane_numxscal=(memWr_dim3/LANE_NUM*memwr_scal);
			//memwr_output_num = ((ceil((float)memWr_dim1/q_vec)-1)*q_vec+rem_size_x)*memwr_scal*memWr_dim2*memWr_dim3/LANE_NUM; // new weight_m is divisible by LANE_NUM
			//memwr_output_num = memWr_dim1*memwr_scal*memWr_dim2*memWr_dim3/LANE_NUM; // new weight_m is divisible by LANE_NUM
            if((memWr_dim3%LANE_NUM)!=0) 
			    //memwr_output_num = (memWr_dim1*memwr_scal*memWr_dim2*(memWr_dim3/LANE_NUM))+(memWr_dim1*memwr_scal_rem_z*memWr_dim2); // new weight_m is divisible by LANE_NUM
                memwr_output_num=(memWr_dim1_orig+2*next_layer_padding)*memWr_dim2*memWr_dim3/VEC_SIZE;
            else
			    memwr_output_num = (memWr_dim1_orig+2*next_layer_padding)*memwr_scal*memWr_dim2*memWr_dim3/LANE_NUM; // new weight_m is divisible by LANE_NUM

                //printf("host:memwr_output_num=%d\n",memwr_output_num);

            scalxq_vec=memwr_scal*q_vec; 
            scalxrem_size_x=memwr_scal*rem_size_x;
            scalxstart_size_x=memwr_scal*start_size_x;
            scal_rem_zxq_vec=memwr_scal_rem_z*q_vec;
            scal_rem_zxrem_size_x=memwr_scal_rem_z*rem_size_x;
            scal_rem_zxstart_size_x=memwr_scal_rem_z*start_size_x;
             
            dim_z_edge_num=(memWr_dim1_orig+2*next_layer_padding)*memwr_scal*memWr_dim2*(memWr_dim3/LANE_NUM);
  
             
			argi = 0;

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &j);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uint), &memwr_output_num);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &q_vec);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &rem_size_x);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &start_size_x);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &memWr_dim1_div_q_vec);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &next_layer_padding);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_ushort), &memWr_dim3_div_lane_numxscal);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_ushort), &memWr_dim1);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_ushort), &memWr_dim2);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			//status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_ushort), &weight_dim4_div_LaneNum);
			//checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uint), &memwr_dim1xdim2);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &pool_bypass);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &memwr_scal);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &memwr_scal_rem_z);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &scalxq_vec);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &scalxrem_size_x);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &scalxstart_size_x);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &scal_rem_zxq_vec);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &scal_rem_zxrem_size_x);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &scal_rem_zxstart_size_x);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uint), &dim_z_edge_num);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			// Select the kernel output mem object source
			if(layer_config[j][memwr_dst]==0){
				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &data_buf[i*input_config[batch_size]+k]);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
			}
			else if(layer_config[j][memwr_dst]==1)
			{
				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &output_buf[i*input_config[batch_size]+k]);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
			}
			else if(layer_config[j][memwr_dst]==2)
			{   
                status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), (&concat_buf[i*input_config[batch_size]+k]));
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
			}
			else // 3
			{
				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &conv_res_buf[i]);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
			}

#ifdef USE_SDX_1DDR
			// Wait until all data are send to DDR memory
			clWaitForEvents(num_devices, write_event);
			clReleaseEvent(write_event[0]);
			checkError(status, "Failed to release buffer write event object");
#endif
			// Excutes Kerne
			
			status = clEnqueueTask(que_memRd[i], knl_memRd[i], 0, NULL, &memRd_event[i]);
			checkError(status, "Failed to launch kernel memRD kernel");

#ifdef XILINX
                status = clEnqueueTask(que_memWr[i], knl_memWr[i], 0, NULL, &memWr_event[i]);
#else // IntelFPGA
            //    printf("\naaaaaaaaaaaaaaaa\n");
                //status = clEnqueueNDRangeKernel(que_memWr[i], knl_memWr[i], 3, NULL, knl_memWr_global_size, knl_memWr_local_size, 0, NULL, &memWr_event[i]);
#endif
			    //checkError(status, "Failed to launch kernel memWr");
            //}

			// kernel ReorgMemWr
			//if ((layer_config[j][pool_on]==0) || (layer_config[j][conv_res_save]==1)){
			//    knl_memWrReorg_global_size[0] = ReorgMemWr_dim1;
			//    knl_memWrReorg_global_size[1] = ReorgMemWr_dim2;
			//    knl_memWrReorg_global_size[2] = layer_config[j][weight_m]; // pool_z equals original weight_m, new weight_m is divisible by LANE_NUM
			//    knl_memWrReorg_local_size[0] = 1;
			//    knl_memWrReorg_local_size[1] = 1;
			//    knl_memWrReorg_local_size[2] = LANE_NUM;

			//    if(k == 0&&pic_num==1)
			//    	printf("\nLaunching kernel ReorgMemWr with local size: %d, %d, %d  (global size: %d, %d, %d)\n",
			//    						(int)knl_memWrReorg_local_size[0], (int)knl_memWrReorg_local_size[1], (int)knl_memWrReorg_local_size[2],
			//    						(int)knl_memWrReorg_global_size[0], (int)knl_memWrReorg_global_size[1], (int)knl_memWrReorg_global_size[2]);
#ifdef XILINX
                status = clEnqueueTask(que_memWr[i], knl_memWr[i], 0, NULL, &memWr_event[i]);
#else // IntelFPGA
			
			status = clEnqueueTask(que_memWr[i], knl_memWr[i], 0, NULL, &memWr_event[i]);
#endif
			    checkError(status, "Failed to launch kernel MemWr");
            //}


		    //if ((layer_config[j][pool_on]==0) || (layer_config[j][conv_res_save]==1)){
            //    status = clWaitForEvents(num_devices, ReorgMemWr_event);
		    //    checkError(status, "Failed to finish ReorgMemWR event");
            //}
            //printf("\nbbbbbbbbbbbbbbbbbb\n");
		    //if (layer_config[j][pool_on]==1){
                //status = clWaitForEvents(num_devices, memWr_event);
		        //checkError(status, "Failed to finish memWR event");
            //}
            //    status = clWaitForEvents(num_devices, pool_event);
		    //    checkError(status, "Failed to finish memWR event");
            //printf("\nccccccccccccccccccc\n");


			// Must release event object to avoid performance degeneration !!!
            //status = clWaitForEvents(num_devices, memRd_event);
		    //checkError(status, "Failed to finish memRd event");
			clReleaseEvent(memRd_event[i]);
			checkError(status, "Failed to release memRD event object");
			//clReleaseEvent(conv_event[i]);
			//checkError(status, "Failed to release Conv event object");
			//if(layer_config[j][pool_on]){
			//	status = clReleaseEvent(pool_event[i]);
			//	checkError(status, "Failed to release pool event object");
			//}
			clReleaseEvent(memWr_event[i]);
			checkError(status, "Failed to release memWR event object");
			//if((layer_config[j][pool_on]==0) || (layer_config[j][conv_res_save]==1)){
			//    clReleaseEvent(ReorgMemWr_event[i]);
			//    checkError(status, "Failed to release ReorgMemWR event object");
			//}
// 			if(layer_config[j][lrn_on]){
//				status = clReleaseEvent(lrn_event[i]);
//				checkError(status, "Failed to release lrn event object");
//			}

			}// end of batch iteration


		}// end of layer iteration

        status = clWaitForEvents(num_devices, memWr_event);
		checkError(status, "Failed to finish memWR event");
		t.stop();
		time = t.get_time_s();
		
		printf("readDataBack enter\n");
		readDataBack();

		printf("readDataBack finish\n");
		if(output_reorder){
				for(register int i=0; i<RESULT_OUT_NUM; i++){
					data_str[i] = (float)(output_reorder[i]*pow(2,-1*precision_config[LAYER_NUM-1][frac_dout]));
			}
		}


	}// end of board iteration


	printf("Total runtime: %fs \n\n", time);
	
	return EXIT_SUCCESS;
}

void readDataBack()
{
	unsigned int read_buf_size;
	cl_int status;
	scoped_array<cl_event> finish_event(num_devices);
	// Read back the results from the device to verify the output
	// Note：only device0 is used here
	if(num_devices!=1)
		printf("Warnning: only the result from device0 will be verified!!!\n\n");

	// Select whith item you would like to compare with the golden ref
	// Item num start from 0
	unsigned batch_item_num = 0;
	if(batch_item_num>(input_config[batch_size]-1)){
		printf("Error: wrong configuration，can't verify the item since it is layer than batch size !!!\n\n");
	}

	if(LAYER_NUM<CONV_NUM){ // verify conv results
		read_buf_size = output_config[output_w]*output_config[output_h]*output_config[output_n];

	}
	else // verify the last conv and all fc results
		read_buf_size = output_config[output_w]*output_config[output_h]*output_config[output_n]*input_config[batch_size];

	// For the last conv layer and all fc layers, read result from one of the fc buffers
	if(layer_config[LAYER_NUM-1][memwr_dst] == 2){
		printf("\nCopyed all batched results from No. %d concat buffers.\n", batch_item_num);
		status = clEnqueueReadBuffer(que_memWr[0], concat_buf[batch_item_num], CL_FALSE,          // read from device0
			0, sizeof(DTYPE) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);
		checkError(status, "Failed to set transfer output data");
	}
	else if(layer_config[LAYER_NUM-1][memwr_dst] == 3){
		printf("\nCopyed all batched results from NO.%d conv_res buffers.\n", batch_item_num);
		status = clEnqueueReadBuffer(que_memWr[0], conv_res_buf[batch_item_num], CL_FALSE,          // read from device0
			0, sizeof(DTYPE) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);
		checkError(status, "Failed to set transfer output data");
	}
	// For other layers, read results from data and output buffers
	else if(layer_config[LAYER_NUM-1][memwr_dst] == 1){// if lrn is used, the mem dst is changed back to src
		printf("\nCopyed one result from NO.%d output buffers.\n", batch_item_num);
		status = clEnqueueReadBuffer(que_memWr[0], output_buf[batch_item_num], CL_FALSE,         // read from device0
			0, sizeof(DTYPE) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);
		checkError(status, "Failed to set transfer output data");
	}
	else{
		printf("\nCopyed one results from NO.%d data buffers.\n", batch_item_num);
		status = clEnqueueReadBuffer(que_memWr[0], data_buf[batch_item_num], CL_FALSE,           // read from device0
			0, sizeof(DTYPE) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);
		checkError(status, "Failed to set transfer output data");
	}

	// Wait for reads to finish
	clWaitForEvents(1, &finish_event[0]);
	clReleaseEvent(finish_event[0]);
	checkError(status, "Failed to release finish event object");

	if(LAYER_NUM>=CONV_NUM){  //Select with batch item you would like to verify from the last conv and all fc output
		//printf("Selected item = %d from the combined batch results in fc buffers\n", batch_item_num);
		extractOutput(output, output_one_item, batch_item_num, input_config[batch_size], output_config[output_w], output_config[output_h], output_config[output_n]);
	}
	else{
		if(layer_config[LAYER_NUM-1][pool_on]==1)
			//extractOutput(output, output_one_item, 0, 1, layer_config[LAYER_NUM-1][pool_x], layer_config[LAYER_NUM-1][pool_y], layer_config[LAYER_NUM-1][pool_z]);
		    extractOutput(output, output_one_item, batch_item_num, input_config[batch_size], output_config[output_w], output_config[output_h], output_config[output_n]);
		else
			//extractOutput(output, output_one_item, 0, 1, layer_config[LAYER_NUM-1][conv_x], layer_config[LAYER_NUM-1][conv_y], layer_config[LAYER_NUM-1][conv_z]);
		    extractOutput(output, output_one_item, batch_item_num, input_config[batch_size], output_config[output_w], output_config[output_h], output_config[output_n]);
	}

	reorderOutput(output_one_item, output_reorder, output_config[output_w], output_config[output_h], output_config[output_n]);
}

void printResult(int offset, int num)
{   
    
    
    printf("\n*********output result start!!!*********\n");
    for(int i=0; i<num; i++)
        printf( "output_reorder[%d]=%d, final_result=%f, \n", offset+i, output_reorder[offset+i], (float)(output_reorder[offset+i]*pow(2,-1*precision_config[LAYER_NUM-1][frac_dout])));
    printf("\n*********output result end!!!!!*********\n");
}

void filePrintResult()
{   
    
    FILE *conv_result;
    conv_result=fopen("conv_result.log", "w");
    for(int i=0; i<output_config[output_w]*output_config[output_h]*output_config[output_n]; i++)
        //fprintf( conv_result, "%f\n", (float)(output_reorder[i]));
        fprintf( conv_result, "%f\n", (float)(output_reorder[i]*pow(2,-1*precision_config[LAYER_NUM-1][frac_dout])));
    
}

void verifyResult(int num)
{

#ifdef USE_OPENCV
    int max_label;
    char * substr;
	softmax(output_reorder, output_one_item);
	max_label = getProb(output_one_item);
	// Show the picture
    substr = &synset_buf[max_label][10];

	Mat img = imread(picture_file_path);
    putText(img,substr,Point(20, 50), CV_FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2, 8);
    if(max_label == label[num-1]){
        putText(img,"True",Point(20, 80), CV_FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2, 8);
    } else {
        putText(img,"False",Point(20, 80), CV_FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2, 8);
        printf("False: True_label = %d Inferred_label = %d\n\n", label[num], max_label);
    }
    imshow( "PipeCNN", img);
	cvMoveWindow("PipeCNN",0,0);//set the window's position
    waitKey(600);
#else
	// Validate the results
	printf("\nStart verifying results ...\n");
	unsigned int err_num;
	float std_err;  // standard errors
	unsigned int batch_item_size;
	// Compare each results with the golden reference data
	batch_item_size = output_config[output_w]*output_config[output_h]*output_config[output_n];
	err_num = 0;
	for (unsigned int j = 0; j < batch_item_size; j++){
		std_err = abs(output_reorder[j] - golden_ref[j]);
		if(std_err > 0){
			err_num++;
			if(err_num<10)
				printf("Item=%d is wrong (result=%f, golden_ref=%f)\n", j, (float)output_reorder[j], (float)golden_ref[j]);
		}
	}
	if(err_num>0)
		printf("Totally %d Wrong Results\n", err_num);
	else{
		printf("\nCheck Pass !!!\n");
		softmax(output_reorder, output_one_item);
		getProb(output_one_item);
	}
	// Dump results and golden_ref for debugging
	dumpResult();
#endif
}

void loadImageToBuffer(DTYPE *imageDeal)
{
	
	cl_int status;
	
	Timer t;  // Timer used for performance measurement
	float time;
	t.start();

	
	// Vectorize the input image by a factor of VEC_SIZE
	for(register unsigned n = 0; n<layer_config[0][data_n]/VEC_SIZE; n++){
		for(register unsigned i = 0; i<layer_config[0][data_h]; i++){
			//for(unsigned j = 0; j<layer_config[0][data_w]; j++){
			for(register unsigned j = 0; j<input_config[image_w]; j++){
				for(register unsigned k = 0; k<VEC_SIZE; k++){
					if((n*VEC_SIZE+k)<layer_config_original[0][data_n]){ //  when layer_config[0][data_n] > layer_config_original[0][data_n], only copy valid pixels
						//data_init[n*VEC_SIZE*layer_config[0][data_h]*layer_config[0][data_w] + i*layer_config[0][data_w]*VEC_SIZE + j*VEC_SIZE + k]
					//		= (DTYPE) image[(n*VEC_SIZE+k)*layer_config[0][data_h]*layer_config[0][data_w] + i*layer_config[0][data_w] + j];
						data_init[n*VEC_SIZE*input_config[image_h]*input_config[image_w] + i*input_config[image_w]*VEC_SIZE + j*VEC_SIZE + k]
							= (DTYPE) imageDeal[(n*VEC_SIZE+k)*input_config[image_h]*input_config[image_w] + i*input_config[image_w] + j];
					}
				}
			}
		}
	}
	
	//for(unsigned k = 0; k<548*544*8; k++){
    //        fprintf(fp_data,"%d\n",data_init[ k]);
    //}

	for(unsigned i = 0; i < num_devices; ++i) {
		// Create data buffers for each batch item
		for(unsigned j = 0; j < input_config[batch_size]; ++j){
#if defined(USE_SDX_1DDR)
			clEnqueueMigrateMemObjects(que_memRd[i],1, &data_buf[i*input_config[batch_size]+j], 0 /* flags, 0 means from host*/,0, NULL,&write_event[i]);
#elif defined(USE_SDX_4DDR)
			// Load image data into buffers
			status = clEnqueueWriteBuffer(que_memRd[i], data_buf[i*input_config[batch_size]+j], CL_TRUE, 0, (layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]) * sizeof(DTYPE), data_init, 0, NULL, NULL);
			checkError(status, "Failed to transfer input image");
#else
			// Load image data into buffers
			//status = clEnqueueWriteBuffer(que_memRd[i], data_buf[i*input_config[batch_size]+j], CL_TRUE, 0, (layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]) * sizeof(DTYPE), data_init, 0, NULL, NULL);
			status = clEnqueueWriteBuffer(que_memRd[i], data_buf[i*input_config[batch_size]+j], CL_TRUE, 0, (input_config[image_w]*input_config[image_h]*layer_config[0][data_n]) * sizeof(DTYPE), data_init, 0, NULL, NULL);
			checkError(status, "Failed to transfer input image");
#endif
		}
	}

	t.stop();
	time = t.get_time_s();
	printf("loadImageToBuffer total time&&&&&&&&&&&&&&&&&&&&&&&%f\n",time);
	
}

// Read all input data and golden ref data
int prepare()
{

	// Load Image data, CNN net weights and golden_results
    ifstream bin_file_r;
    unsigned file_size;
	unsigned weight_size;
	unsigned output_size;
	unsigned godref_size;
	int ptr=0;  // original weight and bias offset for each layer

	unsigned char  conv_win_size_dim1, conv_win_size_dim2;

	unsigned padding_offset[LAYER_NUM];

	// Parameter initialization and safty check
	for(unsigned ll=0; ll<LAYER_NUM; ll++){

		// First, backup the original layer configurations
		for(unsigned ii=0; ii<NUM_CONFIG_ITEM; ii++){
			layer_config_original[ll][ii]=layer_config[ll][ii];
		}

		//if((layer_config[ll][data_w]+2*layer_config[ll][conv_padding])%CONV_GP_SIZE_X != 0){
            layer_config[ll][data_w] = ceil((float)(layer_config[ll][data_w]+2*layer_config[ll][conv_padding])/CONV_GP_SIZE_X)*CONV_GP_SIZE_X;
        //}

		// Second, perform padding on dim4, when it is not divisible by LANE_NUM
		if(layer_config[ll][weight_m]%LANE_NUM != 0){
			printf("\nWarnning: layer-%d requires padding zero-value feature maps for give param LANE_NUM=%d\n", ll+1, LANE_NUM);
			layer_config[ll][weight_m] = ceil((float)layer_config[ll][weight_m]/LANE_NUM)*LANE_NUM;
			layer_config[ll][bias_size] = layer_config[ll][weight_m];
			printf("      original num of feature maps is %d, new value is %d\n", layer_config_original[ll][weight_m], layer_config[ll][weight_m]);

			// padding of weight on dim4 is needed
			padding_offset[ll] = layer_config[ll][weight_m] - layer_config_original[ll][weight_m];
			// check if evenly padding on two sides is possible
			if(((layer_config[ll][weight_m]/LANE_NUM)%2!=0) & (layer_config[ll][conv_split]==1)){
				printf("Error: could not perform padding for split mode, weight_m/LANE_NUM must be divisible by 2 !!!\n\n");
				return 1;
			}
			else{ // padding zeros evenly on two sides of dim4
				//padding_offset[ll] = padding_offset[ll]/2;
		        padding_offset[ll] = 0; //for winograd 2019.02.18
				printf("      padding_offset=%d (layer=%d)\n\n", padding_offset[ll], ll+1);
			}

		}
		else{
		    padding_offset[ll] = 0;
		}

		// Check parameters
		if(ll==0){ // check parameters for layer-1
			if(input_config[image_w] != layer_config_original[ll][data_w] ||  input_config[image_h] != layer_config_original[ll][data_h]
				|| input_config[image_n] != layer_config_original[ll][data_n] || input_config[image_n] != layer_config_original[ll][weight_n]){
					printf("Error: incorrect layer configuration for layer-%d !!!\n", ll+1);
					//return 1;
				}

			if((layer_config_original[ll][weight_n]!=input_config[image_n])){
				printf("\nError: incorrect layer configuration for layer-%d !!!\n", ll+1);
				//return 1;
			}

		}
		else{ // other layers

			// Currently weight_n must be divisible by VEC_SIZE (for first layer, padding is performed when weight_n is not divisible by VEC_SIZE)
			if((layer_config[ll][weight_n]%VEC_SIZE)!=0){
				printf("\nError: incorrect setting of parameter VEC_SIZE !!!\n");
				return 1;
			}
			if((ll!=1)&&(ll!=20) && (ll!=21) && (layer_config_original[ll][data_n]!=layer_config_original[ll-1][conv_z])){
				printf("\nError: incorrect setting of convolution input/output size for layer-%d!!!\n", ll+1);
				return 1;
			}
		}
		if((layer_config_original[ll][conv_x]!=(layer_config_original[ll][data_w]-layer_config_original[ll][weight_w]+2*layer_config_original[ll][conv_padding])/layer_config_original[ll][conv_stride]+1)
			|| (layer_config_original[ll][conv_y]!=(layer_config_original[ll][data_h]-layer_config_original[ll][weight_h]+2*layer_config_original[ll][conv_padding])/layer_config_original[ll][conv_stride]+1)
		    || (layer_config_original[ll][conv_z]!=layer_config_original[ll][weight_m])){
			printf("\nError: incorrect setting of convolution output size or filter params for layer-%d!!!\n", ll+1);
			return 1;
		}
		if(layer_config_original[ll][pool_on] && ((layer_config_original[ll][pool_x]!=(layer_config_original[ll][conv_x]-layer_config_original[ll][pool_size])/layer_config_original[ll][pool_stride]+1)
			|| (layer_config_original[ll][pool_y]!=(layer_config_original[ll][conv_y]-layer_config_original[ll][pool_size])/layer_config_original[ll][pool_stride]+1)
		    || (layer_config_original[ll][pool_z]!=layer_config_original[ll][conv_z]))){
			printf("\nError: incorrect setting of pooling input/output size for layer-%d!!!\n", ll+1);
			return 1;
		}

		if(layer_config[ll][conv_x]==1){ // when only one group for FC layer
			conv_win_size_dim1  = layer_config[ll][weight_w];
		}
		else{
			conv_win_size_dim1  = layer_config[ll][weight_w]+(CONV_GP_SIZE_X-1)*layer_config[ll][conv_stride];
		}
		conv_win_size_dim2    = layer_config[ll][weight_h];
		// check win_buffer size
		if(conv_win_size_dim1*conv_win_size_dim2*layer_config[ll][weight_n]/(VEC_SIZE*W_VEC_SIZE) > WIN_BUF_SIZE){

			printf("Error: required win_buffer size is %d, configured size is %d \n", conv_win_size_dim1*conv_win_size_dim2*layer_config[ll][weight_n]/(VEC_SIZE*W_VEC_SIZE), WIN_BUF_SIZE);
			return 1;
		}
		// check weight_buffer size
		if(layer_config[ll][weight_w]*2*layer_config[ll][weight_h]*layer_config[ll][weight_n]/(VEC_SIZE*W_VEC_SIZE) > WEIGHT_BUF_SIZE){

			printf("Error: required weight_buffer size is %d, configured size is %d \n", layer_config[ll][weight_w]*2*layer_config[ll][weight_h]*layer_config[ll][weight_n]/(VEC_SIZE*W_VEC_SIZE), WEIGHT_BUF_SIZE);
			return 1;
		}

	}

	// image and weight files
	weights      = (DTYPE *)alignedMalloc(sizeof(DTYPE)*WEIGHTS_FILE_SIZE, DMA_ALIGNMENT);
	image        = (DTYPE *)alignedMalloc(sizeof(DTYPE)*IMAGE_FILE_SIZE, DMA_ALIGNMENT);
	memset(image,0,sizeof(DTYPE)*IMAGE_FILE_SIZE);

	// input data buffers
	// padding the input RGB image with extra number of zeros channels, so that data_n/weight_n is divisible by VEC_SIZE
	layer_config[0][weight_n] = ceil((float)layer_config[0][weight_n]/VEC_SIZE)*VEC_SIZE;
	layer_config[0][data_n] = layer_config[0][weight_n];

	//data_init   = (DTYPE *)alignedMalloc(sizeof(DTYPE)*layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n], DMA_ALIGNMENT);
	//memset(data_init, 0, sizeof(DTYPE)*layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]);// fill non-RGB dims with 0
	data_init   = (DTYPE *)alignedMalloc(sizeof(DTYPE)*input_config[image_w]*input_config[image_h]*layer_config[0][data_n], DMA_ALIGNMENT);
	memset(data_init, 0, sizeof(DTYPE)*input_config[image_w]*input_config[image_h]*layer_config[0][data_n]);// fill non-RGB dims with 0

	// final results
	if(LAYER_NUM>=CONV_NUM)// For last conv and all fc layers, all batch results are read back
		output_size = output_config[output_w]*output_config[output_h]*output_config[output_n]*input_config[batch_size];
	else // For other conv layers, only one item of
		output_size = output_config[output_w]*output_config[output_h]*output_config[output_n];

	godref_size = output_config[output_w]*output_config[output_h]*output_config[output_n];

	output          = (DTYPE *)alignedMalloc(sizeof(DTYPE)*output_size, DMA_ALIGNMENT); // vectorized results
	output_one_item = (DTYPE *)alignedMalloc(sizeof(DTYPE)*godref_size, DMA_ALIGNMENT); // one item extracted from batch results
	golden_ref      = (DTYPE *)alignedMalloc(sizeof(DTYPE)*godref_size, DMA_ALIGNMENT);
    output_reorder  = (DTYPE *)alignedMalloc(sizeof(DTYPE)*godref_size, DMA_ALIGNMENT); // reordered results for verifying

	if(weights == NULL || image == NULL || golden_ref == NULL || data_init == NULL || output == NULL || output_one_item == NULL || output_reorder == NULL)
	{
		printf("Not enough memory !!!");
		alignedFree(weights);
		alignedFree(image);
		alignedFree(data_init);
		alignedFree(golden_ref);
		alignedFree(output_one_item);
		alignedFree(output);
		alignedFree(output_reorder);

		return 1;
	}

	// weights and bias	buffers
	for(int j=0; j<LAYER_NUM; j++){

		//weight_size = (layer_config[j][weight_w]*2*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m]);
		weight_size = (WEIGHT_W_VEC_SIZE*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m]);
		weight_conv[j] = (DTYPE *)alignedMalloc(sizeof(DTYPE)*weight_size, DMA_ALIGNMENT);
		bias_conv[j]   = (DTYPE *)alignedMalloc(sizeof(DTYPE)*layer_config[j][bias_size], DMA_ALIGNMENT);

		memset(weight_conv[j], 0, sizeof(DTYPE)*weight_size);             // reset all value (include padding value) to zero
		memset(bias_conv[j], 0, sizeof(DTYPE)*layer_config[j][bias_size]);// reset all value (include padding value) to zero

		if(weight_conv[j] == NULL || bias_conv[j] == NULL )
		{
			printf("Not enough memory !!!");
			for(int i=0; i<=j; i++){
				alignedFree(weight_conv[i]);
				alignedFree(bias_conv[i]);
			}
			return 1;
		}
	}

    // Weights
    bin_file_r.open(weight_file_path, ios::in | ios::binary);

    if(bin_file_r.is_open())
    {
		//Get file size
		bin_file_r.seekg(0, bin_file_r.end);
		file_size = bin_file_r.tellg();
		bin_file_r.seekg(0, bin_file_r.beg);

    	bin_file_r.read((char *)weights, sizeof(DTYPE)*WEIGHTS_FILE_SIZE);
    	printf("\n%d total weights read \n", file_size/((int)sizeof(DTYPE)));
		if(WEIGHTS_FILE_SIZE!=(file_size/(sizeof(DTYPE))))
			printf("Warning: weight file size does not match user configuration !!!\n");
    	bin_file_r.close();
    }
    else
    	printf("Weights file does not exits !!!\n");

	// Synset_words
     //int nn=0;
	 //FILE *fp=fopen(synset_word_file_path,"r");
	 //if(!fp)
	 //{
	 //    printf("Synset word file does not exits !!!\n");
	 //    return 1;
	 //}
	 //while (!feof(fp)){
     //   fgets(synset_buf[nn], 1024, fp);
     //   nn++;
     //}
     //fclose(fp);

#ifdef USE_OPENCV
	// label
	nn=0;
	fp=fopen(LabelPath,"r");
	if(!fp)
	{
		 printf("Label file does not exits !!!\n");
		 return 1;
	 }
	while (!feof(fp)&&nn<PICTURE_NUM){
		 //printf("read%d......\n",nn);
        fgets(label_buf[nn], 1024, fp);
        nn++;
    }
	fclose(fp);
	labelNum();
#endif
    // golden_output
	bin_file_r.open(ref_file_path, ios::in | ios::binary);

    if(bin_file_r.is_open())
    {
		//Get file size
		bin_file_r.seekg(0, bin_file_r.end);
		file_size = bin_file_r.tellg();
		bin_file_r.seekg(0, bin_file_r.beg);

    	bin_file_r.read((char *)golden_ref, sizeof(DTYPE)*godref_size);
    	printf("%d total output reference read \n\n", file_size/((int)sizeof(DTYPE)));
		if(godref_size!=(file_size/(sizeof(DTYPE))))
			printf("Warning: golden reference file size does not match !!!\n");
    	bin_file_r.close();
    }
    else
    	printf("Golden file does not exits !!!\n");

	// Layer-1
	reorderWeights(weights, weight_conv[0], layer_config[0][weight_w], layer_config[0][weight_h], layer_config[0][weight_n], layer_config[0][weight_m], layer_config_original[0][weight_n], layer_config_original[0][weight_m], ptr, padding_offset[0], VEC_SIZE,WEIGHT_W_VEC_SIZE, LANE_NUM);
	//ptr+=layer_config[0][weight_w]*2*layer_config[0][weight_h]*layer_config_original[0][weight_n]*layer_config_original[0][weight_m];
	//ptr+=8*layer_config[0][weight_h]*layer_config[0][weight_n]*layer_config_original[0][weight_m];
	ptr+=WEIGHT_W_VEC_SIZE*layer_config[0][weight_h]*layer_config[0][weight_n]*layer_config[0][weight_m];

    //printf("before of reorderBias: prt=%d,weight_h=%d,weight_n=%d,weight_m=%d\n",ptr,layer_config[0][weight_h],layer_config[0][weight_n],layer_config_original[0][weight_m]);

	reorderBias(weights, bias_conv[0], ptr, padding_offset[0], layer_config[0][bias_size], layer_config_original[0][bias_size], LANE_NUM);
	//ptr+=layer_config_original[0][bias_size];
	ptr+=layer_config[0][bias_size];

	// Other layers
	for(unsigned j=1; j<LAYER_NUM; j++){

		//if(ptr+layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config_original[j][weight_n]*layer_config_original[j][weight_m]>WEIGHTS_FILE_SIZE)
        
		//printf("j=%d,ptr=%d,total ptr=%d weight_h=%d,weight_n=%d,weight_m=%d \n",j,ptr,ptr+8*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m],layer_config[j][weight_h],layer_config[j][weight_n],layer_config[j][weight_m]);
		if(ptr+WEIGHT_W_VEC_SIZE*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m]>WEIGHTS_FILE_SIZE)
		{
			printf("Error：exceed weight file size !!!total ptr=%d \n",ptr+WEIGHT_W_VEC_SIZE*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m]);
			return 1;
		}

		reorderWeights(weights, weight_conv[j], layer_config[j][weight_w], layer_config[j][weight_h], layer_config[j][weight_n], layer_config[j][weight_m], layer_config_original[j][weight_n], layer_config_original[j][weight_m], ptr, padding_offset[j], VEC_SIZE,WEIGHT_W_VEC_SIZE, LANE_NUM);
		//ptr+=layer_config[j][weight_w]*2*layer_config[j][weight_h]*layer_config_original[j][weight_n]*layer_config_original[j][weight_m];
		//ptr+=8*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config_original[j][weight_m];
		ptr+=WEIGHT_W_VEC_SIZE*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m];
		//printf("j=%d,ptr+lj=%d \n",j,ptr);
		reorderBias(weights, bias_conv[j], ptr, padding_offset[j], layer_config[j][bias_size], layer_config_original[j][bias_size], LANE_NUM);
		//ptr+=layer_config_original[j][bias_size];
		ptr+=layer_config[j][bias_size];
		//printf("j=%d,bias_size=%d, ptr+bias_size=%d \n",j,layer_config[j][bias_size],ptr);
	}

	return 0;
}

void reorderWeights(DTYPE *weights, DTYPE *weight_buf, unsigned dim1, unsigned dim2, unsigned dim3, unsigned dim4, unsigned dim3_original, unsigned dim4_original, unsigned offset, unsigned padding_offset, unsigned vecSize,unsigned w_vecSize, unsigned laneNum){

	DTYPE    *copy_with_padding;
    //FILE *fp_weight;
    //fp_weight=fopen("reorderweight.log", "a+");

    //fprintf(fp_weight,"reorderweight para:dim1=%d,dim2=%d,dim3=%d,dim4=%d,dim3_original=%d,dim4_original=%d,offset=%d,padding_offset=%d,vecSize=%d,laneNum=%d,w_vecSize=%d\n",dim1,dim2,dim3,dim4,dim3_original,dim4_original,offset,padding_offset,vecSize,laneNum,w_vecSize);


	// First, copy the data into new buffer and padding in dim3/dim4 with zeros if needed
	copy_with_padding  = (DTYPE *)malloc(sizeof(DTYPE)*dim1*dim2*dim3*dim4);
	if(copy_with_padding == NULL)
	{
		printf("Error: not enough memory when padding weight!!!");
		free(copy_with_padding);
	}
	memset(copy_with_padding, 0, sizeof(DTYPE)*dim1*dim2*dim3*dim4);

/*
	for(unsigned m = 0; m<dim4_original; m++){
		for(unsigned n = 0; n<dim3_original; n++){
			for(unsigned i = 0; i<dim2; i++){
				for(unsigned j = 0; j<dim1; j++){
							copy_with_padding[(padding_offset*dim1*dim2*dim3) + m*dim1*dim2*dim3 + n*dim1*dim2 + i*dim1 + j]
													= (DTYPE) weights[offset+m*dim1*dim2*dim3_original + n*dim1*dim2 + i*dim1 + j];
				}
			}
		}
	}
*/
	// Second, perform vectorization in dim3 by VEC_SIZE and at the same time, perform vectorization in dim4 by a factor of LANE_NUM
	for(unsigned m = 0; m < (dim4/laneNum); m++){
		for(unsigned n = 0; n < (w_vecSize*dim2*dim3/(vecSize*w_vecSize)); n++){
			//for(unsigned i = 0; i < dim2; i++){
			//	for(unsigned j = 0; j < dim1; j++){
					for(unsigned ll = 0; ll < laneNum; ll++){
						for(unsigned k = 0; k < vecSize; k++){
						    for(unsigned w = 0; w < w_vecSize; w++){
							    weight_buf[m*WEIGHT_W_VEC_SIZE*dim2*dim3*laneNum + n*vecSize*w_vecSize*laneNum + ll*vecSize*w_vecSize + k*w_vecSize + w]
										= (DTYPE) weights[offset+(m*laneNum+ll)*dim3*dim2*WEIGHT_W_VEC_SIZE + n*vecSize*w_vecSize + k*w_vecSize + w];
                                //fprintf(fp_weight,"reorderWeights: m=%d,n=%3d,ll=%2d,k=%d,w=%d, weight_buf[%d] = %5d ,weights[%d]=%d\n",m,n,ll,k,w,(m*8*dim2*dim3*laneNum + n*vecSize*w_vecSize*laneNum + ll*vecSize*w_vecSize + k*w_vecSize + w),weight_buf[m*8*dim2*dim3*laneNum + n*vecSize*w_vecSize*laneNum + ll*vecSize*w_vecSize + k*w_vecSize + w],((m*laneNum+ll)*dim3*dim2*8 + n*vecSize*w_vecSize + k*w_vecSize + w),weights[offset+(m*laneNum+ll)*dim3*dim2*8 + n*vecSize*w_vecSize + k*w_vecSize + w]);

                            }
						}
					}
			//	}
			//}
		}
	}

	// release resource
	free(copy_with_padding);
}

void reorderBias(DTYPE *dataIn, DTYPE *bias, unsigned offset, unsigned padding_offset, unsigned dim4, unsigned dim4_original, unsigned laneNum){

	DTYPE *copy_with_padding;

	// first copy the data into new buffer with zero paddings
	copy_with_padding  = (DTYPE *)malloc(sizeof(DTYPE)*dim4);
	if(copy_with_padding == NULL)
	{
		printf("Not enough memory when reordering bias!!!");
		free(copy_with_padding);
	}
	memset(copy_with_padding, 0, sizeof(DTYPE)*dim4);
	// padding evenly on two sides of weight_m
	memcpy(copy_with_padding+padding_offset, dataIn+offset, sizeof(DTYPE)*dim4_original);
	// second, perform vectorization by factor of LANE_NUM
	for(unsigned m = 0; m<(dim4/laneNum); m++){
		for(unsigned ll = 0; ll<laneNum; ll++){
			//bias[m*laneNum + ll] = (DTYPE) copy_with_padding[m*laneNum + ll];
			bias[m*laneNum + ll] = (DTYPE) dataIn[offset + m*laneNum + ll];
            //printf("bias[%d]=%d\n",m*laneNum + ll,bias[m*laneNum + ll]);
		}
	}

	// release resource
	free(copy_with_padding);
}


// Extract one item from batch results
void extractOutput(DTYPE *output, DTYPE *output_one_item, unsigned item_num, unsigned batch_size, unsigned dim1, unsigned dim2, unsigned dim3){

	unsigned char mask = 0xff;
	unsigned char batch_size_in_dim;
	unsigned char batch_size_in_dim_log;
	unsigned char batch_indx_dim1;
	unsigned char batch_indx_dim2;

	if(batch_size==1){
		batch_size_in_dim = 1;
		batch_indx_dim1 = 0;
		batch_indx_dim2 = 0;
	}
	else{
		batch_size_in_dim = log(batch_size)/log(2);
		batch_size_in_dim_log = log(batch_size_in_dim)/log(2);
		batch_indx_dim1 = item_num&(~((mask>>batch_size_in_dim_log)<<batch_size_in_dim_log));
		batch_indx_dim2 = item_num>>batch_size_in_dim_log;
		printf("Batch Size=%d, verifying NO.%d batch item (indx= %d, %d) ...\n", batch_size, item_num, batch_indx_dim1, batch_indx_dim2);
	}


	for(unsigned k = 0; k<(dim3/VEC_SIZE); k++){
		for(unsigned i = 0; i<dim2; i++){
			for(unsigned j = 0; j<dim1; j++){
				for(unsigned vv = 0; vv<VEC_SIZE; vv++){
					output_one_item[k*dim2*dim1*VEC_SIZE + i*dim1*VEC_SIZE + j*VEC_SIZE + vv]
						= output[k*dim2*dim1*batch_size_in_dim*batch_size_in_dim*VEC_SIZE + (i+batch_indx_dim2*dim2)*batch_size_in_dim*dim1*VEC_SIZE + (j+batch_indx_dim1*dim1)*VEC_SIZE + vv];
				}
			}
		}
	}
}


// Re-ordering the vectorized output into scalar form
void reorderOutput(DTYPE *output, DTYPE *output_reorder, unsigned dim1, unsigned dim2, unsigned dim3){

	for(unsigned i = 0; i<dim2; i++){
		for(unsigned j = 0; j<dim1; j++){
			for(unsigned k = 0; k<(dim3/VEC_SIZE); k++){
				for(unsigned vv = 0; vv<VEC_SIZE; vv++){
					output_reorder[(k*VEC_SIZE+vv)*dim2*dim1 + i*dim1 + j]
						= output[k*dim2*dim1*VEC_SIZE + i*dim1*VEC_SIZE + j*VEC_SIZE + vv];
                    //if (i==0 && j==0 && k==0 && vv==1) 
                    //    printf("pointer_output=%d pointer_output_reorder=%d output=%d output_reorder=%d ",k*dim2*dim1*VEC_SIZE + i*dim1*VEC_SIZE + j*VEC_SIZE + vv, (k*VEC_SIZE+vv)*dim2*dim1 + i*dim1 + j,output[k*dim2*dim1*VEC_SIZE + i*dim1*VEC_SIZE + j*VEC_SIZE + vv], output_reorder[(k*VEC_SIZE+vv)*dim2*dim1 + i*dim1 + j]); 
				}
			}
		}
	}
}

void softmax(DTYPE *output_reorder , DTYPE *output)
{
    unsigned int i;
	float data_max=0.0;
	float data_exp;
	float sum_exp=0.0;
	for(i=0;i<output_config[output_n];i++)
	{
	  if(data_max<output_reorder[i])
	    data_max=output_reorder[i];
	}
    for(i=0;i<output_config[output_n];i++)
	{
	   data_exp=exp((float)output_reorder[i]-data_max);
	   sum_exp += data_exp;
	 }
    for(i=0;i<output_config[output_n];i++)
	{
	   data_exp=exp((float)output_reorder[i]-data_max);
	   output[i]=data_exp / sum_exp*100.0;

	 }
}

int getProb(DTYPE *output)
{
    int m=0;
    float max=output[0];

	// find the class with the highest score
    for(unsigned int i=0;i<output_config[output_n];i++){
        if(max<output[i]){
			max=output[i];
            m=i;
		}
    }

	// replace the last two ASCII charactor with space
	//int ii=strlen(synset_buf[m]);
    //synset_buf[m][ii-2]= 32;
    //synset_buf[m][ii-1]= 32;

	//printf("\nThe inference result is %s (the prob is %5.2f) \n\n", synset_buf[m], max);

	return m;

}

void dumpResult(){

	ofstream result_file;

	result_file.open(dump_file_path, ios::out);

	for(unsigned i=0; i<output_config[output_n]; i++){
		result_file << "z=" << i << endl;
		for(unsigned j=0; j<output_config[output_h]; j++){
			result_file << "x=" << j << ": ";
			for(unsigned k=0; k<output_config[output_w]; k++){
					result_file << (float)output_reorder[output_config[output_w]*output_config[output_h]*i + output_config[output_w]*j + k] << "(";
					result_file << (float)golden_ref[output_config[output_w]*output_config[output_h]*i + output_config[output_w]*j + k] << ") ";
			}
			result_file << endl;
		}
		result_file << endl;
	}
	result_file.close();
}

#ifdef USE_OPENCV
// Load image from files
int load_picture(DTYPE *image){

		float *mean_data;

		printf("\nLoading picture %s .....\n\n", picture_file_path);

		// load ILSVRC2012 database mean data
		mean_data = (float *) malloc(sizeof(float)*MEAN_DATA_WIDTH*MEAN_DATA_HEIGHT*MEAN_DATA_CHANNEl);
		if(mean_data == NULL){
			printf("Error: allocating memory for images failed !!!\n");
			return 1;
		}

		FILE *p_mean_data=fopen(mean_data_file_path,"rb");
		fread(mean_data,sizeof(float),MEAN_DATA_WIDTH*MEAN_DATA_HEIGHT*MEAN_DATA_CHANNEl,p_mean_data);

		// load picture from files
		Mat img = imread(picture_file_path);
		// resize pic to MEAN_DATA_WIDTH*MEAN_DATA_HEIGHT, and substract with mean data
		Mat img1;
		resize(img,img1,Size(MEAN_DATA_WIDTH,MEAN_DATA_HEIGHT));
		img1.convertTo(img1,CV_32FC3);
		Mat mean_mat(MEAN_DATA_WIDTH, MEAN_DATA_HEIGHT, CV_32FC3, mean_data);
		img1 = img1 - mean_mat;
		// resize to the input size of the first layer
		Mat img2;
		resize(img1,img2,Size(layer_config_original[0][data_w],layer_config_original[0][data_h]));
		// convert to 8-bit fixed-point
		img2.convertTo(img2,CV_8SC3);
		// reorder channel sequence from RGB to GBR
		DTYPE * data_ptr = (DTYPE*)img2.data;
		unsigned int w,h,c;
		unsigned int k=0;
		for(h=0;h<layer_config_original[0][data_h];h++){
			for(w=0;w<layer_config_original[0][data_w];w++){
				for (c=0;c<layer_config_original[0][data_n];c++){
					 image[c*layer_config_original[0][data_w]*layer_config_original[0][data_h]+h*layer_config_original[0][data_w]+w]=data_ptr[k];
					 k++;
				}
			}
		}
		fclose(p_mean_data);
		free(mean_data);
		return 0;
}

void labelNum()
{
	int i,num;
	char *p;
	for(i=0;i<PICTURE_NUM;i++)
	{
		num=0;
		p=label_buf[i];
		while(*p!='\0')
		{
			if(*p==' ')
			{
				p++;
				break;
			}
			p++;
		}
		while(*p!='\n')
		{
			num=num*10+(*p-'0');
			p++;
		}
		label[i]=num;
	}
}

void numtochar(int num,char *end)
{
	int counter=0;
	while(num>0)
	{
		end[7-counter]='0'+num%10;
		num/=10;
		counter++;
	}

}

void strcopy(DTYPE *str1,DTYPE *str2)
{
	for(unsigned int i=0;i<output_config[output_n];i++)
	{
		str1[i]=str2[i];
	}
}
//Top-k Accuracy
void getAccuracy(DTYPE *output_reorder,int num)
{
	strcopy(searchTop,output_reorder);
	int predictionTop5[5],max;
	float data_max;
	unsigned int i,k;
	float tmp_accuracy1,tmp_accuracy5;
	num=num-1;//label[] begin from 0
	for(k=0;k<5;k++)
	{
		data_max=searchTop[0];
		max=0;
		for(i=0;i<output_config[output_n];i++)
		{
		  if(data_max<searchTop[i])
		  {
			data_max=searchTop[i];
			max=i;
		  }
		}
		predictionTop5[k]=max;
		searchTop[max]=-1;//min
	}
	//top-k
	for(i=0;i<5;i++)
	{
		if(predictionTop5[i]==label[num])
		{
			accuracy5++;
			if(i==0)
				accuracy1++;
			break;
		}
	}
	tmp_accuracy1=accuracy1/(num+1);
	tmp_accuracy5=accuracy5/(num+1);

	printf("Current Top-1 accuracy = %5.3f\n", tmp_accuracy1);
	printf("Current Top-5 accuracy = %5.3f\n", tmp_accuracy5);

}
#endif

// Release all memory resources here
void cleanup()
{
    printf("\ncleanup started!!!\n");

	// Release the opencl runtime resource allocated
	for(unsigned i = 0; i < num_devices; ++i) {
        printf("\nknl_memRd cleanup started!!!\n");
		if(knl_memRd && knl_memRd[i]) {
			clReleaseKernel(knl_memRd[i]);
		}
        //printf("\nknl_conv cleanup started!!!\n");
		//if(knl_conv && knl_conv[i]) {
		//	clReleaseKernel(knl_conv[i]);
		//}
        printf("\nknl_memWr cleanup started!!!\n");
		if(knl_memWr && knl_memWr[i]) {
			clReleaseKernel(knl_memWr[i]);
		}
        //printf("\nknl_memWrReorg cleanup started!!!\n");
		//if(knl_memWrReorg && knl_memWrReorg[i]) {
		//	clReleaseKernel(knl_memWrReorg[i]);
		//}
        //printf("\nknl_pool cleanup started!!!\n");
		//if(knl_pool && knl_pool[i]) {
		//	clReleaseKernel(knl_pool[i]);
		//}
// 		if(knl_lrn && knl_lrn[i]) {
//			clReleaseKernel(knl_lrn[i]);
//		}
		if(que_memRd && que_memRd[i]) {
			clReleaseCommandQueue(que_memRd[i]);
		}
		//if(que_conv && que_conv[i]) {
		//	clReleaseCommandQueue(que_conv[i]);
		//}
		if(que_memWr && que_memWr[i]) {
			clReleaseCommandQueue(que_memWr[i]);
		}
		//if(que_memWrReorg && que_memWrReorg[i]) {
		//	clReleaseCommandQueue(que_memWrReorg[i]);
		//}
		//if(que_pool && que_pool[i]) {
		//	clReleaseCommandQueue(que_pool[i]);
		//}
		if(data_buf && data_buf[i]) {
			clReleaseMemObject(data_buf[i]);
		}
		if(output_buf && output_buf[i]) {
			clReleaseMemObject(output_buf[i]);
		}
		if(weights_buf && weights_buf[i]) {
			clReleaseMemObject(weights_buf[i]);
		}
		if(bias_buf && bias_buf[i]) {
			clReleaseMemObject(bias_buf[i]);
		}
		if(concat_buf && concat_buf[i]) {
			clReleaseMemObject(concat_buf[i]);
		}
		if(conv_res_buf && conv_res_buf[i]) {
			clReleaseMemObject(conv_res_buf[i]);
		}
	}

	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}

	alignedFree(weights);
	alignedFree(image);
	alignedFree(data_init);
	for(int j=0; j<LAYER_NUM; j++){
		alignedFree(weight_conv[j]);
		alignedFree(bias_conv[j]);
	}
	alignedFree(golden_ref);
	alignedFree(output);
	alignedFree(output_reorder);
	alignedFree(output_one_item);

}
