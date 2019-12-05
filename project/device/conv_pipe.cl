/*
 * ------------------------------------------------------
 *
 *   PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs
 *
 * ------------------------------------------------------
 * Filename:
 *   - conv_pipe.cl
 *
 * Author(s):
 *   - Dong Wang, wangdong@m.bjtu.edu.cn
 *
 * History:
 *   - v1.3 Win-Buffer-Based Implementation
 * ------------------------------------
 *
 *   Copyright (C) 2016, Institute of Information Science,
 *   Beijing Jiaotong University. All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 */
/*
 * --------------------------------------------------------------------------
 *
 *   PipeCNN_Winograd: Pipecnn accelerated by Winograd
 *
 * --------------------------------------------------------------------------
 *
 * Author(s):
 *    -Anrong Yang, yangar@lenovo.com
 *
 * --------------------------------------------------------------------------
 *
 * Copyright (C) 2019 Lenovo
 * Licensed under BSD-3, see COPYING.BSD file for details
 *
 * --------------------------------------------------------------------------
 *
 * On the basis of PipeCNN, we use Winograd algorithm to implement CNN and 
 * change the original four kernels of PipeCNN to two.
 *
 * --------------------------------------------------------------------------
 *
 *
 */


//#pragma OPENCL EXTENSION cl_intel_channels : enable
//#define USE_ROM 

// The following macros are used for debug
//#define DEBUG_MEMRD
//#define DEBUG_CONV
//#define DEBUG_POOL
//#define DEBUG_MEMWR
//#define DEBUG_LRN
//#define DEBUG_LRN_OUT


#include "hw_param.cl"
#include "type_def.cl"
#include "rtl_lib.h"


// parallel MAC units including (VEC_SIZE-1) multipliers
//MACTYPE mac(lane_data input, lane_data weights)
//{
//	MACTYPE output = MASK_MULT & CZERO;
//	
//	#pragma unroll
//	for(int i=0; i<VEC_SIZE/4; i++){
//		output += MASK_MULT & mult_add_fix8bx4(input.data[i*4], weights.data[i*4], input.data[i*4+1], weights.data[i*4+1], input.data[i*4+2], weights.data[i*4+2], input.data[i*4+3], weights.data[i*4+3]);
//	}
//	return output;
//}


DPTYPE pool_max(DPTYPE a_in, DPTYPE b_in)
{
	DPTYPE max_value;
	
	if(a_in >= b_in)
		max_value = a_in;
	else
		max_value = b_in;
	
	return max_value;

}

// Fetch Data from Global Memory
__kernel
__attribute__((max_global_work_dim(0)))
void memRead(
			// Params Ports
            uchar  layer_num,
            uchar  conv_row_rem,
			ushort  data_dim1,
			ushort  data_dim2,
			uint   data_dim1xdim2,
			uchar  weight_dim1,
			uchar  weight_dim2,
			ushort weight_dim3,
			ushort weight_dim4_div_lane, // avoid generating divider
			uchar  weight_dim1x2,
			uint   weight_dim1x2x3,
			ushort conv_x,
			uchar  group_size_x,
			uchar  stride,
			uchar  padding,
			uchar  split,
			uchar  group_num_x,
			uint   group_num_y,
			uchar  group_rem_size_x,
			uint   group_num_mul_win_size,
			//uchar  group_rem_size_y, // not used in this version
			uint   group_rem_size_xyz,
            ushort win_size,
			uchar  win_size_x,
			uchar  win_size_y,
			uint   win_size_xyz,  
			// Data Ports
			//__global lane_data        *restrict bottom,
			//__global data_wng_ddr        *restrict bottom,
			__global intvec        *restrict bottom,
			__global channel_vec_wng  *restrict weights,
			__global channel_scal     *restrict bias,

            //ushort  conv_row_num,
		    uint  conv_out_loopnum,
			uint  conv_loop_cnt,
			uchar  control, //[0]-> relu  [1]->bypass pooling
			uchar  fc_en,
			char  frac_w,
            char  frac_b,
			char  frac_din,
			char  frac_dout,
			ushort line_size,  // line_size should be no larger than POOL_LBUF_DEPTH
			ushort col_size,  // line_size should be no larger than POOL_LBUF_DEPTH
			uchar pool_size,  // by now, only pooling size no larger than 3
			uchar pool_stride
            )

{


	// Input Data, Weights and Bias
	//lane_data         data_vec[W_VEC_SIZE];
	//data_wng          data_vec;
	data_wng_ddr        data_vec;
	//data_wng_ddr      __attribute__((register))  data_vec_wr1,data_vec_wr2;
	//data_wng_ddr      __attribute__((register))  data_vec_rd1,data_vec_rd2;
	data_wng_ddr      __attribute__((register))  data_vec_rd1_tmp,data_vec_rd2_tmp;
	intvec      __attribute__((register))  data_vec_wr1,data_vec_wr2;
	intvec      __attribute__((register))  data_vec_rd1,data_vec_rd2;
	data_wng_vec8       data_vec8;
	lane_data           data_vec_zero;
	//data_wng_ddr      data_vec_zero4;
	intvec                data_vec_zero4 = (intvec)(0);
	data_wng_vec8       data_vec_zero8;
    data_wng            buf_data;
	channel_scal        bias_ch_in;
	//ushort            data_offset = 0; // assuming the 1st layer is not in split
	
	// virtual loop counters
	ushort gp_num_x, gp_num_y, out_idx_z;
	ushort gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf;
	uchar  win_itm_x=0;
    uchar  win_itm_y=0;
	ushort win_itm_z=0;
	
	//uchar  gp_item_idx_x;

	
    ushort feature_idx_dim1=0; 
    ushort feature_idx_dim2=0;
	ushort feature_idx_dim3=0;

	//uint   item_loop_bound;
	
	uchar  flag = 0; // ping-pong flag
	//uchar  flaga1 = 2; 
	uchar  flaga1 = 1; 
    ushort flag_cnt = 0;
    uchar  read8_flag = 0;
    //uchar  read8_flag = 1;
	//bool  winbuff_en=0; // ping-pong flag
	
	// Ping-pong buffer
	//__local lane_datai    win_buffer[2][WIN_BUF_SIZE]; // working sequence 0->1->0->1 ...
	//__local data_wng_ddr    win_buffer[WIN_BUF_SIZE][4] __attribute__((bank_bits(10,9),numbanks(4),bankwidth(32))); // working sequence 0->1->0->1 ...
	//__local data_wng_ddr    win_buffer[WIN_BUF_SIZE][3] __attribute__((numbanks(2),bankwidth(32))); // working sequence 0->1->0->1 ...

    #if VEC_SIZE==8
	    __local intvec    win_buffer[WIN_BUF_SIZE][3] __attribute__((numbanks(4),bankwidth(32))); // working sequence 0->1->0->1 ...
    #else
	    __local intvec    win_buffer[WIN_BUF_SIZE][3] __attribute__((numbanks(4),bankwidth(64))); // working sequence 0->1->0->1 ...
    #endif

	//__local data_wng_ddr    win_buffer[WIN_BUF_SIZE][4] __attribute__((memory,numbanks(1),bankwidth(128),singlepump,numreadports(2),numwriteports(1))); // working sequence 0->1->0->1 ...
	//__local data_wng_ddr    win_buffer[WIN_BUF_SIZE][3] __attribute__((memory,numbanks(1),bankwidth(64),doublepump,numreadports(1),numwriteports(1))); // working sequence 0->1->0->1 ...
	//__local data_wng_ddr    win_buffer[WIN_BUF_SIZE][4] __attribute__((memory,numbanks(4),bankwidth(32),numreadports(2),numwriteports(1))); // working sequence 0->1->0->1 ...
	//__local data_wng_ddr    win_buffer[WIN_BUF_SIZE][4] __attribute__((numbanks(4))); // working sequence 0->1->0->1 ...
	//__local data_wng_ddr    win_buffer[4][WIN_BUF_SIZE] __attribute__((numbanks(4),bankwidth(32))); // working sequence 0->1->0->1 ...
	// Weight buffer
	__local channel_vec_wng   __attribute__((numbanks(1)))  weight_buffer[WEIGHT_BUF_SIZE]; 
	//__local channel_vec_wng   __attribute__((numbanks(4),bankwidth(4096)))  weight_buffer[WEIGHT_BUF_SIZE]; 

    uint rd_prt;	

    //for conv--------------------------------------------------- 
	data_wng mac_data;
 	channel_vec_wng mac_weight;
	channel_scal bias_ch_out;
	channel_wvec_wng conv_ch_in;
	//channel_scal bias;

	channel_wvec_32_wng mac_out;
	channel_wvec_32_wng conv_out;
	channel_wvec_32_wng conv_sign_exten;
	channel_wvec_32_wng conv_with_rnd_bit;
	channel_wvec_32_wng conv_sum_bias;

	channel_wvec_wng  conv_final;
	MACTYPE leaky_relu[LANE_NUM];

    channel_vec_16_wng conv_btmuld;
    channel_vec_16_wng conv_data_mux;
    //channel_vec_32_wng conv_btdxgw;
    channel_wvec_32_wng conv_btdxgw;
    //channel_vec_32_wng conv_atxbtdxgw;
    channel_wvec_32_wng conv_atxbtdxgw;
    //channel_vec_32_wng conv_dxw_mux;
    channel_wvec_32_wng conv_dxw_mux;

    uint conv_z_cnt = 0;
    //uint conv_row_cnt = 0;


    //for pooling--------------------------------------------------- 
	channel_wvec_wng pool_final;
	pool_vec_wng __attribute__((numbanks(1))) line_buf_0[POOL_LBUF_DEPTH];
	ushort  line_buf_ptr = 0;
	ushort  col_pool_cnt = 0;
	ushort  row_pool_cnt = 0;
	ushort  row_cnt = 0;
	pool_vec_wng row_pool_reg;
	pool_vec_wng col_pool_reg;
	pool_vec_wng pool_reg;
   //--------------------------------------------------- 
   



	// Initialize the winbuf with the data in the first iteration of the group looping (as gp_num_x_winbuf=0, gp_num_y_winbuf=0)
    #pragma unroll
    for(unsigned char vv=0; vv<VEC_SIZE; vv++){
        data_vec_zero.data[vv] = CZERO;
    }
    //#pragma unroll
    //for(unsigned char ww=0; ww<4; ww++){
    //	    data_vec_zero4.wvec[ww] = data_vec_zero;
    //    //#pragma unroll
    //    //for(unsigned char vv=0; vv<VEC_SIZE; vv++){
    //	//    data_vec_zero4.wvec[ww].data[vv] = CZERO;
    //    //}
    //}
    #pragma unroll
    for(unsigned char ww=0; ww<8; ww++){
    	    data_vec_zero8.wvec[ww] = data_vec_zero;
        //#pragma unroll
        //for(unsigned char vv=0; vv<VEC_SIZE; vv++){
    	//    data_vec_zero8.wvec[ww].data[vv] = CZERO;
        //}
    }
    
/*
    //#pragma ii 2 
    for(ushort win_cnt=0  ; win_cnt<win_size; win_cnt++){
    
        // Winbuffer loading operations
        feature_idx_dim1 = win_itm_x;
        feature_idx_dim2 = win_itm_y;
        feature_idx_dim3 = win_itm_z;
        
        //#pragma unroll
        //for(uchar w_vec_cnt_init=0; w_vec_cnt_init<W_VEC_SIZE; w_vec_cnt_init++){ 
        //    if(((w_vec_cnt_init)>=padding && (w_vec_cnt_init)<data_dim1+padding) && (feature_idx_dim2>=padding && feature_idx_dim2<data_dim2+padding)){
        //    	data_vec.wvec[w_vec_cnt_init] = bottom[feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + (w_vec_cnt_init-padding)];
        //    }else{
        //    	data_vec.wvec[w_vec_cnt_init]= data_vec_zero;
        //    }
        //}


        if(feature_idx_dim2<padding || feature_idx_dim2>=data_dim2+padding){
            data_vec8 = data_vec_zero8;
        } else {
            rd_prt = (feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + feature_idx_dim1)>>2;
            //data_vec_ddr=*((__global data_wng_ddr*)& bottom[rd_pointer]);
            //data_vec = bottom[(feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + feature_idx_dim1)/4];
            data_vec8 = *((__global data_wng_vec8*)&bottom[rd_prt]);
        }

		#pragma unroll
        for (uchar i=0;i<4;i++){
            data_vec_wr1.wvec[i]   = data_vec8.wvec[i];
            //data_vec.wvec[i]   = data_vec8.wvec[i];
            data_vec_wr2.wvec[i]   = data_vec8.wvec[i+4];
        }
        win_buffer[win_cnt][0] = data_vec_wr1;
        //win_buffer[win_cnt][0] = data_vec;
        win_buffer[win_cnt][1] = data_vec_wr2;
       

        //for (uchar i=0;i<4;i++)
        //    for (uchar j=0;j<VEC_SIZE;j++)
        //        printf("data_vec.wvec[%d].data[%d]=%d\n",i,j,data_vec.wvec[i].data[j]);
        
        //win_buffer[flag][win_cnt] = data_vec;
        //win_buffer[flag_cnt][flag] = data_vec;
        //printf("read init: win_size=%d,win_cnt=%d,flag_cnt=%d,flag=%d,win_itm_y=%d,win_itm_z=%d,rd_pointer=%d\n",win_size,win_cnt,flag_cnt,flag,win_itm_y,win_itm_z,(feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + feature_idx_dim1)/4);
        //if (flag_cnt==win_size-1){
        //    flag++;
        //    flag_cnt = 0;
        //}else{ 
        //    flag_cnt++;
        //}
        
        
        // used as loop counters
        if((win_itm_z==weight_dim3/VEC_SIZE-1) && (win_itm_y==win_size_y-1))
        	win_itm_z = 0;
        else if(win_itm_y==win_size_y-1)
        	win_itm_z++;
        
        if(win_itm_y==win_size_y-1)
        	win_itm_y = 0;
        else
        	win_itm_y++;

        if (win_cnt==win_size-1){
            win_itm_x = win_itm_x+group_size_x*stride;
        }

    }
   */ 

	//if(group_num_x==1)
	//	gp_num_x_winbuf = 0; // there is only one group for FC mode when batch=1
	//else
//		gp_num_x_winbuf = 1; // loop start from the second group
//	gp_num_y_winbuf = 0;
//	out_idx_z_winbuf = 0;
	
	// reset global group virtual loop counters
	//gp_num_x = 2;
	gp_num_x = 0;
	gp_num_y = 0;
	out_idx_z = 0;

    uint out_idx_xyz=0;
    ushort win_itm_xyz=0;
    uint index=0;	
    //uint rd_prt;	
    
    //flag=2;	
    //#pragma ivdep array(weight_buffer)
    #pragma ivdep array(win_buffer)
	//Group:for(unsigned int out_idx_xyz=0; out_idx_xyz<(weight_dim4_div_lane*group_num_y*group_num_x+1); out_idx_xyz++){
    while(index!=group_num_mul_win_size) {
            index++;
	        //data_wng_ddr      __attribute__((register))  data_vec_wr1,data_vec_wr2;
	        //data_wng_ddr      __attribute__((register))  data_vec_rd1,data_vec_rd2;
	
            if (win_itm_xyz==0) {	
                //flag = out_idx_xyz & 0x01; //ping-pong flag
				
				win_itm_x = 0;
				win_itm_y = 0;
				win_itm_z = 0;
            }
				
				
	        channel_vec_wng   weight_buf_vec_wr;
	        channel_vec_wng   weight_buf_vec_rd;


			// Winbuffer loading operations
			//feature_idx_dim1 = gp_num_x_winbuf*group_size_x*stride;
			//feature_idx_dim2 = win_itm_y+gp_num_y_winbuf*CONV_GP_SIZE_Y*stride;
			feature_idx_dim1 = gp_num_x*group_size_x*stride;
			feature_idx_dim2 = win_itm_y+gp_num_y*CONV_GP_SIZE_Y*stride;
			feature_idx_dim3 = win_itm_z;

            //#pragma unroll
            //for(uchar w_vec_cnt=0; w_vec_cnt<W_VEC_SIZE; w_vec_cnt++){ 
			//    if(((feature_idx_dim1+w_vec_cnt)>=padding && (feature_idx_dim1+w_vec_cnt)<data_dim1+padding) && (feature_idx_dim2>=padding && feature_idx_dim2<data_dim2+padding)){
			//    	data_vec.wvec[w_vec_cnt] = bottom[feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + (feature_idx_dim1+w_vec_cnt-padding)];
			//   	}
			//    else{ 
			//    	data_vec.wvec[w_vec_cnt] = data_vec_zero;
			//    }
            //}
            
            rd_prt = (feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + feature_idx_dim1)/4;
            if(feature_idx_dim2<padding || feature_idx_dim2>=data_dim2+padding){
                data_vec_wr1 = data_vec_zero4;
                //data_vec = data_vec_zero4;
                //data_vec8 = data_vec_zero8;
                
            //} else if (read8_flag==1) {
            //    data_vec8 = *((__global data_wng_vec8*)&bottom[rd_prt]);
            }else{
                data_vec_wr1 = bottom[rd_prt];
                //data_vec = bottom[rd_prt];
            }
                    
/*
            if (read8_flag==1) {
		        #pragma unroll
                for (uchar i=0;i<4;i++){
                    data_vec_wr1.wvec[i]   = data_vec8.wvec[i];
                    //data_vec.wvec[i]   = data_vec8.wvec[i];
                    data_vec_wr2.wvec[i]   = data_vec8.wvec[i+4];
                }
                //win_buffer[(flag)&0x03][win_itm_xyz]   = data_vec_wr1;
                //win_buffer[(flaga1)&0x03][win_itm_xyz] = data_vec_wr2;
                //win_buffer[win_itm_xyz][flag   & 0x03] = data_vec_wr1;
                //win_buffer[win_itm_xyz][flag   & 0x03] = data_vec;
                win_buffer[win_itm_xyz][flaga1 & 0x03] = data_vec_wr2;

		        //#pragma unroll
                //for (uchar i=0;i<4;i++){
                //    win_buffer[win_itm_xyz][(flag)  &0x03].wvec[i] = data_vec8.wvec[i];
                //    win_buffer[win_itm_xyz][(flaga1)&0x03].wvec[i] = data_vec8.wvec[i+4];
                //}

            }else{                
                //win_buffer[(flag)&0x03][win_itm_xyz] = data_vec;
                //win_buffer[win_itm_xyz][flag & 0x03] = data_vec_wr1;
                //win_buffer[win_itm_xyz][flag & 0x03] = data_vec;
            }
*/
               win_buffer[win_itm_xyz][flag & 0x03] = data_vec_wr1;
               //win_buffer[win_itm_xyz][flag & 0x03] = *((data_wng_ddr*)&data_vec_wr1);

            //if (layer_num ==8)
            //    printf("read: index=%d,flag=%d,flaga1=%d,read8_flag=%d,win_itm_y=%d,win_itm_z=%d,win_itm_xyz=%d,gp_num_x=%d,gp_num_y=%d,conv_row_rem=%d,prt=%d,data=%d,feature_idx_dim2=%d,data_dim2=%d\n",index,flag,flaga1,read8_flag,win_itm_y,win_itm_z,win_itm_xyz,gp_num_x,gp_num_y,conv_row_rem,(feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + feature_idx_dim1)/4,win_buffer[win_itm_xyz][flag & 0x03].wvec[1].data[0],feature_idx_dim2,data_dim2);


			// used as loop counters
			if((win_itm_z==weight_dim3/VEC_SIZE-1) && (win_itm_y==win_size_y-1))
				win_itm_z = 0;
			else if(win_itm_y==win_size_y-1)
					win_itm_z++;

			if(win_itm_y==win_size_y-1)
				win_itm_y = 0;
			else
				win_itm_y++;
							
							
			// Load weight into weight buffer,load bias to bias buffer
			// In this version, grouping is only performed in row (x) direction
            if(gp_num_x==2 && gp_num_y==0){
                weight_buf_vec_wr = weights[out_idx_z*win_size + win_itm_xyz];
				//weight_buffer[win_itm_xyz] = weights[out_idx_z*win_size + win_itm_xyz];
				weight_buffer[win_itm_xyz] = weight_buf_vec_wr;
				bias_ch_in = bias[out_idx_z];
                //printf("conv,read weight: gp_num_x=%d,gp_num_y=%d,weight_addr=%d,out_idx_z=%d,win_size=%d,win_itm_xyz=%d\n",gp_num_x,gp_num_y,out_idx_z*win_size + win_itm_xyz,out_idx_z,win_size,win_itm_xyz);
			}
							 

			// data
			//buf_data = win_buffer[flag][win_itm_xyz];
			//write_channel_intel(data_ch, buf_data);	

            switch (flag) {   
                case 0:
                        //buf_data.wvec[0] = win_buffer[2][win_itm_xyz].wvec[0];
                        //buf_data.wvec[1] = win_buffer[2][win_itm_xyz].wvec[1];
                        //buf_data.wvec[2] = win_buffer[2][win_itm_xyz].wvec[2];
                        //buf_data.wvec[3] = win_buffer[2][win_itm_xyz].wvec[3];
                        //buf_data.wvec[4] = win_buffer[3][win_itm_xyz].wvec[0];
                        //buf_data.wvec[5] = win_buffer[3][win_itm_xyz].wvec[1];
                        //data_vec_rd1 = win_buffer[2][win_itm_xyz];
                        //data_vec_rd2 = win_buffer[3][win_itm_xyz];
                        //data_vec_wr1 = win_buffer[win_itm_xyz][2];
                        //data_vec_wr2 = win_buffer[win_itm_xyz][3];
                        //data_vec_rd1 = win_buffer[win_itm_xyz][2];
                        //data_vec_rd2 = win_buffer[win_itm_xyz][3];
                        data_vec_rd1 = win_buffer[win_itm_xyz][1];
                        data_vec_rd2 = win_buffer[win_itm_xyz][2];
                        break;
                case 1:
                        //data_vec_wr1 = win_buffer[win_itm_xyz][3];
                        //data_vec_wr2 = win_buffer[win_itm_xyz][0];
                        //data_vec_rd1 = win_buffer[win_itm_xyz][3];
                        //data_vec_rd2 = win_buffer[win_itm_xyz][0];
                        data_vec_rd1 = win_buffer[win_itm_xyz][2];
                        data_vec_rd2 = win_buffer[win_itm_xyz][0];
                        break;
                //case 2:
                default:
                        //data_vec_wr1 = win_buffer[win_itm_xyz][0];
                        //data_vec_wr2 = win_buffer[win_itm_xyz][1];
                        //data_vec_rd1 = win_buffer[win_itm_xyz][0];
                        //data_vec_rd2 = win_buffer[win_itm_xyz][1];
                        data_vec_rd1 = win_buffer[win_itm_xyz][0];
                        data_vec_rd2 = win_buffer[win_itm_xyz][1];
                        break;
                //default:
                //        //data_vec_wr1 = win_buffer[win_itm_xyz][1];
                //        //data_vec_wr2 = win_buffer[win_itm_xyz][2];
                //        data_vec_rd1 = win_buffer[win_itm_xyz][1];
                //        data_vec_rd2 = win_buffer[win_itm_xyz][2];
                //        break;
            }   
            data_vec_rd1_tmp = *((data_wng_ddr*)& data_vec_rd1); 
            data_vec_rd2_tmp = *((data_wng_ddr*)& data_vec_rd2); 
            buf_data.wvec[0] = data_vec_rd1_tmp.wvec[0];
            buf_data.wvec[1] = data_vec_rd1_tmp.wvec[1];
            buf_data.wvec[2] = data_vec_rd1_tmp.wvec[2];
            //buf_data.wvec[3] = data_vec_rd1.wvec[3];
            //buf_data.wvec[4] = data_vec_rd2.wvec[0];
            //buf_data.wvec[5] = data_vec_rd2.wvec[1];
            

            if (gp_num_x==1 && weight_dim1==3 && conv_row_rem==3) {
                buf_data.wvec[3] = data_vec_zero;
                buf_data.wvec[4] = data_vec_zero;
                buf_data.wvec[5] = data_vec_zero;
            }else if (gp_num_x==1 && weight_dim1==3 && conv_row_rem==0) {
                //buf_data.wvec[3] = data_vec_wr1.wvec[3];
                //buf_data.wvec[3] = data_vec_rd1.wvec[3];
                buf_data.wvec[3] = data_vec_rd1_tmp.wvec[3];
                buf_data.wvec[4] = data_vec_zero;
                buf_data.wvec[5] = data_vec_zero;
            }else{    
                //buf_data.wvec[3] = data_vec_wr1.wvec[3];
                //buf_data.wvec[4] = data_vec_wr2.wvec[0];
                //buf_data.wvec[5] = data_vec_wr2.wvec[1];
                buf_data.wvec[3] = data_vec_rd1_tmp.wvec[3];
                buf_data.wvec[4] = data_vec_rd2_tmp.wvec[0];
                buf_data.wvec[5] = data_vec_rd2_tmp.wvec[1];
            }

/*
            if (layer_num==13){
            for (uchar i=0;i<W_VEC_SIZE;i++)
                for (uchar j=0;j<VEC_SIZE;j++)
                    if (i<4)
                    printf("read: index=%5d,flag=%d,flaga1=%d,read8_flag=%d,win_itm_x=%d,win_itm_y=%d,win_itm_z=%2d,win_itm_xyz=%2d,gp_num_x=%d,gp_num_y=%d,conv_row_rem=%d,prt=%10d,data=%3d,feature_idx_dim1=%d,feature_idx_dim2=%d,data_dim1=%d,data_dim2=%d,buf_data.wvec[%d].data[%d]=%3d\n",index,flag,flaga1,read8_flag,win_itm_x,win_itm_y,win_itm_z,win_itm_xyz,gp_num_x,gp_num_y,conv_row_rem,(feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + feature_idx_dim1)/4,win_buffer[win_itm_xyz][flag & 0x03].wvec[i].data[j],feature_idx_dim1,feature_idx_dim2,data_dim1,data_dim2,i,j,buf_data.wvec[i].data[j]);
                    else
                    printf("read: index=%5d,flag=%d,flaga1=%d,read8_flag=%d,win_itm_x=%d,win_itm_y=%d,win_itm_z=%2d,win_itm_xyz=%2d,gp_num_x=%d,gp_num_y=%d,conv_row_rem=%d,prt=%10d,data=%3d,feature_idx_dim1=%d,feature_idx_dim2=%d,data_dim1=%d,data_dim2=%d,buf_data.wvec[%d].data[%d]=%3d\n",index,flag,flaga1,read8_flag,win_itm_x,win_itm_y,win_itm_z,win_itm_xyz,gp_num_x,gp_num_y,conv_row_rem,(feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + feature_idx_dim1)/4,win_buffer[win_itm_xyz][(flag+1)%4 & 0x03].wvec[i%4].data[j],feature_idx_dim1,feature_idx_dim2,data_dim1,data_dim2,i,j,buf_data.wvec[i].data[j]);
                    //printf("mac_data.wvec[%d].data[%d]=%d,flag=%d\n",i,j,buf_data.wvec[i].data[j]);
                    //printf("buf_data.wvec[%d].data[%d]=%d\n",i,j,buf_data.wvec[i].data[j]);
            //        printf("mac_data.wvec[1].data[0]=%d\n",buf_data.wvec[1].data[0]);

            }
*/
			// weight and bias fetcher
			weight_buf_vec_rd = weight_buffer[win_itm_xyz];
            //#pragma unroll
			//for(uchar i=0; i<LANE_NUM; i++){
            //    write_channel_intel(weight_ch[i], weight_buf_vec_rd.lane[i]);	
			//}	
	
            //printf("read: group_num_mul_win_size=%d,index=%d,win_size=%d,conv_loop_cnt=%d,flag=%d,flaga1=%d,read8_flag=%d,win_itm_y=%d,win_itm_z=%d,win_itm_xyz=%d,gp_num_x=%d,gp_num_y=%d,conv_row_rem=%d,group_num_x=%d,group_num_y=%d,conv_z_cnt=%d\n",group_num_mul_win_size,index,win_size,conv_loop_cnt,flag,flaga1,read8_flag,win_itm_y,win_itm_z,win_itm_xyz,gp_num_x,gp_num_y,conv_row_rem,group_num_x,group_num_y,conv_z_cnt);
		    // used as virtual group loop counters for winbuf loading operations
/*
		    if (win_itm_xyz==win_size-1) {
		        // used as virtual group loop counters
		        if((out_idx_z==weight_dim4_div_lane-1) && (gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
		        	out_idx_z = 0;
		        else if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
		        	out_idx_z++;	
                
		        if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
		        	gp_num_y = 0;
		        else if(gp_num_x==group_num_x-1)
		        	gp_num_y++;
                
		        if(gp_num_x==group_num_x-1)
		        	gp_num_x = 0;
		        //else if (read8_flag==1)
                //    gp_num_x = 2;
                else                    
		            gp_num_x++;

                //printf("read: index=%d,flag=%d,flaga1=%d,read8_flag=%d,win_itm_y=%d,win_itm_z=%d,win_itm_xyz=%d,gp_num_x=%d,gp_num_y=%d,conv_row_rem=%d,group_num_x=%d,group_num_y=%d,conv_z_cnt=%d\n",index,flag,flaga1,read8_flag,win_itm_y,win_itm_z,win_itm_xyz,gp_num_x,gp_num_y,conv_row_rem,group_num_x,group_num_y,conv_z_cnt);


                if (gp_num_x==1 && weight_dim1==3 && (conv_row_rem==1 || conv_row_rem==2)) {
                    read8_flag = 1;
                    //if (flag == 3){
                    //    flag = 0;
                    //    flaga1 = 1;
                    //}else if (flag == 2){
                    //    flag = 3;
                    //    flaga1 = 0;
                    //}else{
                    //    flag++ ;
                    //    flaga1 = flag + 1;
                    //}
                }else{
                    read8_flag = 0;
                    //if (flaga1 == 3){
                    //    flag = 0;
                    //    flaga1 = flag;
                    //}else {
                    //    flag = flaga1 + 1;
                    //    flaga1 = flag;
                    //}
                }


                if (flag==3)
                    flag = 0;
                else 
                    flag++;

                //out_idx_xyz++; 
                win_itm_xyz=0;

            } else {
                win_itm_xyz++;
            }
*/
            //-------------------------------------------------------------------------------------------------------
            //conv deal with-----------------------------------------------------------------------------------------
            //-------------------------------------------------------------------------------------------------------
	        if (conv_z_cnt == 0) {	
		        //bias_ch_out = read_channel_intel(bias_ch);
		        bias_ch_out = bias_ch_in;

		        #pragma unroll
		        for(unsigned char ll=0; ll<LANE_NUM; ll++){
		            #pragma unroll
                    for(unsigned char i=0; i<W_VEC_SIZE; i++){
		        	    conv_out.lane[ll].data[i] = CZERO;
		        	    //bias.lane[ll] = bias_ch_out.lane[ll];
                    }
		        }
            }

		    //for(int j=0; j<conv_loop_cnt; j++){

            //if (conv_z_cnt < conv_loop_cnt) {
		    	//mac_data = read_channel_intel(data_ch);
		    	mac_data = buf_data;

                //if (layer_num==0 && index >win_size*2){
                //    for (uchar i=0;i<W_VEC_SIZE;i++)
                //        for (uchar j=0;j<VEC_SIZE;j++)
                //            printf("mac_data.wvec[%d].data[%d]=%d\n",i,j,mac_data.wvec[i].data[j]);
                //}


		    	mac_weight = weight_buf_vec_rd;

		    	#pragma unroll
		    	for(unsigned char ll=0; ll<LANE_NUM; ll++){
		    	    //mac_weight.lane[ll] = read_channel_intel(weight_ch[ll]);
		    	    #pragma unroll
		    	    for(unsigned char m=0; m<VEC_SIZE;m++){
                        //conv_btmuld.lane[ll].vec[m].data[0] =  (mac_data.lane[ll].vec[m].data[0] << 2) - (mac_data.lane[ll].vec[m].data[2] << 2) -  mac_data.lane[ll].vec[m].data[2]       +  mac_data.lane[ll].vec[m].data[4];
                        //conv_btmuld.lane[ll].vec[m].data[1] = -(mac_data.lane[ll].vec[m].data[1] << 2) - (mac_data.lane[ll].vec[m].data[2] << 2) +  mac_data.lane[ll].vec[m].data[3]       +  mac_data.lane[ll].vec[m].data[4];
                        //conv_btmuld.lane[ll].vec[m].data[2] =  (mac_data.lane[ll].vec[m].data[1] << 2) - (mac_data.lane[ll].vec[m].data[2] << 2) -  mac_data.lane[ll].vec[m].data[3]       +  mac_data.lane[ll].vec[m].data[4];
                        //conv_btmuld.lane[ll].vec[m].data[3] = -(mac_data.lane[ll].vec[m].data[1] << 1) -  mac_data.lane[ll].vec[m].data[2]       + (mac_data.lane[ll].vec[m].data[3] << 1) +  mac_data.lane[ll].vec[m].data[4];
                        //conv_btmuld.lane[ll].vec[m].data[4] =  (mac_data.lane[ll].vec[m].data[1] << 1) -  mac_data.lane[ll].vec[m].data[2]       - (mac_data.lane[ll].vec[m].data[3] << 1) +  mac_data.lane[ll].vec[m].data[4];
                        //conv_btmuld.lane[ll].vec[m].data[5] =  (mac_data.lane[ll].vec[m].data[1] << 2) - (mac_data.lane[ll].vec[m].data[3] << 2) -  mac_data.lane[ll].vec[m].data[3]       +  mac_data.lane[ll].vec[m].data[5];

                        //BT*d
                        conv_btmuld.lane[ll].vec[m].data[0] =  (mac_data.wvec[0].data[m] << 2) - (mac_data.wvec[2].data[m] << 2) -  mac_data.wvec[2].data[m]       +  mac_data.wvec[4].data[m];
                        conv_btmuld.lane[ll].vec[m].data[1] = -(mac_data.wvec[1].data[m] << 2) - (mac_data.wvec[2].data[m] << 2) +  mac_data.wvec[3].data[m]       +  mac_data.wvec[4].data[m];
                        conv_btmuld.lane[ll].vec[m].data[2] =  (mac_data.wvec[1].data[m] << 2) - (mac_data.wvec[2].data[m] << 2) -  mac_data.wvec[3].data[m]       +  mac_data.wvec[4].data[m];
                        conv_btmuld.lane[ll].vec[m].data[3] = -(mac_data.wvec[1].data[m] << 1) -  mac_data.wvec[2].data[m]       + (mac_data.wvec[3].data[m] << 1) +  mac_data.wvec[4].data[m];
                        conv_btmuld.lane[ll].vec[m].data[4] =  (mac_data.wvec[1].data[m] << 1) -  mac_data.wvec[2].data[m]       - (mac_data.wvec[3].data[m] << 1) +  mac_data.wvec[4].data[m];
                        conv_btmuld.lane[ll].vec[m].data[5] =  (mac_data.wvec[1].data[m] << 2) - (mac_data.wvec[3].data[m] << 2) -  mac_data.wvec[3].data[m]       +  mac_data.wvec[5].data[m];

		    	        #pragma unroll
                        for ( unsigned char n=0;n<W_VEC_SIZE;n++){ 
                            //conv_data_mux.lane[ll].vec[m].data[n] = fc_en ? mac_data.lane[ll].vec[m].data[n] : conv_btmuld.lane[ll].vec[m].data[n];
                            conv_data_mux.lane[ll].vec[m].data[n] = fc_en ? mac_data.wvec[n].data[m] : conv_btmuld.lane[ll].vec[m].data[n];
                        }
                    }//end of m loop         

                    #if VEC_SIZE==8 && LANE_NUM*VEC_SIZE*6 >FPGA_DSP_NUM
                        if (ll<LANE_NUM-1){
                    #endif
                    #if VEC_SIZE==16 && LANE_NUM*VEC_SIZE*6 >FPGA_DSP_NUM
                        if (ll<31){
                    #endif
                    #if VEC_SIZE==8
		    	            #pragma unroll
                            for ( unsigned char n=0;n<W_VEC_SIZE;n++) {
                                //conv_btdxgw.lane[ll].vec[m].data[n] = conv_data_mux.lane[ll].vec[m].data[n] * mac_weight.lane[ll].vec[m].data[n]; 
                                mac_out.lane[ll].data[n] = mult_add_fix8bx16bx4(conv_data_mux.lane[ll].vec[0].data[n], mac_weight.lane[ll].vec[0].data[n], 
                                                                                conv_data_mux.lane[ll].vec[1].data[n], mac_weight.lane[ll].vec[1].data[n], 
                                                                                conv_data_mux.lane[ll].vec[2].data[n], mac_weight.lane[ll].vec[2].data[n], 
                                                                                conv_data_mux.lane[ll].vec[3].data[n], mac_weight.lane[ll].vec[3].data[n])+ 
                                                           mult_add_fix8bx16bx4(conv_data_mux.lane[ll].vec[4].data[n], mac_weight.lane[ll].vec[4].data[n], 
                                                                                conv_data_mux.lane[ll].vec[5].data[n], mac_weight.lane[ll].vec[5].data[n], 
                                                                                conv_data_mux.lane[ll].vec[6].data[n], mac_weight.lane[ll].vec[6].data[n], 
                                                                                conv_data_mux.lane[ll].vec[7].data[n], mac_weight.lane[ll].vec[7].data[n]);
                            }
                    #else
                            #pragma unroll
                            for ( unsigned char n=0;n<W_VEC_SIZE;n++) {
                                //conv_btdxgw.lane[ll].vec[m].data[n] = conv_data_mux.lane[ll].vec[m].data[n] * mac_weight.lane[ll].vec[m].data[n]; 
                                mac_out.lane[ll].data[n] = mult_add_fix8bx16bx4(conv_data_mux.lane[ll].vec[0].data[n], mac_weight.lane[ll].vec[0].data[n],
                                                                                conv_data_mux.lane[ll].vec[1].data[n], mac_weight.lane[ll].vec[1].data[n],
                                                                                conv_data_mux.lane[ll].vec[2].data[n], mac_weight.lane[ll].vec[2].data[n],
                                                                                conv_data_mux.lane[ll].vec[3].data[n], mac_weight.lane[ll].vec[3].data[n])+
                                                           mult_add_fix8bx16bx4(conv_data_mux.lane[ll].vec[4].data[n], mac_weight.lane[ll].vec[4].data[n],
                                                                                conv_data_mux.lane[ll].vec[5].data[n], mac_weight.lane[ll].vec[5].data[n],
                                                                                conv_data_mux.lane[ll].vec[6].data[n], mac_weight.lane[ll].vec[6].data[n],
                                                                                conv_data_mux.lane[ll].vec[7].data[n], mac_weight.lane[ll].vec[7].data[n])+
                                                           mult_add_fix8bx16bx4(conv_data_mux.lane[ll].vec[8].data[n], mac_weight.lane[ll].vec[8].data[n],
                                                                                conv_data_mux.lane[ll].vec[9].data[n], mac_weight.lane[ll].vec[9].data[n],
                                                                                conv_data_mux.lane[ll].vec[10].data[n], mac_weight.lane[ll].vec[10].data[n],
                                                                                conv_data_mux.lane[ll].vec[11].data[n], mac_weight.lane[ll].vec[11].data[n])+
                                                           mult_add_fix8bx16bx4(conv_data_mux.lane[ll].vec[12].data[n], mac_weight.lane[ll].vec[12].data[n],
                                                                                conv_data_mux.lane[ll].vec[13].data[n], mac_weight.lane[ll].vec[13].data[n],
                                                                                conv_data_mux.lane[ll].vec[14].data[n], mac_weight.lane[ll].vec[14].data[n],
                                                                                conv_data_mux.lane[ll].vec[15].data[n], mac_weight.lane[ll].vec[15].data[n]);
                            }
                    #endif

                    //#ifdef VECXLANE_CTRL_VEC8
                    #if VEC_SIZE==8 && LANE_NUM*VEC_SIZE*6 >FPGA_DSP_NUM
                        }else{
		    	            #pragma unroll
                            for ( unsigned char n=0;n<W_VEC_SIZE;n++) {
                                mac_out.lane[ll].data[n] = conv_data_mux.lane[ll].vec[0].data[n]* mac_weight.lane[ll].vec[0].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[1].data[n]* mac_weight.lane[ll].vec[1].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[2].data[n]* mac_weight.lane[ll].vec[2].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[3].data[n]* mac_weight.lane[ll].vec[3].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[4].data[n]* mac_weight.lane[ll].vec[4].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[5].data[n]* mac_weight.lane[ll].vec[5].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[6].data[n]* mac_weight.lane[ll].vec[6].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[7].data[n]* mac_weight.lane[ll].vec[7].data[n];
                            }
                        }
                    #endif
                    #if VEC_SIZE==16 && LANE_NUM*VEC_SIZE*6 >FPGA_DSP_NUM
                        }else{
                            #pragma unroll
                            for ( unsigned char n=0;n<W_VEC_SIZE;n++) {
                                mac_out.lane[ll].data[n] = conv_data_mux.lane[ll].vec[0].data[n]* mac_weight.lane[ll].vec[0].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[1].data[n]* mac_weight.lane[ll].vec[1].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[2].data[n]* mac_weight.lane[ll].vec[2].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[3].data[n]* mac_weight.lane[ll].vec[3].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[4].data[n]* mac_weight.lane[ll].vec[4].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[5].data[n]* mac_weight.lane[ll].vec[5].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[6].data[n]* mac_weight.lane[ll].vec[6].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[7].data[n]* mac_weight.lane[ll].vec[7].data[n]+
                                                           conv_data_mux.lane[ll].vec[8].data[n]* mac_weight.lane[ll].vec[8].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[9].data[n]* mac_weight.lane[ll].vec[9].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[10].data[n]* mac_weight.lane[ll].vec[10].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[11].data[n]* mac_weight.lane[ll].vec[11].data[n]+
                                                           conv_data_mux.lane[ll].vec[12].data[n]* mac_weight.lane[ll].vec[12].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[13].data[n]* mac_weight.lane[ll].vec[13].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[14].data[n]* mac_weight.lane[ll].vec[14].data[n]+ 
                                                           conv_data_mux.lane[ll].vec[15].data[n]* mac_weight.lane[ll].vec[15].data[n];

                            }    
                        }                                     
                    #endif




                    //conv_atxbtdxgw.lane[ll].vec[m].data[0] = conv_btdxgw.lane[ll].vec[m].data[0] + conv_btdxgw.lane[ll].vec[m].data[1] + (conv_btdxgw.lane[ll].vec[m].data[2]     ) + (conv_btdxgw.lane[ll].vec[m].data[3]     ) + (conv_btdxgw.lane[ll].vec[m].data[4] );
                    //conv_atxbtdxgw.lane[ll].vec[m].data[1] = conv_btdxgw.lane[ll].vec[m].data[1] - conv_btdxgw.lane[ll].vec[m].data[2] + (conv_btdxgw.lane[ll].vec[m].data[3] << 1) - (conv_btdxgw.lane[ll].vec[m].data[4] << 1);
                    //conv_atxbtdxgw.lane[ll].vec[m].data[2] = conv_btdxgw.lane[ll].vec[m].data[1] + conv_btdxgw.lane[ll].vec[m].data[2] + (conv_btdxgw.lane[ll].vec[m].data[3] << 2) + (conv_btdxgw.lane[ll].vec[m].data[4] << 2);
                    //conv_atxbtdxgw.lane[ll].vec[m].data[3] = conv_btdxgw.lane[ll].vec[m].data[1] - conv_btdxgw.lane[ll].vec[m].data[2] + (conv_btdxgw.lane[ll].vec[m].data[3] << 3) - (conv_btdxgw.lane[ll].vec[m].data[4] << 3) + (conv_btdxgw.lane[ll].vec[m].data[5] );
                    //conv_atxbtdxgw.lane[ll].vec[m].data[4] = 0;
                    //conv_atxbtdxgw.lane[ll].vec[m].data[5] = 0;
                    conv_atxbtdxgw.lane[ll].data[0] = mac_out.lane[ll].data[0] + mac_out.lane[ll].data[1] + (mac_out.lane[ll].data[2]     ) + (mac_out.lane[ll].data[3]     ) + (mac_out.lane[ll].data[4] );
                    conv_atxbtdxgw.lane[ll].data[1] = mac_out.lane[ll].data[1] - mac_out.lane[ll].data[2] + (mac_out.lane[ll].data[3] << 1) - (mac_out.lane[ll].data[4] << 1);   
                    conv_atxbtdxgw.lane[ll].data[2] = mac_out.lane[ll].data[1] + mac_out.lane[ll].data[2] + (mac_out.lane[ll].data[3] << 2) + (mac_out.lane[ll].data[4] << 2);   
                    conv_atxbtdxgw.lane[ll].data[3] = mac_out.lane[ll].data[1] - mac_out.lane[ll].data[2] + (mac_out.lane[ll].data[3] << 3) - (mac_out.lane[ll].data[4] << 3) + (mac_out.lane[ll].data[5] );
                    conv_atxbtdxgw.lane[ll].data[4] = 0;
                    conv_atxbtdxgw.lane[ll].data[5] = 0;

		    	    #pragma unroll
                    for ( unsigned char n=0;n<W_VEC_SIZE;n++){ 
                        conv_dxw_mux.lane[ll].data[n] = fc_en ? mac_out.lane[ll].data[n] : conv_atxbtdxgw.lane[ll].data[n];
                    }

		    	    #pragma unroll
                    for ( unsigned char n=0;n<W_VEC_SIZE;n++) {
		                //conv_out.lane[ll].data[n] = (MASK_ACCUM & conv_out.lane[ll].data[n]) + (MASK_MULT & conv_dxw_mux.lane[ll].vec[m].data[n]);
		                conv_out.lane[ll].data[n] =  conv_out.lane[ll].data[n] +  conv_dxw_mux.lane[ll].data[n];
                    }

		    		#ifdef DEBUG_CONV
		    		//if(ll==0 && index==9){
		    		if(layer_num==13 && ll==0 && index>=1345&&index<1440){
                        //for ( unsigned char m=0;m<W_VEC_SIZE;m++) { 
                        for ( unsigned char m=0;m<3;m++) { 
                            for ( unsigned char n=0;n<VEC_SIZE;n++) { 
		    		            printf("dot_cnt=%d data=% 10.6f weight=% 10.6f bias=% 10.6f conv_btmuld.lane[0].vec[%d].data[%d]=% 10.6f,mac_out.lane[0].data[%d]=% 10.6f,conv_atxbtdxgw.lane[0].data[%d]=% 10.6f,conv_out[%d] =% 10.6f (loop=%3d, lane= %2d, wvec=%2d, vec=%2d)\n", index, (float)mac_data.wvec[m].data[n], (float)mac_weight.lane[ll].vec[m].data[n], (float)bias_ch_out.lane[ll],n,n,m,(float)conv_btmuld.lane[ll].vec[n].data[m],m, (float)mac_out.lane[ll].data[m],m,(float)conv_atxbtdxgw.lane[ll].data[m],(float)conv_out.lane[ll].data[m], conv_z_cnt, ll, m,n);
                            }
                        }
                        //printf("read: index=%d,flag=%d,flaga1=%d,read8_flag=%d,win_itm_y=%d,win_itm_z=%d,win_itm_xyz=%d,gp_num_x=%d,gp_num_y=%d,conv_row_rem=%d,group_num_x=%d,group_num_y=%d\n",index,flag,flaga1,read8_flag,win_itm_y,win_itm_z,win_itm_xyz,gp_num_x,gp_num_y,conv_row_rem,group_num_x,group_num_y);
                    }
		    		#endif
		    	}//end of ll loop
		    //}// end of j loop


            if (conv_z_cnt == conv_loop_cnt -1) {
		        #pragma unroll
		        for(unsigned char ll=0; ll<LANE_NUM; ll++){

		            #ifdef DEBUG_CONV
		    		if(layer_num==13 && ll==0 && index>=1345&&index<1440){
		    		//if(layer_num==13 && ll==0 && index==1440){
		        	//if(ll==14 && k==69444){
                        for ( unsigned char n=0;n<W_VEC_SIZE;n++) { 
		                    printf("\ndot_cnt=%d conv_out=%f (lane= %d, wvec=%d)\n", index, (float)conv_out.lane[ll].data[n], ll,n);
                        }
		            }
		            #endif
		        	
		            #pragma unroll
                    for(unsigned char i=0; i<W_VEC_SIZE; i++){
		        	    //if(conv_out.lane[ll].data[i]>=0)
		        	    //	conv_sign_exten.lane[ll].data[i] = 0x00;
		        	    //else
		        	    //	conv_sign_exten.lane[ll].data[i] = ~(0xFFFFFFFF>>(frac_w+frac_din-frac_dout-1));
		        	    
		        	    //conv_with_rnd_bit.lane[ll].data[i] = (conv_sign_exten.lane[ll].data[i] | (conv_out.lane[ll].data[i]>>(frac_w+frac_din-frac_dout-1))) + 0x01;
		        	    //conv_with_rnd_bit.lane[ll].data[i] = (conv_out.lane[ll].data[i]>>(frac_w+frac_din-frac_dout-1)) + 0x01;
		        	    conv_with_rnd_bit.lane[ll].data[i] = (conv_out.lane[ll].data[i]>>(frac_w+frac_din-frac_dout-1));
                    }
                    
                    #ifdef DEBUG_CONV
		    		if(layer_num==13 && ll==0 && index>=1345&&index<1440){
		    		//if(layer_num==13 && ll==0 && index==1440){
		    		//if(layer_num==2 && ll==0 && index==55488){
		    		//if(ll==0 && index==3){
		        	//if(ll==14 && k==69444){
		            //if(ll==13 && k==135){
		        	//if(ll==0 && k==0){
                        for ( unsigned char n=0;n<W_VEC_SIZE;n++) { 
		                    printf("\ndot_cnt=%d conv_with_rnd_bit=%f (lane= %d, wvec=%d)\n",index , (float)conv_with_rnd_bit.lane[ll].data[n], ll,n);
                        }
		            }
		            #endif

		            #pragma unroll
                    for(unsigned char i=0; i<W_VEC_SIZE; i++){
		        	    if (frac_b-frac_dout>1)
                            conv_sum_bias.lane[ll].data[i] =  (MASK9B & conv_with_rnd_bit.lane[ll].data[i])+(bias_ch_out.lane[ll]>>(frac_b-frac_dout-1))+0x01;
                        else
                            conv_sum_bias.lane[ll].data[i] =  (MASK9B & conv_with_rnd_bit.lane[ll].data[i])+(bias_ch_out.lane[ll]<<(1-(frac_b-frac_dout)))+0x01;
                    }

                    #ifdef DEBUG_CONV
		    		if(layer_num==13 && ll==0 && index>=1345&&index<1440){
		    		//if(layer_num==13 && ll==0 && index==1440){
		    		//if(layer_num==2 && ll==0 && index==55488){
		    		//if(ll==0 && index==3){
		        	//if(ll==14 && k==69444){
		            //if(ll==13 && k==135){
		        	//if(ll==0 && k==0){
                        for ( unsigned char n=0;n<W_VEC_SIZE;n++) { 
		                    printf("\ndot_cnt=%d, bias[ll]<<(1-frac_br+frac_dout)=%d, conv_sum_bias=%d (lane= %d), MASK9B& conv_with_rnd_bit[ll]=%d\n", index, (bias_ch_out.lane[ll]<<(1-frac_b+frac_dout)), conv_sum_bias.lane[ll].data[n], ll,MASK9B & conv_with_rnd_bit.lane[ll].data[n]);
                        }
		            }
		            #endif

		            #pragma unroll
                    for(unsigned char i=0; i<W_VEC_SIZE; i++){
                        if (conv_sum_bias.lane[ll].data[i] >= 256)
                            conv_final.lane[ll].data[i] = 127;
                        else if (conv_sum_bias.lane[ll].data[i] < -256)
                            conv_final.lane[ll].data[i] = -128;
                        else
		        	        conv_final.lane[ll].data[i] = conv_sum_bias.lane[ll].data[i]>>0x01;
		        	        //conv_final.lane[ll].data[i] = MASK8B & (conv_sum_bias.lane[ll].data[i]>>0x01);

		        	    // Relu operation
		        	    if((control&0x01)==0x01){
		        	    	if((conv_final.lane[ll].data[i] & MASKSIGN) == MASKSIGN) {
                                //leaky_relu[ll]=(conv_final[ll]*LEAKY_RELU);
                                //#ifdef DEBUG_CONV
		                        ////if(ll==1 && k==0){
		        	            //if(ll==0 && k==0){
		                        //    printf("\ndot_cnt=%d, leaky_relu=%f, conv_final=%f (lane= %d)\n", k, (float)leaky_relu[ll], (float)conv_final[ll], ll);
		                        //}
		                        //#endif
                                ////leaky_relu[ll]=(0xFFC00000 | (leaky_relu[ll]>>(LEAKY_RELU_FRAC-1))+0x1);
                                //leaky_relu[ll]=((leaky_relu[ll]>>(LEAKY_RELU_FRAC-1))+0x1);
                                //#ifdef DEBUG_CONV
		        	            //if(ll==0 && k==0){
		                        ////if(ll==1 && k==0){
		                        //    printf("\ndot_cnt=%d, leaky_relu=%x, conv_final=%f (lane= %d)\n", k, leaky_relu[ll], (float)conv_final[ll], ll);
		                        //}
		                        //#endif
                                //leaky_relu[ll]=(leaky_relu[ll]>>1);
                                //#ifdef DEBUG_CONV
		                        ////if(ll==1 && k==0){
		        	            //if(ll==0 && k==0){
		                        //    printf("\ndot_cnt=%d, leaky_relu=%x, conv_final=%f (lane= %d)\n", k, leaky_relu[ll], (float)conv_final[ll], ll);
		                        //}
		                        //#endif
		        	    		//conv_ch_in.lane[ll] = leaky_relu[ll];
		        	    		conv_ch_in.lane[ll].data[i] = 0;
		        	    	} else
		        	    		conv_ch_in.lane[ll].data[i] = conv_final.lane[ll].data[i];
		        	    }
		        	    else
		        	    	conv_ch_in.lane[ll].data[i] = conv_final.lane[ll].data[i];
		            }//end of i loop	
		        	
		        	#ifdef DEBUG_CONV
		    		if(layer_num==13 && ll==0 && index>=1345&&index<1440){
		    		//if(layer_num==13 && ll==0 && index==1440){
		        	//if(ll==0 && k==0){
                        for ( unsigned char n=0;n<W_VEC_SIZE;n++) { 
		        		    printf("dot_cnt=%d sum=%f rnd=%f sum_bias=%f final=%f final=%x (bias=%f lane=%d wvec=%d)\n\n", index, (float)conv_out.lane[ll].data[n], (float)conv_with_rnd_bit.lane[ll].data[n], (float)conv_sum_bias.lane[ll].data[n], (float)conv_final.lane[ll].data[n], conv_final.lane[ll].data[n], (float)bias_ch_out.lane[ll], ll, n);
                        }
                    }
		        	#endif

		        }//end of ll loop

		        // write convoluation results
		        if((control&0x02)==0x02 ){
		        	//by-pass pooling
		        	write_channel_intel(bypass_ch, conv_ch_in);
                }
		        // to pooling kernel
		        if((control&0x02)==0x00 ){
		        	//write_channel_intel(conv_ch, conv_ch_in);

                    //------------------------------------------------------------------------------------------------------
                    //-----------pooling deal with -------------------------------------------------------------------------
                    //------------------------------------------------------------------------------------------------------
                    #pragma unroll
		            for(unsigned char ll=0; ll<LANE_NUM; ll++){

		            	row_pool_reg.lane[ll] = line_buf_0[line_buf_ptr].lane[ll];
		            	
		            	pool_reg.lane[ll].data[0] = pool_max(conv_ch_in.lane[ll].data[0], conv_ch_in.lane[ll].data[1]);
		            	pool_reg.lane[ll].data[1] = pool_max(conv_ch_in.lane[ll].data[2], conv_ch_in.lane[ll].data[3]);
		            	

		            	pool_final.lane[ll].data[0] = pool_max(row_pool_reg.lane[ll].data[0], pool_reg.lane[ll].data[0]);
		            	pool_final.lane[ll].data[1] = pool_max(row_pool_reg.lane[ll].data[1], pool_reg.lane[ll].data[1]);
		            	pool_final.lane[ll].data[2] = 0;
		            	pool_final.lane[ll].data[3] = 0;
		            	pool_final.lane[ll].data[4] = 0;
		            	pool_final.lane[ll].data[5] = 0;

		            	line_buf_0[line_buf_ptr].lane[ll].data[0]= pool_reg.lane[ll].data[0];
		            	line_buf_0[line_buf_ptr].lane[ll].data[1]= pool_reg.lane[ll].data[1];

		            }
		            
		            #ifdef DEBUG_POOL
		                printf("Maxpool conv_loop_cnt=%d,conv_z_cnt=%d,row_cnt=%d, line_buf_ptr=%d, row_pool_cnt=%d\n",conv_loop_cnt, conv_z_cnt, row_cnt,line_buf_ptr, row_pool_cnt );
		            #endif
		            
		            // Generates pooling pipeline register wr/rd pointer
		            if(row_pool_cnt==(pool_size-1)){
		           		write_channel_intel(pool_ch, pool_final);

		            }

		            // Generates line buffer wr/rd pointer
		            if(line_buf_ptr==(col_size-1)){
		            	line_buf_ptr = 0;

		            	// Row counters for recognize frames
		            	if(row_cnt == (line_size-1)) // assuming row_num = line_size, i.e. rectangular frame
		            		row_cnt = 0;
		            	else
		            		row_cnt = row_cnt + 1;

		            	// Pooling window slide counter for rows
		            	if(row_cnt == 0)
		            		row_pool_cnt = 0;
		            	else if(row_pool_cnt==(pool_size-1))
		            		row_pool_cnt = (pool_size-pool_stride);
		            	else
		            		row_pool_cnt = row_pool_cnt + 1;
		            }
		            else{
		            	line_buf_ptr = line_buf_ptr + 1;
		            }
                }
                //pooling completed
		        //printf("Write channel item-%d is written in channel %d...\n", k, ll);
                //printf("conv:output number k = %d \n",k);

                //if (conv_row_cnt == conv_row_num) {
                //    conv_row_cnt = 0; 
                //}else{
                //    conv_row_cnt++;
                //}

                conv_z_cnt = 0;
            } else 
                //if (index >win_size*2 && gp_num_x !=2 && weight_dim1==3 && (conv_row_rem==1 || conv_row_rem==2)) {
                if (index >win_size*2 && read8_flag == 0){ 
                    conv_z_cnt++; 
                
            }//end if (conv_z_cnt == conv_loop_cnt -1) 
                
		    if (win_itm_xyz==win_size-1) {
		        // used as virtual group loop counters
		        if((out_idx_z==weight_dim4_div_lane-1) && (gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
		        	out_idx_z = 0;
		        else if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
		        	out_idx_z++;	
                
		        if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
		        	gp_num_y = 0;
		        else if(gp_num_x==group_num_x-1)
		        	gp_num_y++;
                
		        if(gp_num_x==group_num_x-1)
		        	gp_num_x = 0;
		        //else if (read8_flag==1)
                //    gp_num_x = 2;
                else                    
		            gp_num_x++;

                //printf("read: index=%d,flag=%d,flaga1=%d,read8_flag=%d,win_itm_y=%d,win_itm_z=%d,win_itm_xyz=%d,gp_num_x=%d,gp_num_y=%d,conv_row_rem=%d,group_num_x=%d,group_num_y=%d,conv_z_cnt=%d\n",index,flag,flaga1,read8_flag,win_itm_y,win_itm_z,win_itm_xyz,gp_num_x,gp_num_y,conv_row_rem,group_num_x,group_num_y,conv_z_cnt);


                if (gp_num_x==1 && weight_dim1==3 && (conv_row_rem==1 || conv_row_rem==2)) {
                    read8_flag = 1;
                }else{
                    read8_flag = 0;
                }


                if (flag==2)
                    flag = 0;
                else 
                    flag++;

                win_itm_xyz=0;
            } else {
                win_itm_xyz++;
            }

           // printf("read: group_num_mul_win_size=%d,index=%d,win_size=%d,conv_loop_cnt=%d,flag=%d,flaga1=%d,read8_flag=%d,win_itm_y=%d,win_itm_z=%d,win_itm_xyz=%d,gp_num_x=%d,gp_num_y=%d,conv_row_rem=%d,group_num_x=%d,group_num_y=%d,conv_z_cnt=%d\n",group_num_mul_win_size,index,win_size,conv_loop_cnt,flag,flaga1,read8_flag,win_itm_y,win_itm_z,win_itm_xyz,gp_num_x,gp_num_y,conv_row_rem,group_num_x,group_num_y,conv_z_cnt);
		    //printf("conv: group_num_mul_win_size=%d,index=%6d,conv_loop_cnt=%d,conv_z_cnt=%d,row_cnt=%d, line_buf_ptr=%d, row_pool_cnt=%d, gp_num_x=%d,gp_num_y=%d\n",group_num_mul_win_size,index,conv_loop_cnt, conv_z_cnt, row_cnt,line_buf_ptr, row_pool_cnt, gp_num_x,gp_num_y);


	}//while end 
	//printf("\nKernel memRd finished !!!\n");
}

__kernel
void memWrite(
				// Params Ports
                uchar   layer_num,        // weight_dim4/lane_num*((ceil(conv_x/q_vec)-1)*q_vec+rem_size_x)*scal*conv_y
                uint    out_num,        // weight_dim4/lane_num*((ceil(conv_x/q_vec)-1)*q_vec+rem_size_x)*scal*conv_y
                uchar   q_vec,
                uchar   rem_size_x,     // including next_layer_padding
                uchar   start_size_x,     // including next_layer_padding
				uchar   out_dim1_div_q_vec,       // ceil(conv_x/q_vec)
				uchar   next_layer_padding,       // ceil(conv_x/q_vec)
				ushort  out_dim3_div_lane_numxscal,       // ceil(conv_x/q_vec)
				ushort  out_dim1,       //conv_x+2*next_layer_padding
				ushort  out_dim2,       //conv_y
				//ushort  out_dim3_div_lane_num,
				uint    out_dim1xdim2,  //conv_x*conv_y     
			    uchar   bypass,
                uchar   scal,   //(next layer weight_n>lane_num)?(lane_num/vec_size):(weight_n/vec_size)   
                uchar   scal_rem_z,   //(next layer weight_n>lane_num)?(lane_num/vec_size):(weight_n/vec_size)   
                uchar   scalxq_vec,   //(next layer weight_n>lane_num)?(lane_num/vec_size):(weight_n/vec_size)   
                uchar   scalxrem_size_x,   //(next layer weight_n>lane_num)?(lane_num/vec_size):(weight_n/vec_size)   
                uchar   scalxstart_size_x,   //(next layer weight_n>lane_num)?(lane_num/vec_size):(weight_n/vec_size)   
                uchar   scal_rem_zxq_vec,   //(next layer weight_n>lane_num)?(lane_num/vec_size):(weight_n/vec_size)   
                uchar   scal_rem_zxrem_size_x,   //(next layer weight_n>lane_num)?(lane_num/vec_size):(weight_n/vec_size)   
                uchar   scal_rem_zxstart_size_x,   //(next layer weight_n>lane_num)?(lane_num/vec_size):(weight_n/vec_size)   
                uint    dim_z_edge_num,   //(next layer weight_n>lane_num)?(lane_num/vec_size):(weight_n/vec_size)   
				// Data Ports
                __global lane_data *restrict top
				)
{
    
    channel_wvec_wng __attribute__((register)) ch_data;
    lane_data   output;
    ushort x_dim=0;
    ushort padding_tmp=0;
    bool start_flag=1;
    bool end_flag=0;
    ushort y_dim=0;
    ushort z_dim=0;
    uchar vec_num;
    uchar loop_num;
    uchar width=0;
    uchar width_cnt;
    uchar lane_cnt;
    uchar inner_loop=0;
    uint  index=0;
    lane_data vec_zero;
    
    #pragma unroll
    for(uchar i=0; i<VEC_SIZE; i++)
        vec_zero.data[i]=CZERO;
    
    //printf("\nmemWr 1: out_num=%d\n", out_num);
    
    //for (uint i=0; i<out_num; i++) {
    while(index!=out_num){
        index++;
        //if (layer_num==0)
        //    printf("memWr 1: index=%d, inner_loop=%d \n", index, inner_loop);
        //if (x_dim==0) {
        //    start_flag=1;
        //    padding_tmp=0;
        //} else {
        //    start_flag=0;
        //    padding_tmp=next_layer_padding;
        //}

        //if (x_dim==out_dim1_div_q_vec-1)
        //    end_flag=1;
        //else 
        //    end_flag=0;

        if (inner_loop==0 ) { 
	        if((bypass&0x01)==0x01) {
		    	ch_data = read_channel_intel(bypass_ch);
		    } else{
		    	ch_data = read_channel_intel(pool_ch);
            }
            
            if(index>=dim_z_edge_num && start_flag==1 ) { 
                width=start_size_x;
                loop_num=scal_rem_zxstart_size_x;
            } else if(index>=dim_z_edge_num && end_flag==1) { 
                width=rem_size_x;
                loop_num=scal_rem_zxrem_size_x;
            } else if(index>=dim_z_edge_num) { 
                width=q_vec;
                loop_num=scal_rem_zxq_vec;
            } else if(start_flag==1) { 
                width=start_size_x;
                loop_num=scalxstart_size_x;
            } else if(end_flag==1) { 
                width=rem_size_x;
                loop_num=scalxrem_size_x;
            } else {
                width=q_vec;
                loop_num=scalxq_vec;
            }
            
            //if(index>=dim_z_edge_num ) 
            //    vec_num=scal_rem_z;
            //else
            //    vec_num=scal;
            //     
            //
            //if( start_flag==1 ) { 
            //    width=start_size_x;
            //} else if( end_flag==1) { 
            //    width=rem_size_x;
            //} else {
            //    width=q_vec;
            //}

            //loop_num=vec_num*width;
            //if(layer_num==0) 
            //    printf("memWr 2: width=%d, vec_num=%d, loop_num=%d \n", width, vec_num, loop_num);
            
            //if (x_dim==out_dim1_div_q_vec-1) {
            //    width=rem_size_x;
            //} else {
            //    width=q_vec;
            //}

            //loop_num=width*scal;
            width_cnt=0;
            lane_cnt=0;
        }
        #ifdef DEBUG_MEMWR
        if(layer_num==0) {
            printf("memWr 1: out_num=%d, i=%d,i width=%d, width_cnt=%d, lane_cnt=%d, loop_num=%d, inner_loop=%d, q_vec=%d, rem_size_x=%d, wr_pointer=%d\n", out_num, i, width, width_cnt, lane_cnt, loop_num, inner_loop, q_vec, rem_size_x, (z_dim+lane_cnt)*out_dim1xdim2+y_dim*out_dim1+x_dim*q_vec+width_cnt);
            //printf("\nmemWr 2: z_dim=%d, lane_cnt=%d, out_dim1xdim2=%d, y_dim=%d, out_dim1=%d, x_dim=%d, q_vec=%d, width_cnt=%d\n",z_dim, lane_cnt, out_dim1xdim2, y_dim, out_dim1, x_dim, q_vec, width_cnt );
        }
        #endif
        //for(uchar j=0; j<loop_num; j++) {
            
            if (((start_flag==1 && width_cnt==0) || (end_flag==1 && width_cnt==rem_size_x-1)) && next_layer_padding==1)
                output=vec_zero;
            else if (start_flag==1 && width_cnt<start_size_x  && next_layer_padding==1)
                #pragma unroll
                for(uchar k=0;k<VEC_SIZE; k++) {
                    output.data[k]=ch_data.lane[lane_cnt*VEC_SIZE+k].data[width_cnt-1];
                }
            else {
                #pragma unroll
                for(uchar k=0;k<VEC_SIZE; k++) {
                    output.data[k]=ch_data.lane[lane_cnt*VEC_SIZE+k].data[width_cnt];
                }
            }
            
            top[(z_dim+lane_cnt)*out_dim1xdim2+y_dim*out_dim1+x_dim*q_vec+padding_tmp+width_cnt]=output;
            
            //if (width_cnt==width-1 && lane_cnt==scal-1) 
            //    lane_cnt=0;
            //else if (width_cnt==width-1) 
            //    lane_cnt++;
            
            if (width_cnt==width-1) {
                lane_cnt++;
                width_cnt=0;
            } else { 
                width_cnt++;
            }
        //}
        if (inner_loop==loop_num-1) {
            //if(x_dim==out_dim1_div_q_vec-1 && y_dim==out_dim2-1)
            //    z_dim=z_dim+scal; 
            
            //if(x_dim==out_dim1_div_q_vec-1 && y_dim==out_dim2-1) {
            if(end_flag && y_dim==out_dim2-1) {
                z_dim=z_dim+scal; 
                y_dim=0;
            } else if(end_flag)
            //} else if(x_dim==out_dim1_div_q_vec-1 ) 
                y_dim++;
            
            
            //if(x_dim==out_dim1) 
            //if(x_dim==out_dim1_div_q_vec-1) 
            if(x_dim==out_dim1_div_q_vec-1) 
                x_dim=0;
            //else if(start_flag) 
            //    x_dim=x_dim+start_size_x;
            else 
                //x_dim=x_dim+width;
                x_dim++;
        
            if (x_dim==0) {
                start_flag=1;
                padding_tmp=0;
            } else {
                start_flag=0;
                padding_tmp=next_layer_padding;
            }

            if (x_dim==out_dim1_div_q_vec-1)
                end_flag=1;
            else 
                end_flag=0;

            
            inner_loop=0;
        }
        else
            inner_loop++;
    }//while end
	//printf("\nKernel memWr finished !!!\n");
}

