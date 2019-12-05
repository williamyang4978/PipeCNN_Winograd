/*
 * --------------------------------------------------------------------------
 *
 * Copyright (C) 2019 Lenovo
 * Licensed under BSD-3, see COPYING.BSD file for details
 *
 * --------------------------------------------------------------------------
 *
 * On the basis of pipecnn, we added the configuration file
 * because we implemented CNN with Winograd algorithm.
 *
 * --------------------------------------------------------------------------
 *
 *
 */

#pragma OPENCL EXTENSION cl_intel_channels : enable
#include "hw_param.cl"

// Define the precision of the data-path
typedef char DPTYPE;
typedef short CONVTYPE;
typedef int  MACTYPE;

#if VEC_SIZE==8
    typedef int8  intvec;
#else
    typedef int16  intvec;
#endif

// Vectorized data type
typedef struct {
   DPTYPE data[VEC_SIZE];
} lane_data;

typedef struct {
   DPTYPE data[W_VEC_SIZE];
} w_vec_data;

typedef struct {
   DPTYPE data[WEIGHT_W_VEC_SIZE];
} ddr_weight;

typedef struct {
   lane_data wvec[W_VEC_SIZE];
} data_wng;

typedef struct {
    lane_data wvec[4];
} data_wng_ddr;

typedef struct {
    lane_data wvec[8];
} data_wng_vec8;

typedef struct {
   ddr_weight vec[VEC_SIZE];
} ddr_weight_wng;

// Vectorized sparse sign type
typedef struct {
   ushort data[VEC_SIZE];
} lane_sign;

// Combined vec-data type from multiple lane
typedef struct {
   lane_data lane[LANE_NUM];
} channel_vec;

typedef struct {
   channel_vec w_vec[W_VEC_SIZE];
} weight_tmp;

typedef struct {
   ddr_weight_wng lane[LANE_NUM];
} channel_vec_wng;

// Combined scalar data type from multiple lane
typedef struct {
   DPTYPE lane[LANE_NUM];
} channel_scal;

// Vectorized data type
typedef struct {
   char data[LANE_NUM];
} channel_sign_scal;

// Combined vec-sign type from multiple lane
typedef struct {
   lane_sign lane[LANE_NUM];
} channel_sign_vec;

typedef struct {
   uchar data[VEC_SIZE];
} sparse_data;

typedef struct {
   uchar data[LANE_NUM];
} sparse_sign;



typedef struct {
   w_vec_data lane[LANE_NUM];
} channel_wvec_wng;

typedef struct {
   CONVTYPE data[W_VEC_SIZE];
} w_vec_data_16;

typedef struct {
   MACTYPE data[W_VEC_SIZE];
} w_vec_data_32;

typedef struct {
   w_vec_data_16 vec[VEC_SIZE];
} data_16_wng;

typedef struct {
   data_16_wng lane[LANE_NUM];
} channel_vec_16_wng;

typedef struct {
   w_vec_data_32 vec[VEC_SIZE];
} data_32_wng;

typedef struct {
   data_32_wng lane[LANE_NUM];
} channel_vec_32_wng;

typedef struct {
   w_vec_data_32 lane[LANE_NUM];
} channel_wvec_32_wng;

typedef struct {
   DPTYPE data[W_VEC_SIZE/2-1];
} w_vec_pool;

typedef struct {
   w_vec_pool lane[LANE_NUM];
} pool_vec_wng;



//channel channel_vec_wng         data_ch    __attribute__((depth(0)));
channel data_wng         data_ch    __attribute__((depth(0)));
//channel channel_vec_wng         weight_ch  __attribute__((depth(0)));
channel ddr_weight_wng           weight_ch[LANE_NUM]  __attribute__((depth(0)));
channel channel_scal            bias_ch    __attribute__((depth(8)));
channel channel_wvec_wng        conv_ch    __attribute__((depth(CHN_DEPTH)));
channel channel_wvec_wng        pool_ch    __attribute__((depth(CHN_DEPTH)));
channel channel_wvec_wng        bypass_ch  __attribute__((depth(CHN_DEPTH)));
