/*
 * --------------------------------------------------------------------------
 * 
 * PipeCNN_Winograd: PipeCNN accelerated by Winograd
 * 
 * --------------------------------------------------------------------------
 * 
 * Filename:
 *    - object2py.cpp
 * 
 * Author(s):
 *    -Anrong Yang, yangar@lenovo.com
 * 
 * History:
 *    - v1.0 initial version. 
 * 
 * ---------------------------------------------------------------------------------------------
 * 
 * Copyright (C) 2019 Lenovo
 * Licensed under BSD-3, see COPYING.BSD file for details.
 * 
 * ---------------------------------------------------------------------------------------------
 * 
 * This file contains the API for 8-bit quantization, which are called by modelWinogradCut.py.
 *
 *
 * -------------------------------------------------------------------------------------------- 
 */ 

#include<string>
#include <math.h>
#include<iostream>
#include<boost/python.hpp>
#include <numpy/arrayobject.h>
using namespace std;
using namespace boost::python;

namespace bp = boost::python;

int g_baise_padding_flag = 0;
int g_total_size = 0;



int binary_Winograd1X1_func(object & data_obj,object C, object z, object y, object x, object l, object pad)
{
	PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
	float * data = static_cast<float *>(PyArray_DATA(data_arr));
	
	// using data
	FILE *fp = fopen("./weight.dat","ab+");
	//FILE *fp = fopen("./weight.txt","w");
	if(fp == NULL){
		return 0;
	}
	
	int channel= extract<int>(C);
	int lineZ= extract<int>(z);
	int lineY = extract<int>(y);
	int lineX = extract<int>(x);
	int lineNum= extract<int>(l);
	
	g_baise_padding_flag = extract<int>(pad);
	//int *SumZero= extract<int*>(get_int(Sum));
	
	int len = lineX*lineY;
	int lenZ = lineX*lineY*lineZ;
	
	int mallocLen = channel*lineZ*(lineX+5)*lineY;
	
	//int mallocLen = channel*lineZ*(lineX+7)*lineY;
	
	char *datatmp =(char *) malloc(mallocLen);
	if(datatmp == NULL){
		printf("malloc error!!\n");
		return 0;
	}
	memset(datatmp,0,mallocLen);

	//printf("%d,%d,%d,%d,%d\n",lineX,lineY,lineZ,channel,mallocLen);
	int dataLoopNum = 0;
	
	for (int c = 0; c < channel; c++)
	{
		for (int i = 0; i < lineZ/lineNum; i++)
		{
			for(int j=0;j<lineY;j++)
			{
				for (int k = 0; k < lineNum; k++)
				{
					
					for (int p = 0; p < 6; p++)
					{
						int lenSize = c*lenZ+i*len*lineNum+k*len+j*lineX;
						if(data[lenSize]){
							char dataRound = round(data[lenSize]);
							datatmp[dataLoopNum] = dataRound;
							if(round(data[lenSize])==128){
								datatmp[dataLoopNum] = 127;
							}
			
						}
						dataLoopNum++;
					}
					
					//dataLoopNum=dataLoopNum+2;
									
				}
						
			}
					
		}
	}
	
	fwrite(datatmp,mallocLen,1,fp);
	g_total_size = g_total_size+mallocLen;
	
	if(g_baise_padding_flag)
	{	
		char buff[1024]={0};
		fwrite(buff,1024,1,fp);
		g_total_size = g_total_size+1024;
	}

		
	fclose(fp);
	
	free(datatmp);
	

	return 0;

}

int binary_Winograd3X3_func(object & data_obj,object C, object z, object y, object x, object l, object pad)
{
    PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
    float * data = static_cast<float *>(PyArray_DATA(data_arr));
	
    // using data
	FILE *fp = fopen("./weight.dat","ab+");
	
	//FILE *fp = fopen("./weight.txt","w");
	if(fp == NULL){
		return 0;
	}
	
	int channel= extract<int>(C);
    int lineZ= extract<int>(z);
	int lineY = extract<int>(y);
	int lineX = extract<int>(x);
	int lineNum= extract<int>(l);
	
	g_baise_padding_flag = extract<int>(pad);
	//int *SumZero= extract<int*>(get_int(Sum));
	
	int len = lineX*lineY;
	int lenZ = lineX*lineY*lineZ;
	
	int mallocLen = channel*lineZ*(lineX)*lineY;
	
	//int mallocLen = channel*lineZ*lineX*lineY;
	char *datatmp =(char *) malloc(mallocLen);
	if(datatmp == NULL){
		printf("malloc error!!\n");
		return 0;
	}
	memset(datatmp,0,mallocLen);

	//printf("%d,%d,%d,%d,%d\n",lineX,lineY,lineZ,channel,mallocLen);
	int dataLoopNum = 0;
	
	for (int c = 0; c < channel; c++)
	{
		for (int i = 0; i < lineZ/lineNum; i++)
		{
			for(int j=0;j<lineY;j++)
			{
				for (int k = 0; k < lineNum; k++)
				{					
					for (int p = 0; p < lineX; p++)
					{
						int lenSize = c*lenZ+i*len*lineNum+k*len+j*lineX+p;
												
						if(data[lenSize]){
							char dataRound = round(data[lenSize]);
                                                        
							datatmp[dataLoopNum] = dataRound;
                                                        
							if(round(data[lenSize])==128){
								datatmp[dataLoopNum] = 127;
							}
			
						}
						
						dataLoopNum++;
					}
				
					//dataLoopNum=dataLoopNum+2;
				}
						
			}
					
		}
	}

	fwrite(datatmp,mallocLen,1,fp);

	
	g_total_size = g_total_size+mallocLen;
	//printf("g_total_size==%d\n",g_total_size);
	fclose(fp);
	
	free(datatmp);
	

    return 0;

}

int binary_bais_func(object & data_obj, object x)
{
    PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
    float * data = static_cast<float *>(PyArray_DATA(data_arr));
	
	FILE *fp = fopen("./weight.dat","ab+");
	
	//FILE *fp = fopen("./weightINQ.dat","ab+");
	if(fp == NULL){
		return 0;
	}
    // using data
    int a = extract<int>(x);
	char *datatmp = (char *)malloc(a);
	if(datatmp == NULL){
		return 0;
	}
	//printf("datatmp bais：\n");
	for (int i = 0; i < a; i++)
    {
        datatmp[i] = round(data[i]);
		//printf("%d\n",datatmp[i]);
		
    }
	fwrite(datatmp,a,1,fp);
	
	if(g_baise_padding_flag==1){
		char buff[2]={0};
		fwrite(buff,2,1,fp);
	}
	
	fclose(fp);
	
	free(datatmp);
	
    return 0;

}

int weight_read_func(object & data_obj, object x, object y, object z, object j)
{
    PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
    float * data = static_cast<float *>(PyArray_DATA(data_arr));
    // using data
    
    int a = extract<int>(x);
	int b = extract<int>(y);
	int c = extract<int>(z);
	int d = extract<int>(j);
	int len = a*b*c*d;
	
	for (int i = 0; i < len; i++)
    {
		data[i] = round(data[i]);
		if(round(data[i])==128)
		{
			data[i] = 127;
		}
		
    }
	
    return 0;

}


int bais_read_func(object & data_obj, object x)
{
    PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
    float * data = static_cast<float *>(PyArray_DATA(data_arr));
    // using data
    
    int a = extract<int>(x);
	int len = a;
	
	for (int i = 0; i < len; i++)
    {
		data[i] = round(data[i]);
		if(round(data[i])==128)
		{
			data[i] = 127;
		}
		
    }
	
    return 0;

}



BOOST_PYTHON_MODULE(compression_object)
{
	def("binary_bais_func", binary_bais_func);
	def("binary_Winograd3X3_func", binary_Winograd3X3_func);
	def("binary_Winograd1X1_func", binary_Winograd1X1_func);
	def("weight_read_func", weight_read_func);
	def("bais_read_func", bais_read_func);
}


