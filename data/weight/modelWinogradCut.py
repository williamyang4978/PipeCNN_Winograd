# ---------------------------------------------------------------------------
#  
#  PipeCNN_Winograd: PipeCNN accelerated by Winograd
#  
# --------------------------------------------------------------------------
#  
#  Filename:
#     - modelWinogradCut.py
#  
#  Author(s):
#     -Anrong Yang, yangar@lenovo.com
#  
#  History:
#     - v1.0 initial version. 
# 
#  
# --------------------------------------------------------------------------
#  
#  Copyright (C) 2019 Lenovo
#  Licensed under BSD-3, see COPYING.BSD file for details.
#  
# --------------------------------------------------------------------------
#  
# This program implements 8-bit quantization.
#
#
#
#  
# --------------------------------------------------------------------------





import argparse
import compression_object
import numpy as np
import math
import pdb

import sys
#caffe_root = '/home/caffe-yolo9000/'
#sys.path.insert(0, caffe_root + 'python')
import caffe


#vector size
vectorSize = 8

#line number
lineNumber = 32.0

G = np.array([[1.0/4,0,0],[-1.0/6,-1.0/6,-1.0/6],[-1.0/6,1.0/6,-1.0/6],[1.0/24,1.0/12,1.0/6],[1.0/24,-1.0/12,1.0/6],[0,0,1]],dtype='float32')
sumber = 0
sum1 = 0
listSum =[]
listSum1 =[]

def winogard_weight_bais_fuc(weight,bais,layer_name):
    print layer_name + ':'
    shape = weight.shape
    weightData = np.zeros((shape[0] * shape[1] * shape[2] * 6), dtype='float32')
    weightData = weightData.reshape(shape[0], shape[1], shape[2], 6)

    loopNum = range(shape[0])
    loopW = range(shape[1])
    global sumber,sum1
    if shape[2] != 1 and shape[3] != 1:
        for i in loopNum:
            for n in loopW:
                weightZ = weight[i, n, :, :]
                weightT = np.transpose(weightZ)
                dataDot = np.dot(G, weightT)
                dataT = np.transpose(dataDot)
                weightData[i, n, :, :] = dataT
    else:
        weightData = weight

    shape = weightData.shape
    padding = 0

    listSum.append(shape[1] * shape[2] * 6)
    sumber = sumber + shape[0] * shape[1] * shape[2] * 6

    maxtmp = np.max(weightData)
    mintmp = np.min(weightData)
    if maxtmp < abs(mintmp):
        maxtmp = abs(mintmp)
    if maxtmp > 0:
        m_param = 7 - math.ceil(math.log(maxtmp, 2))
        weightData = weightData * (2 ** m_param)
    else:
        m_param = 7
    #pdb.set_trace()
    if shape[2] != 1 and shape[3] != 1:
        compression_object.binary_Winograd3X3_func(weightData, shape[0], shape[1], shape[2], shape[3],
                                                          vectorSize, padding)
    else:
        #if layer_name == "conv_reg":
        #    padding = 1
        compression_object.binary_Winograd1X1_func(weightData, shape[0], shape[1], shape[2], shape[3],
                                                          vectorSize, padding)

    maxtmp = np.max(bais)
    mintmp = np.min(bais)

    if maxtmp < abs(mintmp):
        maxtmp = abs(mintmp)
    if maxtmp > 0:
        b_param = 7 - math.ceil(math.log(maxtmp, 2))
        bais = bais * (2 ** b_param)
    else:
        b_param = 7
    shape = bais.shape

    sum1 = sum1 + shape[0]
    listSum1.append(shape[0])
    compression_object.binary_bais_func(bais, shape[0])
    print "weight:", m_param, "bais:", b_param


def caffemodel_deal_func(model_weights, model_def):

    net = caffe.Net(model_def, model_weights, caffe.TEST)
    for layer_name, param in net.params.iteritems():
        try:
            name = layer_name.split("-")
            strname = 'conv_reg'

            if len(name)==2 and name[1]=='conv':
                name = name[0]
                weight = net.params[name + '-conv'][0].data
                u = net.params[name + '-bn'][0].data
                var = net.params[name + '-bn'][1].data
                e = net.params[name + '-bn'][2].data
                y = net.params[name + '-scale'][0].data
                b = net.params[name + '-scale'][1].data

                shape = weight.shape

                shape0 = int(lineNumber*math.ceil(float(shape[0])/lineNumber))
                shape1 = int(vectorSize * math.ceil(float(shape[1]) / vectorSize))

                bycle = range(shape[0])
                for i in bycle:
                    tmp = y[i]/math.sqrt(var[i]/e[0]+1e-5)
                    weight[i,:,:,:] = weight[i,:,:,:]*tmp
                    b[i] = (-u[i]*y[i]/e[0])/math.sqrt(var[i]/e[0]+1e-5)+b[i]
                if (layer_name == "layer1-conv"):
                    data = np.zeros((shape0 * shape1 * shape[2] * shape[3]), dtype='float32')
                    data1 = np.zeros(shape0, dtype='float32')
                    data = data.reshape(shape0, shape1, shape[2], shape[3])
                    shape = weight.shape
                    loopNum1 = range(shape[0])
                    loopNum2 = range(shape[1])

                    for i in loopNum1:
                        data1[i] = b[i]
                        for j in loopNum2:
                            data[i, j, :, :] = weight[i, j, :, :]

                    weight = data
                    b =data1
                else:
                    data = np.zeros((shape0 * shape1 * shape[2] * shape[3]), dtype='float32')
                    data1 = np.zeros(shape0, dtype='float32')
                    data = data.reshape(shape0, shape1, shape[2], shape[3])
                    shape = weight.shape
                    loopNum1 = range(shape[0])
                    loopNum2 = range(shape[1])

                    for i in loopNum1:
                        data1[i] = b[i]
                        for j in loopNum2:
                            data[i, j, :, :] = weight[i, j, :, :]


                    weight = data
                    b = data1

                winogard_weight_bais_fuc(weight, b, layer_name)

            elif layer_name == strname:
                weight = net.params[layer_name][0].data
                b = net.params[layer_name][1].data

                shape = weight.shape
                shape0 = int(lineNumber * math.ceil(float(shape[0]) / lineNumber))
                shapeBais = int(56 * math.ceil(float(shape[0]) / 56.0))

                shape1 = int(vectorSize * math.ceil(float(shape[1]) / vectorSize))

                data = np.zeros((shape0 * shape1 * shape[2] * shape[3]), dtype='float32')
                data1 = np.zeros(shape0, dtype='float32')
                data = data.reshape(shape0, shape1, shape[2], shape[3])
                shape = weight.shape
                loopNum1 = range(shape[0])

                for i in loopNum1:
                    data1[i] = b[i]
                    data[i, :, :, :] = weight[i, :, :, :]

                weight = data
                b = data1

                winogard_weight_bais_fuc(weight, b, layer_name)

        except IndexError:
            continue
        finally:
            print ' '

    print sumber,sum1,sumber+sum1
    print listSum
    print listSum1


parser = argparse.ArgumentParser()
parser.add_argument('caffemodel',help='the path of caffe model')
parser.add_argument('prototxt',help='the path of caffe model prototxt')


args = parser.parse_args()

caffemodel_deal_func(args.caffemodel,args.prototxt)


