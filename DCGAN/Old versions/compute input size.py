#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:00:28 2019

@author: aloisblarre
"""

#%% Computes the output size of a convTranspose2d operation

H_in = 8
W_in = 1
stride = [1,1]
padding = [0,0]
dilatation = [1,1]
kernel_size = [3,3]
output_padding = [0,0]

H_out = (H_in - 1)*stride[0] - 2*padding[0] + dilatation[0]*(kernel_size[0] - 1) + output_padding[0] + 1
W_out = (W_in - 1)*stride[1] - 2*padding[1] + dilatation[1]*(kernel_size[1] - 1) + output_padding[1] + 1

print('H_out = ', H_out)
print('W_out = ', W_out)

#%% Computes the output size of a conv2d operation

H_in = 6
W_in = 1
stride = [1,1]
padding = [0,0]
dilatation = [1,1]
kernel_size = [6,1]

H_out = (H_in + 2*padding[0] - dilatation[0]*(kernel_size[0] - 1) - 1)/stride[0] + 1
W_out = (W_in + 2*padding[1] - dilatation[1]*(kernel_size[1] - 1) -1)/stride[1] + 1

print('H_out = ', H_out)
print('W_out = ', W_out)