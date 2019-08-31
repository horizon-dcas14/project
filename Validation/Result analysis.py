# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:42:19 2019

@author: ftxsilva
"""

import os
import math
import csv
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import random

#%%
## DESTANDARDIZATION OF THE DATA:
## we must have the results from the dcgan 
## accordingly to the original distribution

### Take the mean and standard deviation from original data
with open('../Data/all_recorded_data.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    data = list(reader)

header = np.array(data[0][:49])
data = np.array(data)

data = data[1:,:49]
data = data.astype(float)

## mean and standard deviation
## used to standardized the data
mean = np.mean(data,axis=0)
std = np.std(data,axis=0)

## We only need
## robot_x robot_y theta
mean = mean[3:6]
std = std[3:6]

### Create new files
### following the original distribution
for i in range(50):
    try:
        with open('Last Epoch/DCGAN' + str(i) + '.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            data = np.array(list(reader))
            data = data.astype(float)
            data = data*std + mean
        with open('Trajectories/'+ str(i) + '.csv','w',newline='') as newfile :
            writer = csv.writer(newfile,delimiter=',')    
            writer.writerow(['robot_x','robot_y','robot_theta'])
            for row in data :
                writer.writerow(row)
        ### PLOTTING                
        ### Drawing the environment
        fig=plt.figure()
        treesX = np.array([4.55076, -0.7353, -15.10146, -2.6019, -1.33158, 16.58292, 16.87086, 0.6078, -16.65378])
        treesY = [14.66826,14.75052,15.76476,6.7425,10.02042,-12.5847,-16.01952,-16.23906,-16.23906]
        batteryX = [0.0]
        batteryY = [-10.0]
        waterX = [16.0]
        waterY = [16.29]
        plt.plot(treesX, treesY, 'g^')
        plt.plot(batteryX, batteryY, 'rs')
        plt.plot(waterX, waterY, 'bp')
        
        ### Let's plot the trajectories generated
        plt.plot(data.transpose((1,0))[0],data.transpose((1,0))[1], 'm*-')
        j=0
        for xs,ys in zip(data.transpose((1,0))[0], data.transpose((1,0))[1]):
            j += 1
            plt.text(xs, ys, '%d' % (j))
        xlabel('x')
        ylabel('y')
        title('Trajectorie %d' %(i))    
        plt.savefig('Trajectories/Trajectoire_%d.png' %(i))   
        show()
        
    except FileNotFoundError :
            print('FILE NOT FOUND')
    
#%%
### RUN NEW SAMPLES AFTER TRAINING
## generate fake samples
noise = torch.randn(b_size, nz, 1, 1, device=device)
fake = netG(noise)

## draw the trajectories
for i in range(len(fake)):
    data1=np.array(fake[i].detach().numpy())
    data1 = data1.astype(float)
    data1=data1*std+mean
    fig=plt.figure()
    ## write files for further analysis
    with open('New results/DCGAN_' + str(i) + '.csv', 'w', newline = '') as alldata :
            writer = csv.writer(alldata,delimiter=',')
            j=0;
            for j in range(10):
                writer.writerow([data1[0][j][0],data1[0][j][1],data1[0][j][2]])
                j+=1
    ### Drawing the environment
    fig=plt.figure()
    treesX = np.array([4.55076, -0.7353, -15.10146, -2.6019, -1.33158, 16.58292, 16.87086, 0.6078, -16.65378])
    treesY = [14.66826,14.75052,15.76476,6.7425,10.02042,-12.5847,-16.01952,-16.23906,-16.23906]
    batteryX = [0.0]
    batteryY = [-10.0]
    waterX = [16.0]
    waterY = [16.29]
    plt.plot(treesX, treesY, 'g^')
    plt.plot(batteryX, batteryY, 'rs')
    plt.plot(waterX, waterY, 'bp')
    
    ### Plot results
    plt.plot(data1[0].transpose((1,0))[0],data1[0].transpose((1,0))[1], 'r*-')
    j=0
    for xs,ys in zip(data1[0].transpose((1,0))[0], data1[0].transpose((1,0))[1]):
        j += 1
        plt.text(xs, ys, '%d' % (j))
    xlabel('x')
    ylabel('y')
    title('Trajectorie %d' %(i))    
    plt.savefig('New results/Trajectoire_%d.png' %(i))   
    show()