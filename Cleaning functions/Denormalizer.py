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

### Take the mean and standard deviation from original data
with open('../Data/all_recorded_data.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    data = list(reader)

header = np.array(data[0][:49])
data = np.array(data)

data = data[1:,:49]
data = data.astype(float)

mean = np.mean(data,axis=0)
std = np.std(data,axis=0)

## We only need
## robot_x robot_y theta
mean = mean[3:6]
std = std[3:6]

### Create new files
### following the original distribution
for i in range(8):
        with open('../DCGAN/Training/training_' + str(i) + '.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            data = np.array(list(reader))
            data = data.astype(float)
            data = data*std + mean
            #print(data)
        with open('../DCGAN/Generated Samples/'+ str(i) + '.csv','w',newline='') as newfile :
            writer = csv.writer(newfile,delimiter=',')    
            writer.writerow(['robot_x','robot_y','robot_theta'])
            for row in data :
                writer.writerow(row)