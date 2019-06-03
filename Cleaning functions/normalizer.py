#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:25:55 2019

@author: aloisblarre
"""
import math
import csv
import numpy as np
     
with open('../Data/all_recorded_data.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    data = list(reader)

header = np.array(data[0])
data = np.array(data)
data = data[1:][:44]
data = data.astype(float)
data -= np.mean(data,axis=0)
data /= np.std(data,axis=0)
    
with open('../Data/all_recorded_data_normalized.csv','w',newline='') as newfile :
    writer = csv.writer(newfile,delimiter=',')
    writer.writerow(header)
    for row in data :
        writer.writerow(row)