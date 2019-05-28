#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:48:24 2019

@author: aloisblarre
"""

import math
import csv
import numpy as np
     
with open('./all_recorded_data.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    data = list(reader)

print('Data loaded')
n = 0
header = np.array(data[0])
data = np.array(data)
data = data[1:,:44]
data = data.astype(float)
data -= np.mean(data,axis=0)
data /= np.std(data,axis=0)
print('Data normalized')

for i in range(1191) :
    with open('./data/'+str(i)+'.csv','w') as newfile :
        writer = csv.writer(newfile,delimiter=',')
        while data[n,0] > data[n+1,0] :
            writer.writerow(data[n])
            n += 1
        writer.writerow(data[n])
        n += 1
        print('File ./data/' + str(i) + '.csv created')

