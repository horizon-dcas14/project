#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:48:24 2019

@author: aloisblarre
@author: ftxsilva
"""

"""
This script was written to take the data from our autonomous dataset and break
it in several files containing 10 consecutive seconds of simulation.

Files that have less than 10s will be discarded.

"""

import os
import math
import csv
import numpy as np
     
with open('../Data/cleaned_normalized/Autonomous.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    data = list(reader)
    header = np.array(data[0])
    n_rows = sum(1 for line in data)

print('Data loaded')
n = 1

# list to store the files to be deleted
tobedel = []

for i in range(int(n_rows/10)) :
    with open('../Data/autonomous_cleaned_normalized/'+str(i)+'.csv','w',newline='') as newfile :
        writer = csv.writer(newfile,delimiter=',')
        writer.writerow(header)
        count = 0
        while count<10:
            writer.writerow(data[n])
            if(int(data[n][0])-int(data[n+1][0]) != 1 and count!=9):
                count = 10
                tobedel.insert(0,i)
            else:
                count+=1          
            n+=1
        print('File ../Data/autonomous_cleaned_normalized/' + str(i) + '.csv created')

#remove files
for j in tobedel:
    os.remove('../Data/autonomous_cleaned_normalized/'+str(j)+'.csv')
    print('File Removed!:' + str(j))