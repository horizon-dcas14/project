# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:01:14 2019

@author: ftxsilva
"""

"""
This script was written to split our data into two categories: 
    1) autonomous mode
    2) manual mode
    
MUST BE RUN AFTER datacleaningtoall.py
    
"""

import math
import csv
import numpy as np

#import current data
#all assembled in one sole file

with open('../Data/all_recorded_data_normalized.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    data = list(reader)
    header = np.array(data[0])
    data = np.array(data)

#create new files
with open('../Data/cleaned_normalized/Autonomous.csv','w',newline='') as newfile1 :
        writer1 = csv.writer(newfile1,delimiter=',')
        writer1.writerow(header)

        with open('../Data/cleaned_normalized/Human.csv','w', newline='') as newfile2 :
                writer2 = csv.writer(newfile2,delimiter=',')
                writer2.writerow(header)
        
                #split data
                n = 1
                while n < len(data):
                    if float(data[n][1]) < 0 :
                        writer2.writerow(data[n])
                    elif float(data[n][1]) > 1:
                        writer1.writerow(data[n])
                    n += 1
 