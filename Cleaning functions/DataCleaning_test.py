#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:36:56 2019

@author: ftxsilva
"""

import os
import math
import csv
import torch
import numpy as np

# =============================================================================
# Cleaning functions
# =============================================================================

def direction(old,new) :
    old = old.astype(float)
    new = new.astype(float)
    return new[2] - old[2]

def avancement(old,new) :
    old = old.astype(float)
    new = new.astype(float)
    return math.sqrt(pow(old[0]-new[0],2)+pow(old[1]-new[1],2))

def keys(string) :
    activations = np.zeros(5)
    if string == '-1' :
        return activations
    while (string != '') :
        if 'left' in string[:4] : 
            string = string[4:]
            activations[0] += 1
        elif 'right' in string[:5] :
            string = string[5:]
            activations[1] += 1
        elif 'front' in string[:5] :
            string = string[5:]
            activations[2] += 1
        elif 'back' in string[:4] :
            string = string[4:]
            activations[3] += 1
        elif 'spac' in string[:4] :
            string = string[4:]
            activations[4] += 1
        else :
            string = string[1:]
    return activations

def clicks(string) :
    activations = np.zeros(14)
    if string == '-1' :
        return activations
    while (string != '') :
        if 'left' in string[:4] : 
            string = string[4:]
            activations[0] += 1
        elif 'right' in string[:5] :
            string = string[5:]
            activations[1] += 1
        elif 'push' in string[:4] :
            string = string[4:]
            activations[2] += 1
        elif 'wrench' in string[:6] :
            string = string[6:]
            activations[3] += 1
        elif 'leak_1' in string[:6] :
            string = string[6:]
            activations[4] += 1
        elif 'leak_2' in string[:6] :
            string = string[6:]
            activations[5] += 1
        elif 'leak_3' in string[:6] :
            string = string[6:]
            activations[6] += 1
        elif 'leak_4' in string[:6] :
            string = string[6:]
            activations[7] += 1
        elif 'leak_5' in string[:6] :
            string = string[6:]
            activations[8] += 1
        elif 'leak_6' in string[:6] :
            string = string[6:]
            activations[9] += 1
        elif 'leak_7' in string[:6] :
            string = string[6:]
            activations[10] += 1
        elif 'leak_8' in string[:6] :
            string = string[6:]
            activations[11] += 1
        elif 'leak_9' in string[:6] :
            string = string[6:]
            activations[12] += 1
        elif 'rm_alarm' in string[:8] :
            string = string[8:]
            activations[13] += 1
        else :
            string = string[1:]
    return activations

def binaryToData(state,n):
    state = str(state)
    if 9 - n >= len(state) :
        return 0
    else :
        return int(state[n-10+len(state)])

count = 0

# =============================================================================
# Regrouping data into one file
# =============================================================================

with open('../Data/all_recorded_data.csv', 'w', newline = '') as alldata :
    writer = csv.writer(alldata,delimiter=',')
    for i in range(9978): #9978
        try :
            with open('../Data/recorded_csv_data2/FRGrecord_' + str(i) + '.csv', 'r') as file:
                 data = csv.reader(file, delimiter=',')
                 line_count = 0
                 for row in data :
                     if line_count == 0 :
                         if i == 7 :
                             writer.writerow(row[:6] + 
                                             ['tree 1', 'tree 2','tree 3', 'tree 4','tree 5', 'tree 6','tree 7', 'tree 8', 'tree 9'] + 
                                             row[7:11] +
                                             ['leak 1', 'leak 2','leak 3', 'leak 4','leak 5', 'leak 6','leak 7', 'leak 8', 'leak 9'] +
                                             ['direction','avancement'] + 
                                             ['left', 'right', 'front', 'back', 'space'] +
                                             ['left', 'right', 'push', 'wrench', 'leak 1', 'leak 2','leak 3', 'leak 4','leak 5', 'leak 6','leak 7', 'leak 8', 'leak 9', 'rm_alarm'] + 
                                             row[14:])
                     elif line_count == 1 :
                         writer.writerow(np.concatenate(([
                                row[0],
                                row[1],
                                row[2],
                                row[3],
                                row[4],
                                row[5],
                                binaryToData(row[6],1),
                                binaryToData(row[6],2),
                                binaryToData(row[6],3),
                                binaryToData(row[6],4),
                                binaryToData(row[6],5),
                                binaryToData(row[6],6),
                                binaryToData(row[6],7),
                                binaryToData(row[6],8),
                                binaryToData(row[6],9),
                                row[7],
                                row[8],
                                row[9],
                                row[10],
                                binaryToData(row[11],1),
                                binaryToData(row[11],2),
                                binaryToData(row[11],3),
                                binaryToData(row[11],4),
                                binaryToData(row[11],5),
                                binaryToData(row[11],6),
                                binaryToData(row[11],7),
                                binaryToData(row[11],8),
                                binaryToData(row[11],9),
                                0,
                                0],
                                keys(row[12]),
                                clicks(row[13]),
                                [row[14],
                                row[15]
                                ])))
                     else :
                         writer.writerow(np.concatenate(([
                                row[0],
                                row[1],
                                row[2],
                                row[3],
                                row[4],
                                row[5],
                                binaryToData(row[6],1),
                                binaryToData(row[6],2),
                                binaryToData(row[6],3),
                                binaryToData(row[6],4),
                                binaryToData(row[6],5),
                                binaryToData(row[6],6),
                                binaryToData(row[6],7),
                                binaryToData(row[6],8),
                                binaryToData(row[6],9),
                                row[7],
                                row[8],
                                row[9],
                                row[10],
                                binaryToData(row[11],1),
                                binaryToData(row[11],2),
                                binaryToData(row[11],3),
                                binaryToData(row[11],4),
                                binaryToData(row[11],5),
                                binaryToData(row[11],6),
                                binaryToData(row[11],7),
                                binaryToData(row[11],8),
                                binaryToData(row[11],9),
                                direction(old_state,np.array(row[3:6])),
                                avancement(old_state,np.array(row[3:6]))],
                                keys(row[12]),
                                clicks(row[13]),
                                [row[14],
                                row[15]
                                ])))
                     old_state = np.array(row[3:6])
                     line_count += 1
        except FileNotFoundError :
            count += 1

print('Data collected into one file')

# =============================================================================
# Splitting data based en robot mode
# =============================================================================

#import current data
#all assembled in one sole file

with open('../Data/all_recorded_data.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    data = list(reader)
    header = np.array(data[0])
    data = np.array(data)

#create new files
with open('../Data/Autonomous_non_normalized.csv','w',newline='') as newfile1 :
        writer1 = csv.writer(newfile1,delimiter=',')
        writer1.writerow(header)

        with open('../Data/Human_non_normalized.csv','w', newline='') as newfile2 :
                writer2 = csv.writer(newfile2,delimiter=',')
                writer2.writerow(header)
        
                #split data
                n = 1
                while n < len(data):
                    if float(data[n][1]) == 0 :
                        writer2.writerow(data[n])
                    elif float(data[n][1]) == 1:
                        writer1.writerow(data[n])
                    n += 1

print('Files splitted based on robot mode')

# =============================================================================
# Breaking down Autonomous data in 10 second sequences
# =============================================================================

with open('../Data/Autonomous_non_normalized.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    data = list(reader)
    header = np.array(data[0])
    n_rows = sum(1 for line in data)

print('Autonomous data loaded')
n = 1

# list to store the files to be deleted
tobedel = []

for i in range(int(n_rows/10)) :
    with open('../Data/original data/Autonomous data/'+str(i)+'.csv','w',newline='') as newfile :
        writer = csv.writer(newfile,delimiter=',')
        writer.writerow(header)
        count = 0
        while count<10:
            writer.writerow(data[n])
            if(float(data[n][0])-float(data[n+1][0]) < 0 and count!=9):
                count = 10
                tobedel.insert(0,i)
            else:
                count+=1          
            n+=1

#remove files
for j in tobedel:
    os.remove('../Data/original data/Autonomous data/'+str(j)+'.csv')
print('All normalized breakdowned autonomous files treated')

with open('../Data/Human_non_normalized.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    data = list(reader)
    header = np.array(data[0])
    n_rows = sum(1 for line in data)

print('Human data loaded')
n = 1

# =============================================================================
# Breaking down Human data in 10 second sequences
# =============================================================================

# list to store the files to be deleted
tobedel = []

for i in range(int(n_rows/10)) :
    with open('../Data/original data/Human data/'+str(i)+'.csv','w',newline='') as newfile :
        writer = csv.writer(newfile,delimiter=',')
        writer.writerow(header)
        count = 0
        while count<10:
            writer.writerow(data[n])
            if(float(data[n][0])-float(data[n+1][0]) < 0 and count!=9):
                count = 10
                tobedel.insert(0,i)
            else:
                count+=1          
            n+=1

#remove files
for j in tobedel:
    os.remove('../Data/original data/Human data/'+str(j)+'.csv')

print('All normalized breakdowned human files treated')

# =============================================================================
# Removing temporary files
# =============================================================================

os.remove('../Data/all_recorded_data.csv')
os.remove('../Data/all_recorded_data_normalized.csv')
os.remove('../Data/Autonomous_non_normalized.csv')
os.remove('../Data/Human_non_normalized.csv')