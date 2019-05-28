#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:25:55 2019

@author: aloisblarre
"""
import math
import csv

def direction(string) :
    string = string [:7]
    if '-1' in string or 'other' in string :
        return -1
    if 'back' in string :
        return 0
    if 'left' in string :
        return 1
    if 'front' in string :
        return 2
    if 'right' in string :
        return 3
    if 'spac' in string :
        return 4
    print('Direction error : ' + string)
    return 'error'

c = 0

for i in range(9978):
    try :
        with open('../recorded_csv_data2/FRGrecord_' + str(i) + '.csv', 'r') as file:
             data = csv.reader(file, delimiter=',')
             with open('./cleaned_data/FRGrecord' + str(c) + '_cleaned.csv', 'w') as cleaned_file:
                 cleaned_data = csv.writer(cleaned_file, delimiter=',')
                 line_count = 0
                 for row in data:
                     if line_count == 0 :
                         cleaned_data.writerow(row)
                     else :
                        cleaned_data.writerow([
                                float(row[0])/60-0.5,
                                row[1],
                                row[2],
                                float(row[3])/40,
                                float(row[4])/40,
                                float(row[5])/(2*math.pi) + 0.5,
                                row[6],
                                float(row[7])/100,
                                float(row[8])/220 + 0.5,
                                float(row[9])/100 + 0.5,
                                float(row[10])/100 + 0.5,
                                row[11],
                                direction(row[12]),
                                row[13],
                                row[14],
                                row[15]
                                ])
                     line_count += 1
        c += 1
    except FileNotFoundError :
        print('File ' + '../recorded_csv_data2/FRGrecord_' + str(i) + '.csv' + ' was not found')
             
     
