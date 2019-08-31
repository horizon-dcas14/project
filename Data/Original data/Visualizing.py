# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:48:11 2019

@author: felipetxs
"""

import os
import math
import csv
import torch
import psutil ; #print(list(psutil.virtual_memory())[0:2])
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import random

######################################
###       Data analysis
######################################

random.seed(99)

######################################
###       AUTONOMOUS
######################################


### Open files
count = 0
data = []
for i in range(15130):
        try :
            with open('Autonomous data/'+ str(i) + '.csv', 'r') as file:
                ###  We will need only
                ### robot_x, robot_y, theta
                 data.append(pd.read_csv(file, usecols = ['robot_x','robot_y', 'robot_theta']))
                 #data.append(list(csv.reader(file, delimiter=',')))
        
        except FileNotFoundError :
            count += 1
            
### Files not found or non existant          
print('Files not found:', count)

### Take a look at the categories
### and first 3 lines
print(data[0].head(3))

### Make sure the values are in the correct format
print(data[0]['robot_x'].dtypes)
print(data[0]['robot_y'].dtypes)
print(data[0]['robot_theta'].dtypes)

### Let's plot some trajectories (n)
### to grasp what we are looking at
n = 10
x = random.sample(range(0,len(data)),n)
for i in range(10):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(data[x[i]]['robot_x'],data[x[i]]['robot_y'], 'r*-')
    j=1
    for xs,ys in zip(data[x[i]]['robot_x'], data[x[i]]['robot_y']):
        j += 1
        plt.text(xs, ys, '%d' % (j))
    xlabel('x')
    ylabel('y')
    title('Trajectorie %d' %(x[i]))
    plt.savefig('Trajectories/Aut_%d.png' %(i))        
    show()

######################################
###       HUMAN
######################################


### Open files
count = 0
data = []
for i in range(15130):
        try :
            with open('Human data/'+ str(i) + '.csv', 'r') as file:
                ###  We will need only
                ### robot_x, robot_y, theta
                 data.append(pd.read_csv(file, usecols = ['robot_x','robot_y', 'robot_theta']))
                 #data.append(list(csv.reader(file, delimiter=',')))
        
        except FileNotFoundError :
            count += 1
            
### Files not found or non existant          
print('Files not found:', count)

### Take a look at the categories
### and first 3 lines
print(data[0].head(3))

### Make sure the values are in the correct format
print(data[0]['robot_x'].dtypes)
print(data[0]['robot_y'].dtypes)
print(data[0]['robot_theta'].dtypes)

### Let's plot some trajectories (n)
### to grasp what we are looking at
n = 10
x = random.sample(range(0,len(data)),n)
for i in range(10):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(data[x[i]]['robot_x'],data[x[i]]['robot_y'], 'r*-')
    j=1
    for xs,ys in zip(data[x[i]]['robot_x'], data[x[i]]['robot_y']):
        j += 1
        plt.text(xs, ys, '%d' % (j))
    xlabel('x')
    ylabel('y')
    title('Trajectorie %d' %(x[i]))
    plt.savefig('Trajectories/Hum_%d.png' %(i))         
    show()

