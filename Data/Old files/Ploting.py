# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:53:19 2019

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

### Open files
count = 0
data = []
for i in range(10):
        try :
            with open('../DCGAN/Generated Samples/'+ str(i) + '.csv', 'r') as file:
                ###  We will need only
                ### robot_x, robot_y, theta
                data.append(pd.read_csv(file, usecols = ['robot_x','robot_y', 'robot_theta']))          
        except FileNotFoundError :
            print('FILE NOT FOUND')
### Let's plot the trajectories generated
i=0
for i in range(len(data)):
    fig=plt.figure()
    plt.plot(data[i]['robot_x'],data[i]['robot_y'], 'r*-')
    j=1
    for xs,ys in zip(data[i]['robot_x'], data[i]['robot_y']):
        j += 1
        plt.text(xs, ys, '%d' % (j))
    xlabel('x')
    ylabel('y')
    title('Trajectorie %d' %(i))    
    plt.savefig('Trajectories/Trajectoire_%d.png' %(i))   
    show()
                 