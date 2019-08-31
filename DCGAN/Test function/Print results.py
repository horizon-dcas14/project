# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:20:26 2019

@author: felip
"""
import csv
import numpy as np


for i in range(len(csv_list)):
    with open('Training/' + str(i) + '.csv', 'w', newline = '') as alldata :
        writer = csv.writer(alldata,delimiter=',')
        j=0;
        array = csv_list[i].numpy()
        for j in range(10):
            writer.writerow([array[0][0][j][0],array[0][0][j][1],array[0][0][j][2]])
            j+=1
            

for i in range(len(fake_samples)):
    with open('Results/' + str(i) + '.csv', 'w', newline = '') as alldata :
        writer = csv.writer(alldata,delimiter=',')
        j=0;
        array = fake_samples[i].numpy()
        for j in range(10):
            writer.writerow([array[0][0][j][0],array[0][0][j][1],array[0][0][j][2]])
            j+=1
# =============================================================================
#             
# plt.figure(figsize=(10,5))
# plt.title("Generator and Discriminator Loss During Training Lr=0.002 beta1=0.5")
# plt.plot(G_losses,label="G")
# plt.plot(D_losses,label="D")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig('losses1.png')
# plt.show()
# =============================================================================
