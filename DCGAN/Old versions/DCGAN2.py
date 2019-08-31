# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:16:14 2019
@author: ftxsilva
"""

#%% Importing modules
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import time
import math
import tkinter

#%% Initializing parameters
# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "../Data/Autonomous data/"
# Number of workers for dataloader
workers = 0
# Batch size during training
batch_size = 128 # 128
# Number of channels in the training data.
nc = 1
# Number of random parameters in the generator's input
nr = 50
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 15
# Learning rate for optimizers
lr = 0.0001 # 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
#Number of variables representing th situation
input_variables = 3
# Size of z latent vector (i.e. size of generator input)
nz = nr

"""
Create the appropriate dataset class format for our problem
"""
#%% Dataset creation
def get_data(dataroot):
    df = pd.read_csv(dataroot, usecols = ['robot_x','robot_y', 'robot_theta'])
    df = df.values
    df = torch.DoubleTensor([df])
    return df

data_sets = dset.DatasetFolder(dataroot, 
                                   loader=get_data, extensions=['.csv'])
dataloader = torch.utils.data.DataLoader(data_sets, batch_size, shuffle = True, num_workers = workers)
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#%% weights initialization
# further used in the generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        m.weight.data = m.weight.data.float()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        m.weight.data = m.weight.data.float()
        m.bias.data = m.bias.data.float()
        

#%% Create the generator
# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 4, (6,1), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 6 x 1
            nn.ConvTranspose2d(ngf * 4, ngf * 2, (3,1), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 1
            nn.ConvTranspose2d( ngf * 2, 1, (3,3), 1, 0, bias=False),
            nn.Tanh()
            # state size. (1) x 10 x 18
        )
    def forward(self, input):
            return self.main(input)
        
# Create the generator
netG = Generator(ngpu).to(device)
netG = netG.float()
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

#%% Create the discriminator
# Discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc=1) x 10 x 18
            nn.Conv2d(nc, ndf, (3,input_variables), 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 8 x 1
            nn.Conv2d(ndf, ndf * 2, (3,1), 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 6 x 1
            nn.Conv2d(ndf * 2, 1, (6,1), 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
            return self.main(input)
        
# Create the Discriminator
netD = Discriminator(ngpu).to(device)
netD = netD.float()
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)


#%% Loss function and optimizer initialization
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#%% Training the DCGAN
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
netD = netD.float()
netG = netG.float()

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    print('Epoch : ', epoch)
    for i, data in enumerate(dataloader, 0):
        #data = data
        #for j in range(128):
        data[0] = data[0].float()
        #Extract the data representing the situations from data 
        situations = torch.split(data[0],input_variables,3)[0]
        #print(np.shape(situations))
       
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nr, 1, 1, device=device)
        #Reqhape the situation vector to match the random vector
        situations = torch.reshape(situations,(b_size,10*input_variables,1,1))
        #Concatenate the situation vector with the random vector
        noise = torch.cat((noise,situations),1)

        # Generate fake image batch with G
        fake = netG(noise)
        fake = torch.cat((torch.split(data[0],input_variables,3)[0],fake),3)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%% Génération de nouvelles parties

for i, data in enumerate(dataloader, 0):
    data[0] = data[0].float()
    situations = torch.split(data[0],input_variables,3)[0]
    real_cpu = data[0].to(device)
    b_size = real_cpu.size(0)

    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(b_size, nr, 1, 1, device=device)
    situations = torch.reshape(situations,(b_size,10*input_variables,1,1))
    noise = torch.cat((noise,situations),1)
    fake = netG(noise)
    fake = torch.cat((torch.split(data[0],input_variables,3)[0],fake),3)
    break

fake2 = np.array(fake.detach())

#%% Affichage des résultats

sf = 6                                                            ### Définition des fonctions ###

mean = [ 3.87047857e+02,  4.86219043e-01, -8.95424774e-01,  2.83569667e+00,
        1.18524379e+00, -8.96262946e-02,  4.81105819e-01,  4.93504824e-01,
        6.19885973e-01,  4.91782204e-01,  4.81375782e-01,  5.83476349e-01,
        6.03074362e-01,  5.20195787e-01,  6.73261472e-01,  5.90625510e+01,
        2.71148088e+01,  3.22042461e+01,  2.84388398e+01,  2.69406789e-01,
        2.74764265e-01,  2.71264390e-01,  2.68690102e-01,  2.71952152e-01,
        2.84267597e-01,  2.84354371e-01,  2.72549927e-01,  2.85855236e-01,
        8.35345857e-04,  6.34343741e-01,  4.95883068e-01,  3.64838633e-01,
        6.69944786e-01,  8.46204773e-02,  1.89166779e-02,  1.21017245e-01,
        1.21451114e-01,  2.33122505e-01,  8.37945198e-02,  7.82570688e-03,
        8.03782050e-03,  7.61037943e-03,  8.04746203e-03,  8.33992171e-03,
        7.64251785e-03,  7.99925439e-03,  7.56217179e-03,  7.43040424e-03,
        1.98422646e-02]

std =[ 1.64473653e+02, 4.99810049e-01, 7.66177769e-01, 7.58661770e+00,
       1.09672450e+01, 1.76889881e+00, 4.99642882e-01, 4.99957811e-01,
       4.85414620e-01, 4.99932463e-01, 4.99653018e-01, 4.92982453e-01,
       4.89260336e-01, 4.99591964e-01, 4.69020748e-01, 2.81293167e+01,
       2.51941501e+01, 2.70581870e+01, 3.21741079e+01, 4.43651632e-01,
       4.46395412e-01, 4.44612214e-01, 4.43278390e-01, 4.44965368e-01,
       4.51064885e-01, 4.51106376e-01, 4.45271226e-01, 4.51820783e-01,
       8.34338279e-01, 8.65259623e-01, 2.75119243e+00, 2.36601477e+00,
       2.79843203e+00, 1.05756455e+00, 1.70870291e-01, 4.45785884e-01,
       4.51866149e-01, 9.66967280e-01, 4.60258569e-01, 9.32204478e-02,
       9.34778503e-02, 9.27024626e-02, 9.36659259e-02, 9.46134349e-02,
       9.10910690e-02, 9.51846167e-02, 9.11154121e-02, 8.95071739e-02,
       1.39803306e-01]
    
def clavier(event):    
    '''
    Entrée :
        Un event
    Sortie :
        Affichage de la figure correspondant à l'event
    '''
    touche=event.keysym             
    if touche == "Left":          
        extract.set(extract.get()-1) 
        canvas.delete(tkinter.ALL)
        print_trees(fake2[extract.get()][0])
        print_positions(fake2[extract.get()][0])
        print_lines(fake2[extract.get()][0])                  
    elif touche == "Right":         
        extract.set(extract.get()+1)  
        canvas.delete(tkinter.ALL)
        print_trees(fake2[extract.get()][0])
        print_positions(fake2[extract.get()][0])
        print_lines(fake2[extract.get()][0])                 
    else :                          
        return                     

def tdmu(coord) : #to_decent_mesuring_unit
    return (coord+20)*2.5*sf

def treat_data(array) :
    array[:,0,:,0] *= std[3]
    array[:,0,:,0] += mean[3] + 20
    array[:,0,:,0] *= 2.5 * sf 
    array[:,0,:,1] *= std[4]
    array[:,0,:,1] += mean[4] + 20
    array[:,0,:,1] *= 2.5 * sf 
    array[:,0,:,2] *= std[5]
    array[:,0,:,2] += mean[5]
    array[:,0,:,16] *= std[28]
    array[:,0,:,16] += mean[28]
    array[:,0,:,16] += array[:,0,:,2]
    array[:,0,:,17] *= std[29]
    array[:,0,:,17] += mean[29]
    array[:,0,:,17] *= 2.5 * sf 

def tree_state(val):
    if val > 0 :
        return '#fff444444'
    else :
        return '#000fff000'
        
def print_trees(array) :
    tree1 = canvas.create_oval(tdmu(4.55076)-2*sf,100*sf-(tdmu(14.66826)+2*sf),tdmu(4.55076)+2*sf,100*sf-(tdmu(14.66826)-2*sf),fill=tree_state(array[0,3]))
    tree2 = canvas.create_oval(tdmu(-0.7353)-2*sf,100*sf-(tdmu(14.75052)+2*sf),tdmu(-0.7353)+2*sf,100*sf-(tdmu(14.75052)-2*sf),fill=tree_state(array[0,4]))
    tree3 = canvas.create_oval(tdmu(-15.10146)-2*sf,100*sf-(tdmu(15.76476)+2*sf),tdmu(-15.10146)+2*sf,100*sf-(tdmu(15.76476)-2*sf),fill=tree_state(array[0,5]))
    tree4 = canvas.create_oval(tdmu(-2.6019)-2*sf,100*sf-(tdmu(6.7425)+2*sf),tdmu(-2.6019)+2*sf,100*sf-(tdmu(6.7425)-2*sf),fill=tree_state(array[0,6]))
    tree5 = canvas.create_oval(tdmu(-1.33158)-2*sf,100*sf-(tdmu(10.02042)+2*sf),tdmu(-1.33158)+2*sf,100*sf-(tdmu(10.02042)-2*sf),fill=tree_state(array[0,7]))
    tree6 = canvas.create_oval(tdmu(16.58292)-2*sf,100*sf-(tdmu(-12.5847)+2*sf),tdmu(16.58292)+2*sf,100*sf-(tdmu(-12.5847)-2*sf),fill=tree_state(array[0,8]))
    tree7 = canvas.create_oval(tdmu(16.87086)-2*sf,100*sf-(tdmu(-16.01952)+2*sf),tdmu(16.87086)+2*sf,100*sf-(tdmu(-16.01952)-2*sf),fill=tree_state(array[0,9]))
    tree8 = canvas.create_oval(tdmu(0.6078)-2*sf,100*sf-(tdmu(-16.23906)+2*sf),tdmu(0.6078)+2*sf,100*sf-(tdmu(-16.23906)-2*sf),fill=tree_state(array[0,10]))
    tree9 = canvas.create_oval(tdmu(-16.65378)-2*sf,100*sf-(tdmu(-16.23906)+2*sf),tdmu(-16.65378)+2*sf,100*sf-(tdmu(-16.23906)-2*sf),fill=tree_state(array[0,11]))
    red_square = canvas.create_rectangle(tdmu(0)-2*sf,100*sf-(tdmu(-10)+2*sf),tdmu(0)+2*sf,100*sf-(tdmu(-10)-2*sf),fill='red')
    blue_square = canvas.create_rectangle(tdmu(16)-2*sf,100*sf-(tdmu(16.29)+2*sf),tdmu(16)+2*sf,100*sf-(tdmu(16.29)-2*sf),fill='blue')

def print_lines(array) :
    for i,line in enumerate(array,1):
        canvas.create_line(line[0],100*sf-(line[1]),line[0]+line[17]*math.cos(line[16]),100*sf-(line[1]+line[17]*math.sin(line[16])),fill='blue',width=3,arrow='last')


def print_positions(array):
    for i,line in enumerate(array,1):
        canvas.create_oval(line[0]-1.5*sf,100*sf-(line[1]+1.5*sf),line[0]+1.5*sf,100*sf-(line[1]-1.5*sf),fill='grey')
        canvas.create_text(line[0],100*sf-line[1],text=str(i))

### Corps du programme ###

fenetre=tkinter.Tk()                           
frame=tkinter.Frame(fenetre)                    
frame_1=tkinter.Frame(frame)                    
frame_2=tkinter.Frame(frame)                    
frame_3=tkinter.Frame(frame)           
frame_1_1=tkinter.Frame(frame_1)                
frame_1_2=tkinter.Frame(frame_1)              
frame_3_1=tkinter.Frame(frame_3)     
frame_3_2=tkinter.Frame(frame_3)    
frame_4=tkinter.Frame(frame)           
canvas=tkinter.Canvas(fenetre,width=sf*100, height=sf*100, background='green')        

treat_data(fake2)

extract=tkinter.IntVar()                                                    
champ_extract = tkinter.Entry(frame_1_2, textvariable=extract)                    
extract.set(0)                                                           
fenetre.bind("<Key>",clavier)

#Affichage des éléments
frame.pack(side=tkinter.LEFT)
frame_1.pack(side=tkinter.TOP)
frame_1_1.pack(side=tkinter.LEFT)
frame_1_2.pack(side=tkinter.RIGHT)
frame_2.pack()
frame_3.pack()                     
frame_3_1.pack(side=tkinter.LEFT)           
frame_3_2.pack(side=tkinter.RIGHT)          
frame_4.pack(side=tkinter.BOTTOM)           
canvas.pack(side=tkinter.RIGHT)
fenetre.mainloop()