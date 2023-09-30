# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 09:47:59 2022

@author: zhouji
"""

import tifffile
import numpy as np
import h5py
import matplotlib.pyplot as plt
import porespy as ps
import os

# Init parameters
sample_type = 'ketton'
sample_dim = '3D' # 2D or 3D 
edge_length = 128 #image size
data_enhancement = False # whether to increase the number of dataset
stride = 64 # stride at which images are extracted
original_data = "./data/ketton/original/raw/ketton.tif" # You have to have unzipped the tif image first.

img = tifffile.imread(original_data)
img = img / 255 # binary
porosity = ps.metrics.porosity(img)
print(porosity) # calculate porosity
# Let's plot the typical image size so we can get an idea how big the images will be.
# plt.imshow(img[32, :, :], cmap="Greys")


N = edge_length
M = edge_length
O = edge_length

I_inc = stride
J_inc = stride
K_inc = stride

if sample_dim == '3D':
    #Have to have this directory to create dataset
    target_direc = os.path.join('./datasets', (sample_type + str(edge_length) + '_' + sample_dim))
    if not os.path.exists(target_direc):
        os.makedirs(target_direc)
    if len(os.listdir(target_direc)) != 0 and data_enhancement:
        count = len(os.listdir(target_direc)) # num
    else:
        count = 0
    
    for i in range(0, img.shape[0], I_inc):
        for j in range(0, img.shape[1], J_inc):
            for k in range(0, img.shape[2], K_inc):
                subset = img[i:i+N, j:j+M, k:k+O]
                if subset.shape == (N, M, O):
                    target_hdf5 = os.path.join(target_direc, sample_type + '_' + str(count) + ".hdf5")
                    f = h5py.File(target_hdf5, "w")
                    f.create_dataset('data', data=subset, compression="gzip")
                    f.close()
                    count += 1
    print (count)

elif sample_dim == '2D':
    target_direc = os.path.join('./datasets', (sample_type + str(edge_length) + '_' + sample_dim))
    if not os.path.exists(target_direc):
        os.makedirs(target_direc)
    if len(os.listdir(target_direc)) != 0 and data_enhancement:
        count = len(os.listdir(target_direc)) # num
    else:
        count = 0
    for i in range(0, img.shape[0], I_inc):
        for j in range(0, img.shape[2], J_inc):
            for k in range(0, img.shape[1], 10):
                subset = img[i:i+N, k, j:j+M]
                if subset.shape == (N, M):
                    target_hdf5 = os.path.join(target_direc, sample_type + '_' + str(count) + ".hdf5")
                    f = h5py.File(target_hdf5, "w")
                    f.create_dataset('data', data=subset, compression="gzip")
                    f.close()
                    count += 1
    print (count)

