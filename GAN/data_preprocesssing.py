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

#You have to have unzipped the tif image first.
img = tifffile.imread("./data/berea/original/raw/berea/berea.tif")
img = img / 255
img = 1-img
porosity = ps.metrics.porosity(img) 
# # 原数据由tiff转换为hdf5，方便分析
# Original_tif = h5py.File("./data/berea/Original_berea.hdf5", "w")
# Original_tif.create_dataset('data',data=img,compression="gzip")
# Original_tif.close()


#Let's plot the typical image size so we can get an idea how big the images will be.
# plt.imshow(img[32, :, :], cmap="Greys")

count = 0

edge_length = 200 #image dimensions
stride = 200 #stride at which images are extracted

N = edge_length
M = edge_length
O = edge_length

I_inc = stride
J_inc = stride
K_inc = stride

#Have to have this directory to create dataset
target_direc = "./datasets/berea200/berea_"
for i in range(0, img.shape[0], I_inc):
    for j in range(0, img.shape[1], J_inc):
        for k in range(0, img.shape[2], K_inc):
            subset = img[i:i+N, j:j+M, k:k+O]
            if subset.shape == (N, M, O):
                f = h5py.File(target_direc+str(count)+".hdf5", "w")
                f.create_dataset('data', data=subset, compression="gzip")
                f.close()
                count += 1
print (count)