# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:23:28 2023

@author: zhouji
"""

import h5py
import os
import numpy as np
import porespy as ps
import pandas as pd
import cv2
from skimage.filters import threshold_otsu
from skimage import io 

# init paramaters
sample_type = 'beadpack'
sample_dim = '3D' # 2D or 3D 
sample_size = 64
dataroot = os.path.join("./datasets/", (sample_type + str(sample_size) + '_' + sample_dim))
output_path = os.path.join('./datasets/', sample_type + 
                           str(sample_size) + '_' + sample_dim + 'original_porosity.csv')

image_filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]
num = len(image_filenames)
porosity = np.zeros((num,))
image_filenames.sort(key=lambda x: int(x.split(sample_type + "_")[1].split(".")[0]))

jj = 0
for filename in enumerate(image_filenames):
    with h5py.File(filename[1], 'r') as f:
        img = f['data'][()].astype(np.float32) 
        f.close()
        img_otsu = threshold_otsu(img)
        img_out = (img >= img_otsu).astype(np.float32)
        # img_out = 1-img_out
        # io.imsave(f'./datasets/sample{jj}.tif',img_out[:,:,32])
        porosity[filename[0]] = ps.metrics.porosity(img_out)
        jj = jj + 1
          
data =   { 'filename' : image_filenames,
        'label' : porosity}

df = pd.DataFrame(data)
df.to_csv(output_path, encoding='utf-8')

