# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:17:02 2022

@author: zhouj
"""

import numpy as np
import porespy as ps
import tifffile
import glob
import h5py

ps.visualization.set_mpl_style()
np.random.seed(10)

resolution = 64
data11 = np.empty((resolution*3,resolution//2-1))

#获取指定目录下的所有图片
img_list = glob.glob(r'./output/g_0.2/*.hdf5') #加上r让字符串不转义

A = np.zeros((len(img_list),resolution//2-1))
n_ = 0
for file in img_list:
    # img = tifffile.imread(file)
    f = h5py.File(file, 'r')
    img = f['data'][()]
    img = 1- img
    for i in range(resolution):
        data_x = ps.metrics.two_point_correlation(im=img[i])
        data_y = ps.metrics.two_point_correlation(im=img[:,i])
        data_z = ps.metrics.two_point_correlation(im=img[:,:,i])
        data11[i] = data_x.probability
        data11[resolution+i] = data_y.probability
        data11[resolution*2+i] = data_z.probability
    A[n_]=np.mean(data11,axis=0)
    n_ += 1
    print(n_)
np.savetxt(r'./analysis/Caovariance_Bedpack_96_real.csv', A, fmt ='%1.6f', delimiter=',')
