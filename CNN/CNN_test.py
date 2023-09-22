# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:59:38 2023

@author: zhouji
"""
import torch.nn as nn
import torch
from dataset_CNN import HDF5Dataset,Dataset
import h5py
import numpy as np
from torch import Tensor
import os
from os.path import join
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as data
from models import ResNet_model,CNN_model11

os.environ["CUDA_VISIBLE_DEVICES"]="0" # 默认显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model= CNN_model11()
print(model)
model.to(device)
# model = nn.DataParallel(model).to(device) # 使用单机多Gpu训练时用
model.load_state_dict(torch.load("./weights1/CNN_predict_final0811.pth"),strict=False)


dataroot = './output/porosity0.1984valid2/' # 数据集目录
csv_path = './output/porodit0.1984_valid2_Permeability_caculation1_x.csv' # 标签目录
dataset = '3D' # 数据类型
if dataset in ['3D']:
    dataset = Dataset(dataroot,csv_path,stage=None, max_stage=None)
assert dataset

permeability_list = list()
permeability = list()
with torch.no_grad():
    perdiction_loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False)
    for testImgs,_ in perdiction_loader:
        testImgs = testImgs.to(device)
        outputs = model(testImgs).squeeze()
        outputs = outputs.detach().cpu().numpy() * 0.558309 + 1.186716

        permeability_list.append(round(outputs,4))
 
np.savetxt('./porosity0.1984valid2_test_permeability1.csv',permeability_list)


