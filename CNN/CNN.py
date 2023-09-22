# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:54:38 2023

@author: zhouji
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2" # 默认显卡
from dataset_CNN import HDF5Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split

from torchsummary import summary
from tqdm import tqdm
from sklearn.metrics import r2_score
from torchvision import transforms
import pandas as pd
from models import CNN_model11
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.manual_seed(2023)


# 数据
img_rootdir = './datasets/'
label_rootdir = './permeability_label1/'



# 载入数据集
data = HDF5Dataset(img_rootdir,label_rootdir,input_transform=None,stage=None, max_stage=None)

# 划分数据集
train_size = int(0.8*len(data))
test_size = len(data) - train_size
traindata,testdata = random_split(data, [train_size,test_size])

#设定每一个Batch的大小
Batch_size = 100

traindataloader = DataLoader(dataset=traindata, batch_size=Batch_size, shuffle=True)
testdataloader = DataLoader(dataset=testdata, batch_size=Batch_size, shuffle=False)


torch.manual_seed(1234)
# 实例化模型并显示网络架构

model = CNN_model11()
# print(model)
model = nn.DataParallel(model).to(device) # 使用单机多Gpu训练时用
# model.to(device)
# # 显示模型架构
summary(model,(1,64,64,64))

# 实例损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

valid_R2_score = []
train_R2_score = []
loss_list = []
valid_loss = []
# 定义训练轮次
EPOCHS = 300
#存储训练过程
history = {'Test Loss':[],'Test Accuracy':[]} 
for epoch in range(1,EPOCHS + 1):
    processBar = tqdm(traindataloader,unit = 'step') 
    model.train(True) #  设置模式为训练模式，默认为True

    for step,(trainImgs,labels,porosity_labels) in enumerate(processBar):
        # 数据存放到device,cpu或者gpu
        trainImgs = trainImgs.to(device)
        labels = labels.to(device).float()
        porosity_labels = porosity_labels.to(device).float()
        

        model.zero_grad() # 梯度清零，避免累加
        outputs = model(trainImgs).squeeze()
        # outputs = model(trainImgs,porosity_labels).squeeze() # 孔隙量物理参数数据输入网络

        train_R2 = r2_score(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), multioutput= 'uniform_average')
        train_R2_score.append(train_R2)
        loss = criterion(outputs,labels) # 计算损失
        loss_list.append(loss)
        
        loss.backward() # 反向传播
        optimizer.step() # 迭代步

        processBar.set_description("[%d/%d] Loss: %.4f  train_R2_loss: %.4f" % 
                            (epoch,EPOCHS,loss.item(),train_R2))

        
        # 测试过程
        if step == len(processBar)-1:
            model.train(False)
            for testImgs,labels,porosity_labels in testdataloader:
                testImgs = testImgs.to(device)
                
                validlabels = labels.to(device).float()
                porosity_labels = porosity_labels.to(device).float()
                # validoutputs = model(testImgs,porosity_labels).squeeze() # 孔隙量物理参数数据输入网络

                validoutputs = model(testImgs).squeeze()
                
                # loss = criterion(validoutputs,validlabels)
                # valid_loss.append(loss)
                valid_R2 = r2_score(validlabels.detach().cpu().numpy(), validoutputs.detach().cpu().numpy(), multioutput= 'uniform_average')
                valid_R2_score.append(valid_R2)
        
        
    if epoch%50==0:
        torch.save(model.state_dict(), "./weights1/CNN_predict"+ str(epoch) + ".pth")
    if epoch%30==0:
        R2_score1 = pd.DataFrame(data=valid_R2_score)
        R2_score1.to_csv('valid_R2_score' + str(epoch) + '.csv')

np.savetxt('./train_R2_score.csv',train_R2_score)
np.savetxt('./train_loss.csv',loss_list)   
np.savetxt('./valid_R2_score_final.csv',valid_R2_score)     
# np.savetxt('./valid_loss.csv',valid_loss)

torch.save(model.state_dict(), "./weights1/CNN_predict_final0811.pth")


