#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:48:01 2023

@author: amax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 



class ResBlock1(nn.Module):
    def __init__(self, in_chans,out_chans):
        super(ResBlock1, self).__init__()
        
        self.conv1 = nn.Conv3d(in_chans,out_chans,kernel_size=1,stride=1, padding=0)
        
        self.conv2 = nn.Conv3d(out_chans,out_chans,kernel_size=3,stride=1, padding=1)
        
        self.conv3 = nn.Conv3d(out_chans,out_chans,kernel_size=3,stride=1, padding=1)
        
        self.batch_norm = nn.BatchNorm3d(num_features=out_chans)  
        

        nn.init.kaiming_normal_(self.conv1.weight,nonlinearity='relu') 
        nn.init.kaiming_normal_(self.conv2.weight,nonlinearity='relu') 

        nn.init.constant_(self.batch_norm.weight, 0.5)  

        nn.init.zeros_(self.batch_norm.bias)
        
       

    def forward(self, x):
        # out = F.relu(self.batch_norm(self.conv1(x)))
        # out = self.batch_norm(self.conv2(x))  
       
        
        out = F.relu(self.batch_norm(self.conv1(x)))
        out = F.relu(self.batch_norm(self.conv2(out)))
        out = self.batch_norm(self.conv3(out))
        
        
        # out = F.relu(self.conv1(x))
        # out = F.relu(self.conv2(out))
        # out = self.conv3(out)
        
        return F.relu(out + x)

# Resnet   
class ResNet_model(nn.Module):
    def __init__(self, in_chans1=1,out_chans1=16, n_blocks=2):
        super(ResNet_model, self).__init__()
        self.in_chans1 = in_chans1
        self.out_chans1 = out_chans1
        # self.batch_norm1 = nn.BatchNorm3d(num_features=out_chans1) 
        self.conv1 = nn.Conv3d(self.in_chans1, self.out_chans1, kernel_size=1, stride=1, padding=0)
        # self.BN1 = nn.BatchNorm3d(num_features = self.out_chans1)  
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock1(in_chans=self.out_chans1,out_chans=self.out_chans1)])
        ) 
        
        self.conv2 = nn.Conv3d(self.out_chans1, self.out_chans1, kernel_size=3, stride=1, padding=1)        

        self.conv3 = nn.Conv3d(self.out_chans1, self.out_chans1, kernel_size=3, stride=1, padding=1) 
        
        self.fc1 = nn.Linear(4 * 4 * 4 * self.out_chans1 + 1, out_features=250, bias=True)
        self.fc2 = nn.Linear(250, 50)
        self.fc3 = nn.Linear(50, 1)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, porosity_labels):
        out = F.relu(self.conv1(x))


        
        out = self.resblocks(out)
        out = F.max_pool3d(F.relu(self.conv2(out)), 2) 
        
        out = self.resblocks(out)
        out = F.max_pool3d(F.relu(self.conv2(out)), 2)
        
        out = self.resblocks(out)
        out = F.max_pool3d(F.relu(self.conv2(out)), 2)

        out = self.resblocks(out)
        out = F.max_pool3d(F.relu(self.conv2(out)), 2)
        
        out = F.relu(self.conv3(out))
        
        out = out.view(-1, 4 * 4 * 4 * self.out_chans1)

        porosity_labels = porosity_labels.view(-1,1)
        out = torch.cat((out, porosity_labels),dim=1)
        
        out = self.dropout(F.leaky_relu(self.fc1(out),negative_slope=0.02))
        
        out = self.dropout(F.leaky_relu(self.fc2(out),negative_slope=0.02))
        
        out = self.fc3(out)

        return out    



# CNN
class CNN_model11(nn.Module):
    def __init__(self):
        super(CNN_model11, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, stride=1, padding=1)
        self.BN1 = nn.BatchNorm3d(num_features=8)
        self.conv2 = nn.Conv3d(8, 16, 3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm3d(num_features=16)
        self.conv3 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.BN3 = nn.BatchNorm3d(num_features=32)
        self.conv4 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.BN4 = nn.BatchNorm3d(num_features=16)
        self.conv5 = nn.Conv3d(16, 8, 1, stride=1, padding=0)
        self.BN5 = nn.BatchNorm3d(num_features=8)

        self.fc1 = nn.Linear(8 * 4 * 4 * 4, 100)

        self.fc2 = nn.Linear(100, 30)

        self.fc3 = nn.Linear(30, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool3d(x, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool3d(x, 2)

        x = F.relu(self.conv5(x))

        x = x.view(-1, 8 * 4 * 4 * 4)
        x = self.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.1))
        x = self.dropout(F.leaky_relu(self.fc2(x), negative_slope=0.1))
        x = self.fc3(x)
        return x

# CNN
class CNN_model111(nn.Module):
    def __init__(self):
        super(CNN_model111, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, stride=1, padding=1)
        self.BN1 = nn.BatchNorm3d(num_features=8)
        self.conv2 = nn.Conv3d(8, 16, 3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm3d(num_features=16)
        self.conv3 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.BN3 = nn.BatchNorm3d(num_features=16)
        self.conv4 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.BN4 = nn.BatchNorm3d(num_features=16)
        self.conv5 = nn.Conv3d(16, 8, 1, stride=1, padding=0)
        self.BN5 = nn.BatchNorm3d(num_features=8)

        self.fc1 = nn.Linear(8 * 4 * 4 * 4, 60)

        self.fc2 = nn.Linear(60, 30)

        self.fc3 = nn.Linear(30, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.max_pool3d(x, 2)

        x = F.relu(self.BN2(self.conv2(x)))
        x = F.max_pool3d(x, 2)

        x = F.relu(self.BN3(self.conv3(x)))
        x = F.max_pool3d(x, 2)

        x = F.relu(self.BN4(self.conv4(x)))
        x = F.max_pool3d(x, 2)

        x = F.relu(self.BN5(self.conv5(x)))

        x = x.view(-1, 8 * 4 * 4 * 4)
        x = self.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.1))
        x = self.dropout(F.leaky_relu(self.fc2(x), negative_slope=0.1))
        x = self.fc3(x)
        return x
