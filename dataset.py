import torch.utils.data as data
from torch import Tensor
from os import listdir
from os.path import join
import numpy as np
import h5py
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
import torch
"""
该文件根据数据集制作torch的dataset类，将图像和标签对应封装，方便后续迭代取样
"""

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".hdf5", ".h5"])


def load_img(filepath, stage, max_stage):
    img = None
    with h5py.File(filepath, "r") as f:
        img = f['data'][()]
        # print('h5py',img.shape)
    img = np.expand_dims(img, axis=0)
    torch_img = Tensor(img)
    torch_img = torch_img.div(255).sub(0.5).div(0.5)
    #torch_img = torch_img.sub(0.5).div(0.5)
    # print(torch_img)
    return torch_img

class HDF5Dataset(data.Dataset):
    def __init__(self, image_dir, csv_path,input_transform=None, target_transform=None,
                 stage=None, max_stage=None):
        super(HDF5Dataset, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_filenames.sort(key=lambda x: int(x.split(".")[1].split(".")[0].split("_")[1]))
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path,encoding='utf-8')
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.stage = stage
        self.max_stage = max_stage
        # self.images = list()
        # self.labels = list()
        # self.image_filenames = self.df['filename']
        
        self.labels= self.df['label']            

    def __getitem__(self, index):
        # img = load_img(self.df['filename'][index], self.stage, self.max_stage)
        #print('HDF5---》', input.size)
        images = load_img(self.image_filenames[index], self.stage, self.max_stage)
        if self.input_transform is not None:
            images = self.input_transform(images)
        else:
            images = images
  
        labels = self.labels[index]

        return images, labels

    def __len__(self):
        return len(self.df)

if __name__ == '__main__':
    dataroot = './datasets/berea64/'
    csv_path = './datasets/berea64_label.csv'
    # df1 = pd.read_csv(csv_path,encoding='utf-8')
    # img = load_img(df1['filename'], stage=None, max_stage=None)
    dataset = HDF5Dataset(dataroot,csv_path,
                           input_transform=None,
                          stage=None, max_stage=None
                          )
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                              shuffle=True)
    
    for i, (real_images,porosity_label) in enumerate(train_loader):
        print(i)
        print(real_images.shape)
        print(porosity_label.shape)
# images, labels = dataset.__getitem__(34) # get the 34th sample