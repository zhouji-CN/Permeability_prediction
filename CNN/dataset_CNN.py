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
from skimage.filters import threshold_otsu
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
    img = (1-img) * 255 # 测试原始样本用，因为其固相和孔隙是反的

    img = np.expand_dims(img, axis=0)
    torch_img = Tensor(img)
    # torch_img = torch_img.div(255).sub(0.5).div(0.5)
    torch_img = torch_img.sub(0.5).div(0.5) #
    # print(torch_img)
    return torch_img

class HDF5Dataset(data.Dataset):
    def __init__(self,img_rootdir,label_rootdir, input_transform=None, target_transform=None,
                 stage=None, max_stage=None):
        super(HDF5Dataset, self).__init__()  

        self.labels = list()
        self.labels1 = list()
        
        self.porosity = list()
        self.porosity1 = list()
        
        self.image_filenames = list()
        self.image_filenames1 = list()
        self.stage = stage
        self.max_stage = max_stage
        self.input_transform = input_transform
        self.target_transform = target_transform
        for ii in range(13,29):
            image_dir = img_rootdir + 'porosity' + str(ii*0.01) + '/'
            csv_path = label_rootdir + '/fake'+str(ii*0.01)+'_Permeability_caculation_x.csv'
            self.csv_path = csv_path
            self.df = pd.read_csv(self.csv_path,encoding='utf-8')       
            self.image_filename = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
            self.image_filename.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            
            self.image_filenames.extend(self.image_filename)
            self.labels.extend(self.df['permeability'])
            self.porosity.extend(self.df['porosity'])
                       
        filtered_data = [(num1, num2, img) for num1, num2, img in zip(self.labels, self.porosity, self.image_filenames) if self.start <= num1 <= self.end]
        self.labels1, self.porosity1, self.image_filenames1 = zip(*filtered_data)
        
        miu = np.mean(self.labels1)
        sigma = np.std(self.labels1)
        print(miu)
        print(sigma)
        self.labels1 = (self.labels1 - miu) / sigma 
        print(min(self.labels1))


    def __getitem__(self, index):
        images = load_img(self.image_filenames1[index], self.stage, self.max_stage)
        

        if self.input_transform is not None:
            images = self.input_transform(images)
        else:
            images = images   
  
        labels = self.labels1[index]
        
        porosity_labels = self.porosity1[index]

        return images, labels, porosity_labels



    def __len__(self):
        return len(self.labels1)


class Dataset(data.Dataset):
    def __init__(self,img_rootdir,label_rootdir, input_transform=None, target_transform=None,
                 stage=None, max_stage=None):
        super(Dataset, self).__init__()  

        self.labels = list()
        self.image_filenames = list()
        self.stage = stage
        self.max_stage = max_stage
        self.input_transform = input_transform
        self.target_transform = target_transform

        self.image_filename = [join(img_rootdir, x) for x in listdir(img_rootdir) if is_image_file(x)]
        self.image_filename.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        self.image_filenames.extend(self.image_filename)
        
        self.csv_path = label_rootdir
        self.df = pd.read_csv(self.csv_path,encoding='utf-8')
        self.labels.extend(self.df['permeability'])

    def __getitem__(self, index):
        images = load_img(self.image_filenames[index], self.stage, self.max_stage)
        
        if self.input_transform is not None:
            images = self.input_transform(images)
        else:
            images = images   
  
        labels = self.labels[index]

        return images, labels

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    img_rootdir = './output/porosity0.1984valid'
    label_rootdir = './output/porosity0.1984_valid_Permeability_caculation1_x.csv'
    
    dataset = Dataset(img_rootdir,label_rootdir,input_transform=None,stage=None, max_stage=None)
    
    # img_rootdir = './datasets/'
    # label_rootdir = './permeability_label1/'
    # dataset = HDF5Dataset(img_rootdir,label_rootdir,input_transform=None,stage=None, max_stage=None)
    print(dataset.__len__()) 
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=50,
    #                                           shuffle=True)
    
    # for i, (real_images,porosity_label) in enumerate(train_loader):
    #     print(i)
    #     print(real_images.shape)
    #     print(porosity_label.shape)
