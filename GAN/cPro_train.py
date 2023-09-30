import argparse
import torch
import torchvision.utils as vutils
import os
from torch.optim import Adam
from torch.autograd import grad
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from dataset import HDF5Dataset
from hdf5_io import save_hdf5
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import h5py
from tqdm import tqdm


work_dir = os.path.expandvars('./')
# torch.set_default_tensor_type(torch.FloatTensor)
# os.environ["CUDA_VISIBLE_DEVICES"]="0,2"

# 参数定义
num_stages = 5
# num_epochs = 500
num_epochs = list([2, 2, 2, 2, 2, 2, 2, 2, 1])
base_channels = 16
batch_size = list([32, 32, 32, 16, 2, 2, 2, 2, 2]) # 4->8->16->32->64->128->256,随着尺寸变大，batchsize改小
dataroot = './datasets/berea64/' # 数据集目录
csv_path = './datasets/berea64_label.csv' # 标签目录
dataset = '3D' # 数据类型
image_size = 64
image_channels = 1
device = torch.device('cuda')
       
    
# 计算梯度惩罚项
class GradientPenalty:
    def __init__(self, batch_size, gp_lambda, device):
        self.batch_size = batch_size
        self.gp_lambda = gp_lambda
        self.device = device
    # __call__方法使类对象可以像实例对象那样被调用
    def __call__(self, discriminator, real_data, fake_data, progress):
        # 定义alpha为[batch_size,1,1,1,1]的张量
        alpha = torch.rand(self.batch_size, 1, 1, 1, 1, requires_grad=True, device=self.device)
        # print('alpha-->', alpha.shape)
        # print('real_data-->', real_data.shape)
        # print('fake_data-->', fake_data.shape)
        interpolates = (1 - alpha) * real_data + alpha * fake_data # 插值
        d_interpolates = discriminator(gporosity_label,interpolates, progress.alpha, progress.stage)

        gradients = grad(outputs=d_interpolates,
                         inputs=interpolates,
                         grad_outputs=torch.ones(d_interpolates.size(), device=self.device),
                         create_graph=True,
                         retain_graph=True,
                         only_inputs=True)[0]
        gradients = gradients.view(self.batch_size, -1)
        gradient_penalty = self.gp_lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

# 渐进项
class Progress:
    def __init__(self, max_stage, max_epoch, max_step):
        self.alpha = 0
        self.stage = 0
        self.max_stage = max_stage
        self.max_epoch = max_epoch
        self.max_step = max_step

    def progress(self, current_stage, current_epoch, current_step):
        self.stage = current_stage
        # 当前step/最大step
        p = (current_epoch * self.max_step + current_step) / (self.max_epoch * self.max_step)
        self.alpha = p if 0 < current_stage < self.max_stage else 1


generator = Generator(max_stage=num_stages, base_channels=base_channels, image_channels=image_channels).to(device)
generator = nn.DataParallel(generator).to(device) # 使用单机多Gpu训练时用
discriminator = Discriminator(max_stage=num_stages, base_channels=base_channels, image_channels=image_channels).to(device)
discriminator = nn.DataParallel(discriminator).to(device)


g_optimizer = Adam(generator.parameters(), lr=1e-3, betas=(0, 0.99))
d_optimizer = Adam(discriminator.parameters(), lr=1e-3, betas=(0, 0.99))

start_epoch = -1
start_stage = -1

for stage in range(start_stage+1,num_stages + 1):
    #train_dataset = get_dataset(data_name=opt.dataset, data_root=opt.data_root, stage=stage, max_stage=opt.num_stages, train=True)
    #train_loader = DataLoader(train_dataset, batch_size=opt.batch_size[stage], shuffle=True)
    if dataset in ['3D']:
        dataset = HDF5Dataset(dataroot,csv_path,
                              stage=None, max_stage=None
                              )
    assert dataset

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size[stage],
                                             shuffle=True)

    gp = GradientPenalty(batch_size[stage], 10, device) # opt.batch_size[stage]输入当前stage的batch_size()
    progress = Progress(num_stages, num_epochs[stage], len(train_loader)) # 实例化progress类,输入最大stage,当前stage的num_epoch和一个train_loader的长度
    # 定于渐进过程的固定参数max_stage, max_epoch, max_step
    
    generator.train()
    
    discriminator.train()
    for epoch in range(start_epoch +1 ,num_epochs[stage]):
        starttime = time.time()
        for i, (real_images,porosity_label) in enumerate(train_loader): # i是batch的序号
            # print(i)
            # print(real_images.shape)
            # print(porosity_label.shape)
            f = open(work_dir+"training_curve.csv", "a") # 创建文件在末尾写入
            real_images = real_images.to(device)
            # print('RRRRR--->',real_images.shape)
            # F.interpolate通过插值方法，对输入的张量数组进行上\下采样操作,得到不同stage所用的真实图像
            real_images = F.interpolate(real_images, size=[4 * 2 ** min(stage, num_stages),
                                                           4 * 2 ** min(stage, num_stages),
                                                           4 * 2 ** min(stage, num_stages)],mode='trilinear')
            # real_images.to(device)
            porosity_label = porosity_label.to(device)
            progress.progress(stage, epoch, i) # 计算当前α

            # discriminator 梯度清零
            generator.zero_grad()
            discriminator.zero_grad()

            # 噪声向量与标签数据组合成新的潜向量
            fix_latent_dim = min(base_channels * 2 ** num_stages-128, 512) 
            fix_z = torch.randn((batch_size[stage], fix_latent_dim, 1, 1, 1),device=device) 
            fix_z = fix_z.type(torch.cuda.FloatTensor) # 转换为FloatTensor
            porosity_label = porosity_label.view(batch_size[stage], 1, 1, 1, 1) 
            dporosity_label = porosity_label.view(batch_size[stage], 128, 1, 1, 1) 
            dporosity_label = dporosity_label.type(torch.cuda.FloatTensor) 
            
            # 生成假图像
            with torch.no_grad():
                fake_images = generator(dporosity_label, fix_z, progress.alpha, progress.stage)
            weidu = 4 * 2 ** min(stage, num_stages) 
            gporosity_label = porosity_label.expand(batch_size[stage], 1, weidu, weidu, weidu)
            gporosity_label = gporosity_label.type(torch.cuda.FloatTensor)
            
            d_real = discriminator(gporosity_label,real_images, progress.alpha, progress.stage).mean()
            d_fake = discriminator(gporosity_label,fake_images, progress.alpha, progress.stage).mean()

            gradient_penalty = gp(discriminator, real_images.data, fake_images.data, progress)

            epsilon_penalty = (d_real ** 2).mean() * 0.001 

            d_loss = d_fake - d_real
            d_loss_gp = d_loss + gradient_penalty + epsilon_penalty 

            d_loss_gp.backward()
            d_optimizer.step()

            # generator
            generator.zero_grad()
            discriminator.zero_grad()

            latent_dim = min(base_channels * 2 ** num_stages, 512)
            z = torch.randn(batch_size[stage], latent_dim, 1, 1, 1, device=device)
            
            fake_images = generator(dporosity_label,fix_z, progress.alpha, progress.stage)

            g_fake = discriminator(gporosity_label,fake_images, progress.alpha, progress.stage).mean()
            g_loss = -g_fake

            g_loss.backward()
            g_optimizer.step()
        endtime = time.time()
        epoch_time = endtime-starttime
        print("Stage:{} | Epoch :{} | Dis Loss:{:.5f} | Gen Loss:{:.5f}| time:{:.2f}".format(stage, epoch, d_loss, g_loss,epoch_time))
        f.write("Stage:{} | Epoch :{} | Dis Loss:{:.5f} | Gen Loss:{:.5f}| time:{:.2f}"
                .format(stage, epoch, d_loss, g_loss,epoch_time))
        f.write('\n')
        f.close()
        if epoch % 20 == 0:
            fake = generator(dporosity_label,fix_z, progress.alpha, progress.stage)
            save_hdf5(fake, work_dir +'/fake_sample/' + 'fake_samples_stage{0}_{1}.hdf5'.format(stage,epoch))
            # fake_images = fake.permute(0, 2, 3, 4, 1).detach().cpu()
            fake_images = fake.detach().cpu() 
            section = int(fake.size(4)/2) 
            fake_images_sec = fake_images[:,:,:,:,section] 
            plt.figure(figsize=(6, 6))
            plt.axis("off")
            plt.imshow(np.transpose(vutils.make_grid(fake_images_sec, nrow=6, padding=2, normalize=True),(1,2,0)))
            plt.savefig("fig/stage{}_Epoch_%d_{}".format(stage,'berea') %(epoch+1))
            plt.close('all')
        if epoch % 100 == 0:
            torch.save(generator.state_dict(), "./weights/generator_stage{}_epoch{}.pth".format(stage,epoch))
            torch.save(discriminator.state_dict(), "./weights/discriminator_stage{}_epoch{}.pth".format(stage,epoch))
            
        # do checkpointing
        if epoch % 30 == 0:
            checkpoint = {
                    'netG': generator.state_dict(),
                    'netD': discriminator.state_dict(),
                    'optimizerD':d_optimizer.state_dict(),
                    'optimizerG':g_optimizer.state_dict(),
                    'epoch': epoch,
                    'stage': stage
                }
            if not os.path.isdir("./checkpoint"):
                os.mkdir("./checkpoint")
            torch.save(checkpoint, './checkpoint/ckpt_best_%d_%d.pth' %(stage,epoch))
            # 在训练过程中每个多少个epoch保存一次网络参数，便于恢复，提高程序的鲁棒性
            
            
torch.save(generator.state_dict(), "./weights/generator.pth")
torch.save(discriminator.state_dict(), "./weights/discriminator.pth")
    













