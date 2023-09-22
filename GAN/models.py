import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Conv3d(nn.Module): # 三维卷积
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = math.sqrt(2.0) / math.sqrt(in_channels)

    def forward(self, x):
        out = F.conv3d(x, self.weight * self.scale, self.bias, self.stride, self.padding)
        return out


class PixelNorm(nn.Module): # PixelNorm没有需要训练的参数，减轻了计算开销
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8) 
        # torch.rsqrt开方后取倒数
        return x


class MiniBatch(nn.Module): # 此恒定特征图是从跨空间位置的一批中所有特征的标准偏差得出的
    def __init__(self):
        super(MiniBatch, self).__init__()
        self.offset = 1e-8

    def forward(self, x):
        # 计算N个标准偏差，首先计算每一张特征图的标准偏差，
        stddev = torch.sqrt(torch.mean((x - torch.mean(x, dim=0, keepdim=True))**2, dim=0, keepdim=True) + self.offset)
        inject_shape = list(x.size())[:] # 列表，包括批次N，通道，以及图片尺寸
        inject_shape[1] = 1 # 通道数=1
        inject = torch.mean(stddev, dim=1, keepdim=True) # 标准偏差平均后得到一个值
        inject = inject.expand(inject_shape) # 复制该值
        return torch.cat((x, inject), dim=1) # 和输出进行堆叠


class ConvBlock(nn.Module): # 卷积块，包括卷积层、激活层和PixelNorm层
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act, norm):
        super(ConvBlock, self).__init__()
        self.conv = Conv3d(in_channels, out_channels, kernel_size, stride, padding)

        if act == "lrelu":
            self.act = nn.LeakyReLU(0.2)
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = None

        if norm:
            self.norm = PixelNorm()
        else:
            self.norm = None

    def forward(self, x):
        out = self.conv(x)
        if self.act is not None:
            out = self.act(out)
        if self.norm is not None:
            out = self.norm(out)

        return out


class Generator(nn.Module): # 生成器
    def __init__(self, max_stage=5, base_channels=16, image_channels=1):
        super(Generator, self).__init__()
        self.max_stage = max_stage

        self.toRGBs = nn.ModuleList()
        for i in reversed(range(self.max_stage + 1)): # reversed返回一个反转的迭代器,[7->0]
            in_channels = min(base_channels * 2 ** i, 512)
            self.toRGBs.append(ConvBlock(in_channels, image_channels, 1, 1, 0, "tanh", False)) # 通道转换

        self.blocks = nn.ModuleList()
        self.blocks.append(self.first_conv_block(base_channels * 2 ** self.max_stage, base_channels * 2 ** self.max_stage))
        for i in reversed(range(self.max_stage)):
            in_channels = min(base_channels * 2 ** (i + 1), 512)
            out_channels = min(base_channels * 2 ** i, 512)
            self.blocks.append(self.conv_block(in_channels, out_channels, 3, 1, 1))

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, label, inputs, alpha, stage):
        stage = min(stage, self.max_stage)

        x1 = inputs
        x2 = label
        # x2.view()
        x = torch.cat((x1,x2),dim=1)
        # print('input--->',x.shape)
        for i in range(0, stage):
            x = self.blocks[i](x)
            x = self.upsample(x)
            #print('stage--->',x.shape)

        identity = x
        x = self.blocks[stage](x)
        x = self.toRGBs[stage](x)

        if alpha % 1 != 0: # 如果有α，则融合图像
            identity = self.toRGBs[stage - 1](identity)
            x = alpha * x + (1 - alpha) * identity
        # print('output-->',x.shape)
        return x

    def first_conv_block(self, in_channels, out_channels):
        layers = nn.Sequential(
            ConvBlock(in_channels, out_channels, 4, 1, 3, "lrelu", True),
            ConvBlock(out_channels, out_channels, 3, 1, 1, "lrelu", True))
        return layers

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layers = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, stride, padding, "lrelu", True),
            ConvBlock(out_channels, out_channels, kernel_size, stride, padding, "lrelu", True))
        return layers


class Discriminator(nn.Module):
    def __init__(self, max_stage=5, base_channels=16, image_channels=1):
        super(Discriminator, self).__init__()
        self.max_stage = max_stage

        self.fromRGBs = nn.ModuleList() # 构造函数列表
        for i in reversed(range(self.max_stage + 1)):
            out_channels = min(base_channels * 2 ** i, 512)
            self.fromRGBs.append(ConvBlock(image_channels, out_channels, 1, 1, 0, "lrelu", False))

        self.blocks = nn.ModuleList()
        self.blocks.append(self.first_conv_block(base_channels * 2 ** self.max_stage, 1))
        for i in reversed(range(self.max_stage)):
            in_channels = min(base_channels * 2 ** i, 512)
            out_channels = min(base_channels * 2 ** (i + 1), 512)
            self.blocks.append(self.conv_block(in_channels, out_channels, 3, 1, 1))

        self.downsample = nn.AvgPool3d(2, 2)
        self.minibatch = MiniBatch()

    def forward(self, label, inputs, alpha, stage):
        stage = min(stage, self.max_stage)
        x2 = label
        x1 = inputs
        # x1 = x1.view(-1, 1, 64, 64, 64)        
        x = torch.cat((x1,x2),dim=1)
        x = self.fromRGBs[stage](inputs)
        #print('from-RGB-->',x.shape)
        for i in range(stage, 0, -1):
            x = self.blocks[i](x)
            #print('block[i]-->',x.shape)
            x = self.downsample(x)
            #print('downsample-->',x.shape)
            if i == stage and alpha % 1 != 0:
                identity = self.downsample(inputs)
                identity = self.fromRGBs[stage - 1](identity)
                x = alpha * x + (1 - alpha) * identity

        x = self.minibatch(x)
        x = self.blocks[0](x)

        return x.squeeze()

    def first_conv_block(self, in_channels, out_channels):
        layers = nn.Sequential(
            ConvBlock(in_channels + 1, in_channels, 3, 1, 1, "lrelu", False),
            ConvBlock(in_channels, in_channels, 4, 1, 0, "lrelu", False),
            nn.Conv3d(in_channels, out_channels, 1, 1, 0))
        return layers

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layers = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size, stride, padding, "lrelu", False),
            ConvBlock(in_channels, out_channels, kernel_size, stride, padding, "lrelu", False))
        return layers


