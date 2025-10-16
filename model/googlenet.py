import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import time


# 定义Inception块 包括四条并行路径
class Inception(nn.Module):
    # c1-c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1: 1x1卷积 + BatchNormalize + ReLU
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size = 1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace = True)
        )
        
        # 线路2: 1x1卷积 + 3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size = 1),
            nn.BatchNorm2d(c2[0]),
            nn.ReLU(inplace = True),
            nn.Conv2d(c2[0], c2[1], kernel_size = 3, padding = 1),
            nn.BatchNorm2d(c2[1]),
            nn.ReLU(inplace = True)
        )

        # 线路3: 1x1卷积 + 5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size = 1),
            nn.BatchNorm2d(c3[0]),
            nn.ReLU(inplace = True),
            nn.Conv2d(c3[0], c3[1], kernel_size = 5, padding = 2),
            nn.BatchNorm2d(c3[1]),
            nn.ReLU(inplace = True)
        )

        # 线路4: 3x3最大池化 + 1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            nn.Conv2d(in_channels, c4, kernel_size = 1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # 在通道维度上拼接
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
    

# GoogLeNet整体结构
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        
        # 1. 7x7卷积层 + 3x3最大池化 (输入: 32x32x3)
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) # 32x32 -> 16x16
        )
        
        # 2. 1x1卷积层 + 3x3卷积层 + 3x3最大池化
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,192, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)  # 16x16 -> 8x8
        )

        # 3. 串联2个Inception模块 + 3x3最大池化
        # 第一个inception输出通道 64+128+32+32=256
        # 第二个inception输出通道 128+192+96+64=480
        self.b3 = nn.Sequential(
            Inception(192, 64, (96,128), (16,32), 32),
            Inception(256, 128, (128,192), (32,96), 64),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)  # 8x8 -> 4x4
        )
        
        # 4. 串联5个Inception模块 + 3x3最大池化
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)  # 4x4 -> 2x2
        )

        # 5. 串联2个Inception模块 + 全局平均池化层
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 10)
        
    def forward(self,x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x