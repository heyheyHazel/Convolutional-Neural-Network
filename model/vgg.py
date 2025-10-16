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

# 定义模型2——VGG16
class VGG16(nn.Module):
    def __init__(self, num_classes=10, batch_norm=True):
        super(VGG16, self).__init__()
        # 卷积特征提取部分，是否使用批归一化（默认启用）
        self.features = self._make_layers(batch_norm)
        # 自适应池化层 (替代固定尺寸池化)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接分类器，三个全连接层
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    # VGG16的卷积层配置（13层卷积）
    def _make_layers(self, batch_norm):
        cfg = [
            64, 64, 'M',        # Block1: 2个卷积层 + 1个池化层
            128, 128, 'M',      # Block2: 2个卷积层 + 1个池化层
            256, 256, 256, 'M', # Block3: 3个卷积层 + 1个池化层
            512, 512, 512, 'M', # Block4: 3个卷积层 + 1个池化层
            512, 512, 512, 'M'  # Block5: 3个卷积层 + 1个池化层
        ]
        
        layers = []
        in_channels = 3         # 初始输入通道数（RGB三通道）  
        
        # 遍历列表，依次添加层
        for v in cfg:
            if v == 'M': # 池化层标记
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else: # 卷积层
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # 是否批归一化
                if batch_norm: # 使用归一化
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else: # 不使用
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v # 更新输入通道数为当前层输出
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x) 
        x = self.adaptive_pool(x)  
        x = torch.flatten(x, 1) 
        x = self.classifier(x) 
        return x