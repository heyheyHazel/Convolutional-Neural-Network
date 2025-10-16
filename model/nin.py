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

# 定义模型NiN
class NIN(nn.Module):
    # 定义NiN整体网络结构
    # 三个NiN块 + 三个最大池化 + 一个全局平均池化
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            self._nin_block(3, 96, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(3, stride=2),
            self._nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            self._nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            self._nin_block(384, num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    # 定义NiN块
    def _nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        return x.flatten(1)