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

# 定义残差块 residual block
class BasicBlock(nn.Module):
    expansion = 1   # 通道的扩展系数，应满足in_channels = expansion * out_channels

    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicBlock, self).__init__()

        # 残差块的左侧部分（正常连接）
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差块的右侧部分（短路连接）
        self.shortcut = nn.Sequential()
        # 确认输入输出通道数相等
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 完整的ResNet网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # full connection layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out) 
        out = self.layer2(out)  
        out = self.layer3(out)  
        out = self.layer4(out)  
        out = self.avgpool(out) 
        out = torch.flatten(out, 1)
        out = self.fc(out) 
        return out

# 创建ResNet模型的辅助函数
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
def ResNet50():
    from torchvision.models.resnet import Bottleneck
    return ResNet(Bottleneck, [3, 4, 6, 3])
