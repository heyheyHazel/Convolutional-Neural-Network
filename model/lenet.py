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

# 定义模型——LeNet
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolution and pooling layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.sigmoid = nn.Sigmoid()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Full connection layer
        self.flatten = nn.Flatten()  # flatten as a vector: 5*5*16 = 400
        self.linear1 = nn.Linear(400, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    # Definite forward propagation
    def forward(self, x):
        x = self.pool1(self.sigmoid(self.conv1(x)))
        x = self.pool2(self.sigmoid(self.conv2(x)))
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x