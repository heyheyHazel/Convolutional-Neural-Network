# Convolutional-Neural-Network
Here is PyTorch implementation of CNN.

![GitHub last commit](https://img.shields.io/github/last-commit/heyheyHazel/Convolutional-Neural-Network)
![GitHub repo size](https://img.shields.io/github/repo-size/heyheyHazel/Convolutional-Neural-Network)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)

## 🎯 项目简介

从零开始实现各种卷积神经网络架构的完整代码库，包含经典CNN模型以及在计算机视觉任务中的应用。本项目旨在深入理解CNN的工作原理，并提供可复用的实现代码。


## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.6.0
- **核心语言**: Python 3.11.13
- **数据处理**: NumPy｜Pandas
- **可视化**: Matplotlib｜Seaborn

## 📁 项目目录

### 🔧 CNN完整代码实现

- [🧩 数据加载与处理](/dataloader)
- [🛫 经典模型实现](/model)
  - [LeNet--首个CNN](lenet.py)
  - [AlexNet--更大更深的网络](alexnet.py)
  - [VGG--块状网络结构](vgg.py)
  - [NiN--网络中的网络](nin.py)
  - [GoogLeNet--并行结构的网络](googlenet.py)
  - [ResNet--残差网络结构](resnet.py)
- [🔥 模型训练与预测](/prediction)






## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/heyheyHazel/Convolutional-Neural-Network.git
cd Convolutional-Neural-Network

# 创建虚拟环境并安装基础的包(conda)
conda create -n pytorch_env python=3.9 -y
conda activate pytorch_env
conda install pytorch pandas numpy matplotlib ipykernel  -y
