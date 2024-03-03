import torch
from torch import nn

net = nn.Sequential(
    # 第一层是卷积层，输入通道数为1，输出通道数为6，卷积核大小为5，边缘填充为2
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    # 第二层是平均池化层，池化核大小为2，步长为2
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 第三层是卷积层，输入通道数为6，输出通道数为16，卷积核大小为5
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    # 第四层是平均池化层，池化核大小为2，步长为2
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 第五层是平坦层，用于将多维输入一维化，以便用于全连接层
    nn.Flatten(),
    # 第六层是全连接层，输入特征数为16*5*5，输出特征数为120
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    # 第七层是全连接层，输入特征数为120，输出特征数为84
    nn.Linear(120, 84), nn.Sigmoid(),
    # 第八层是全连接层，输入特征数为84，输出特征数为10
    nn.Linear(84, 10))

X = torch.rand(1, 1, 28, 28)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)