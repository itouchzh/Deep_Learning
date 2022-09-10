from operator import mod
from turtle import forward
from cv2 import randn
from matplotlib.pyplot import cla
from sqlalchemy import true
import torch
import torch.nn as nn


class channel_attention(nn.Module):
    # 缩减倍率
    def __init__(self, channel, ratio=8):
        super(channel_attention, self).__init__()
        self.max = nn.AdaptiveMaxPool2d(1)  # 1,3,1,1
        self.avg = nn.AdaptiveAvgPool2d(1)  # 1,3,1,1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y1 = self.max(x).view(b, c)
        y2 = self.avg(x).view(b, c)
        y1 = self.fc(y1)
        y2 = self.fc(y2)
        y = torch.add(y1, y2).view(b, c, 1, 1)
        return x * self.sig(y)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # torch.max()有两个返回值，第一个为张量，第二个为索引
        y1, _ = torch.max(x, dim=1, keepdim=True)
        y2 = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([y1, y2], dim=1)
        y = self.conv(y)
        return x * self.sig(y)


class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = channel_attention(channel, ratio)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        y = self.channel_attention(x)
        y = self.spatial_attention(y)
        return y


model = CBAM(8)
print(model)
input = torch.randn(1, 8, 6, 6)
output = model(input)
print(output.shape)
