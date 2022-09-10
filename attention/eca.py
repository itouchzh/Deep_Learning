from numpy import pad
import torch
import torch.nn as nn
import math


class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg(x).view([b, 1, c])
        y = self.conv(y)
        y = self.sig(y).view([b, c, 1, 1])
        out = x * y
        return out


model = ECA(8)
model = model.cuda()
input = torch.randn(1, 8, 12, 12).cuda()
output = model(input)
print(output.shape)  # (1, 8, 12, 12)

