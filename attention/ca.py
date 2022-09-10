import torch
import torch.nn as nn

# 非线性处理部分.
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA(nn.Module):
    def __init__(self, in_channel, out_channel, ratio=8):
        super(CA, self).__init__()
        # height 方向平均池化，把w压缩为1
        self.x_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.y_avg = nn.AdaptiveAvgPool2d((1, None))
        # mip : 输出通道
        mip = max(8, in_channel // ratio)

        self.conv1 = nn.Conv2d(in_channel, mip, kernel_size=1, stride=1, bias=False)

        self.bn = nn.BatchNorm2d(mip)

        self.act = h_swish()
        self.conv2 = nn.Conv2d(
            mip, out_channels=out_channel, kernel_size=1, stride=1, bias=False
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        identity = x
        x_h = self.x_avg(x).permute(0, 1, 3, 2)  # (b, c, h, 1)->(b, c, 1, h)
        x_w = self.y_avg(x)  # (b, c, 1, w)

        y = torch.cat([x_h, x_w], dim=3)  # (b, c, 1, (w + h))
        y = self.conv1(y)

        y = self.bn(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=3)  # x_h:(b, c, 1, h),x_w:(b, c, 1, w)
        x_h = x_h.permute(0, 1, 3, 2)

        a_h = self.conv2(x_h).sigmoid()
        a_w = self.conv2(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


model = CA(8, 8)
model = model.cuda()
torch.manual_seed(0)
torch.cuda.manual_seed(0)
input = torch.randn(1, 8, 12, 12).cuda()
output = model(input)
print(output.shape)  # (1, 8, 12, 12)

