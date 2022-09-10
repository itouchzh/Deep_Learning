import torch
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, channel,ratio = 4):
        super(SE,self).__init__()
        # 第一步：全局平均池化,输入维度(1,1,channel)
        self.avg = nn.AdaptiveAvgPool2d(1)
        # 第二步：全连接，通道数量缩减
        self.fc1 = nn.Linear(channel, channel//ratio,False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(channel// ratio, channel,False)
        self.act2 = nn.Sigmoid()
    def forward(self,x):
        b, c, h, w = x.size()
        # b, c, 1, 1->b,c
        y = self.avg(x).view(b,c)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.act2(y).view(b,c,1,1)
        # print(y)
        return x * y

model = SE(8)
model = model.cuda()
input = torch.randn(1, 8, 12, 12).cuda()
output = model(input)
print(output.shape) # (1, 8, 12, 12)