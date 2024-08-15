import torch
import torch.nn as nn
# 自定义损失函数类
class MyMseLoss(nn.Module):
    def __init__(self):
        super(MyMseLoss, self).__init__()

    def forward(self, output, target):
        # 计算损失
        loss = torch.mean((output - target) ** 2)  # 简单的均方误差
        return loss

class MutiLoss(nn.Module):
    def __init__(self):
        super(MutiLoss, self).__init__()

    def forward(self, outs, target):
        losses = 0
        for i in range(3):
            _,len,_ = outs[i].shape
            # 计算损失
            loss = torch.mean((outs[i] - target[:,:len,:]) ** 2)  # 简单的均方误差
            losses+=loss*((i+1)/6)
        return loss
