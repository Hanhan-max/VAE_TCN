import torch
import torch.nn as nn
import torch.nn.functional as F
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

class VaeLoss(nn.Module):
    def __init__(self):
        super(VaeLoss, self).__init__()

    def forward(self,recon_x, x,input,beta=1.0):
        """
        VAE损失函数。

        参数:
        recon_x: 重构的图像 (batch_size, channels, height, width)
        x: 原始输入图像 (batch_size, channels, height, width)
        mu: 潜在空间的均值 (batch_size, latent_dim)
        logvar: 潜在空间的对数方差 (batch_size, latent_dim)
        beta: KL散度的权重系数 (float)

        返回:
        loss: VAE的总损失 (scalar)
        """

        # 重构损失（均方误差）
        recon_loss = F.mse_loss(recon_x, x)
        recon_dist = F.mse_loss(recon_x, input)

        # KL散度损失，这里的0.5是为了让结果与原始论文一致
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        # kl_loss = torch.mean(kl_loss)  # 取平均值作为损失

        # 总损失
        loss = recon_loss

        return loss