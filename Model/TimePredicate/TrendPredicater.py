import math
import torch.nn.functional as F
import torch
from torch import nn

"""
时间序列预测模块主要预测数据趋势，用于指导生成模型生成数据
模型本体为：
    数据嵌入层 ---> Temporal Attention ---> Res and Norm ---> TCN  ---> Res and Norm ---> 1x1 conv ---> 
"""

## 位置嵌入
## 参数 总维度
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

## 值嵌入
## 参数 输入维度，总维度
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

# 数据嵌入层
## 参数 输入维度，总维度
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

## 注意力层
## 参数 输入维度，头数量，头维度，输出维度
class AattentionLayer(nn.Module):
    def __init__(self,input_dim,num_heads,head_dim,output_dim):
        super(AattentionLayer, self).__init__()
        self.dim = num_heads*head_dim
        self.head =num_heads
        self.head_dim =head_dim
        self.qkv_layer = nn.Linear(input_dim,self.dim*3)
        self.fc_out = nn.Linear(self.dim, output_dim)
        self.soft = nn.Softmax(dim=3)
    def forward(self,data,mask=None):
        N = data.shape[0]
        L = data.shape[1]
        qkv = self.qkv_layer(data)
        q = qkv[:,:,:self.dim].reshape(N,L,self.head,self.head_dim)
        k = qkv[:,:,self.dim:self.dim*2].reshape(N,L,self.head,self.head_dim)
        v = qkv[:,:,self.dim*2:].reshape(N,L,self.head,self.head_dim)

        scores = torch.einsum("nqhd,nkhd->nhqk",[q,k])
        if mask is not None:
            mask = mask.unsqueeze(1)
            # 使用 expand 将矩阵扩展为 (128, 2, 120, 120)
            mask = mask.expand(-1, scores.shape[1], -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = self.soft(scores/(self.head_dim**(1/2)))
        out = torch.einsum("nhql,nlhd->nqhd",[attention,v]).reshape(N,L,self.dim)
        out = self.fc_out(out)
        return out

#TCN 层
## 参数 输入维度，输出维度，卷积核大小（卷积核长度）
class TCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0.1, dilation=1):
        super(TCNLayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1d(x.transpose(1,2)).transpose(1,2)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x + residual

#学习层
# 输入维度，头数量，头维度，输出维度，隐藏层维度，核大小
class ATlayer(nn.Module):
    def __init__(self,input_dim,num_heads,head_dim,output_dim,hidden_dim,kernel_size=12,drop_rate =0.1):
        super(ATlayer, self).__init__()
        self.tcn = TCNLayer(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, dropout_rate=0.1)
        self.att = AattentionLayer(input_dim=hidden_dim,num_heads=num_heads,head_dim=head_dim,output_dim=output_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 =nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop_rate)
    def forward(self,x):
        x = self.tcn(x)
        res = self.att(x)
        x = self.drop(self.norm2(x + res))
        out = self.relu(self.fc(x))
        return out

class OutLayer(nn.Module):
    def __init__(self,input_len,output_len,input_dim,output_dim):
        super(OutLayer, self).__init__()

        self.fc = nn.Linear(input_len,output_len)
        self.out = nn.Linear(input_dim,output_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = x.transpose(1,2)
        x = self.relu(self.fc(x)).transpose(1,2)
        out = self.out(x)
        return out

"""
多步生成预测，将已生成的时间序列与输入一同放入训练
"""
# class mutistepOut(nn.Module):
#     def __init__(self):
#         super(mutistepOut, self).__init__()
#         self
#
#     def forward(self,x):
#         out_1 = self.fc_out(x)

class mutistepOutLayer(nn.Module):
    def __init__(self,input_len,output_len,input_dim,output_dim,out_step):
        super(mutistepOutLayer, self).__init__()
        self.output_len = output_len
        self.output_dim = output_dim
        self.input_len = input_len
        self.len = output_len/out_step

        self.result = None
        self.fc_dim = nn.Linear(input_dim,output_dim)
        self.relu = nn.ReLU()
        self.concat = nn.Conv1d(in_channels=2, out_channels=1,kernel_size=1)
        self.fc_outs = nn.ModuleList([
            nn.Linear(input_len,int(self.len*(i+1))) for i in range(out_step)
        ])

    def forward(self,x):
        # B,720,1
        x = self.relu(self.fc_dim(x)).transpose(1,2)
        outs = []
        for fc_out in self.fc_outs:
            out = fc_out(x)
            _,_,len = out.shape
            outs.append(out.transpose(1,2))
            ## 相加
            res =F.pad(out, (0, int(self.input_len - len)))
            x = torch.cat((x, res), dim=1)
            x = self.relu(self.concat(x))
        return outs

class ATime(nn.Module):
    def __init__(self,c_in,c_out,input_len,output_len,embed_dim,num_heads,head_dim,hidden_dim,kernel_size=12,num_layers=4):
        super(ATime, self).__init__()

        self.embed = DataEmbedding(c_in=c_in,d_model=embed_dim)

        self.atlayer = nn.ModuleList(
            [ATlayer(input_dim=embed_dim,num_heads=num_heads,head_dim=head_dim,output_dim=embed_dim,hidden_dim=hidden_dim,kernel_size=kernel_size) for _ in
             range(num_layers - 1)])

        self.outlayer = OutLayer(input_len=input_len,output_len=output_len,input_dim=embed_dim,output_dim=c_out)
        self.mutiout = mutistepOutLayer(input_len=input_len,output_len=output_len,input_dim=embed_dim,output_dim=c_out,out_step=3)

    def forward(self,input):
        x = self.embed(input)
        for layer in self.atlayer:
            x = layer(x)
        out = self.outlayer(x)
        return out
