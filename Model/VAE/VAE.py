import math

import torch
import torch.nn as nn




class vaeAttn(nn.Module):
    def __init__(self,embedding_dim,num_heads,d_model):
        super(vaeAttn, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim =  d_model // num_heads
        assert (
                self.head_dim * num_heads == d_model
        ), "Embedding size needs to be divisible by heads"

        self.qkv = nn.Linear(embedding_dim,3*d_model)
        self.out = nn.Linear(d_model,embedding_dim)
        self.soft = nn.Softmax(dim=3)
    def forward(self,x,mask=None):
        B,L,D = x.shape
        x = self.qkv(x)
        q = x[:,:,:self.d_model].reshape(B,L,self.num_heads,self.head_dim)
        k = x[:,:,self.d_model:2*self.d_model].reshape(B,L,self.num_heads,self.head_dim)
        v = x[:,:,2*self.d_model:].reshape(B,L,self.num_heads,self.head_dim)
        scores = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        if mask is not None:
            mask = mask.unsqueeze(1)
            # 使用 expand 将矩阵扩展为 (128, 2, 120, 120)
            mask = mask.expand(-1, scores.shape[1], -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = self.soft(scores/(self.head_dim**(1/2)))
        out = torch.einsum("nhql,nlhd->nqhd",[attention,v]).reshape(B,L,self.d_model)
        out = self.out(out)
        return out

class VaeEncoder(nn.Module):
    def __init__(self,embedding_dim,num_heads,d_model,drop=0.1):
        super(VaeEncoder, self).__init__()
        self.attn = vaeAttn(embedding_dim = embedding_dim,num_heads=num_heads,d_model=d_model)
        self.fc1 = nn.Linear(embedding_dim,embedding_dim)
        self.fc2 = nn.Linear(embedding_dim,embedding_dim)
        self.drop = nn.Dropout(drop)
        self.lnorm_1 = nn.LayerNorm(embedding_dim)
        self.lnorm_2 = nn.LayerNorm(embedding_dim)
        self.relu = nn.SiLU()
    def forward(self,x):
        attn_out = self.attn(x)
        x = self.drop(self.lnorm_1(x+attn_out))
        fc_out = self.relu(self.fc1(x))
        fc_out = self.fc2(fc_out)
        x = self.drop(self.lnorm_2(x+fc_out))
        return x



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


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class EncoderEnd(nn.Module):
    def __init__(self,embedding_dim):
        super(EncoderEnd, self).__init__()
        self.mean = nn.Linear(embedding_dim,embedding_dim)
        self.logvar = nn.Linear(embedding_dim,embedding_dim)

    def forward(self,encode):
        mean = self.mean(encode)
        logvar = self.logvar(encode)

        return mean,logvar


class timeVae(nn.Module):
    def __init__(self,input_dim,embedding_dim,num_heads,d_model,e_num_layers=2,d_num_layers=2,drop=0.1):
        super(timeVae, self).__init__()
        self.embedding = DataEmbedding(input_dim,embedding_dim)
        self.encoders = nn.ModuleList(
            [VaeEncoder(embedding_dim,num_heads,d_model,drop=0.1) for _ in
             range(e_num_layers - 1)])
        self.encoder_end = EncoderEnd(embedding_dim=embedding_dim)
        self.decoder = nn.ModuleList(
            [VaeEncoder(embedding_dim,num_heads,d_model,drop=0.1) for _ in
             range(d_num_layers - 1)])
        self.output = nn.Linear(embedding_dim,input_dim)
        self.silu = nn.SiLU()
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input, mask=None):
        x = self.embedding(input)
        x = input + self.silu(x)
        for layer in self.encoders:
            x = layer(x)
        mu, logvar = self.encoder_end(x)
        z = self.reparameterize(mu, logvar)
        for layer in self.encoders:
            z = layer(z)
        out = self.output(self.silu(z))
        return out, mu, logvar

class AEdown(nn.Module):
    def __init__(self,embedding_dim,num_heads,d_model,e_num_layers):
        super(AEdown, self).__init__()
        self.u_down = nn.ModuleList(
            [VaeEncoder(embedding_dim, num_heads, d_model, drop=0.1) for _ in
             range(e_num_layers - 1)])
    def forward(self,x):
        for layer in self.u_down:
            x = layer(x)
        return x

class AEup(nn.Module):
    def __init__(self,embedding_dim,num_heads,d_model,d_num_layers,output_dim):
        super(AEup, self).__init__()
        self.u_up = nn.ModuleList(
            [VaeEncoder(embedding_dim, num_heads, d_model, drop=0.1) for _ in
             range(d_num_layers - 1)])
        self.output = nn.Linear(embedding_dim, output_dim)
    def forward(self,x):
        for layer in self.u_up:
            x = layer(x)
        out = self.output(x)
        return out

class Time_AE(nn.Module):
    def __init__(self,input_dim,embedding_dim,num_heads,d_model,e_num_layers=2,d_num_layers=2,drop=0.1):
        super(Time_AE, self).__init__()
        self.embedding = DataEmbedding(input_dim,embedding_dim)
        self.encoder = AEdown(embedding_dim,num_heads,d_model,e_num_layers)
        self.decoder = AEup(embedding_dim,num_heads,d_model,d_num_layers,output_dim=input_dim)
        self.silu = nn.SiLU()

    def forward(self,input):
        x = self.embedding(input)
        x = self.encoder(x)
        x = self.decoder(x)
        return x