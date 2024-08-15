import numpy
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from Model.MyMseLoss import MyMseLoss
from Model.TimePredicate.TrendPredicater import ATime
from Model.VAE.VAE import Time_AE
from tools.getData import getOneData

path ='../data/processe_data/yeWei_4step.csv'
input_len = 720
predicte_len=360
c_in = 1
c_out = 1

train_dataloader, eval_dataloader, test_dataloader, mystand = getOneData(path,input_len,predicte_len,1,1,128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

model = Time_AE(input_dim=1,embedding_dim=4,num_heads=4,d_model=256)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

timemodel = ATime(c_in,c_out,input_len,predicte_len,8,1,16,8,25,3)
state_dict = torch.load('../save/time/model_outlayer_80.pth')
# 将state_dict加载到模型中
timemodel.load_state_dict(state_dict)
timemodel = timemodel.to(device)
criterion = MyMseLoss()

epochs = 40
for epoch in tqdm(range(epochs)):
    model.train()
    timemodel.eval()
    losses_train = []
    for i, item in enumerate(train_dataloader):
        input = item[0][:, :input_len].to(device)
        traget = item[0][:, input_len:].to(device)
        input = timemodel(input)
        out= model(input)
        loss = criterion(out, traget)
        losses_train.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tqdm.write(f"\t Epoch {epoch + 1} / {epochs}, Loss: {(sum(losses_train) / len(losses_train)):.6f}")
    if (epoch + 1) % 2 == 0:
        model.eval()
        with torch.no_grad():
            losses = []
            for i, item in enumerate(eval_dataloader):
                input = item[0][:, :input_len].to(device)
                traget = item[0][:, input_len:].to(device)
                input = timemodel(input)
                out = model(input)
                loss = criterion(out, traget)
                losses.append(loss)
            print(f'\nEpoch [{epoch + 1}/{epochs}], Eval_Loss: {sum(losses) / len(losses) :.4f}')
    scheduler.step()
#
result = []
# vae_state_dict = torch.load('../save/AE/aoutencoder_state_dict.pth')
# # 将state_dict加载到模型中
# model.load_state_dict(vae_state_dict)

true = []
for i, item in enumerate(test_dataloader):
    input = item[0][:, :input_len].to(device)
    traget = item[0][:, input_len:]
    input = timemodel(input)
    out = model(input)
    result.append(out.detach().cpu().numpy())
    true.append(traget)
result_np = np.concatenate(result, axis=0)
true_np = np.concatenate(true, axis=0)

if mystand != None:
    result_np = mystand.inverse_transform(result_np)
    true_np = mystand.inverse_transform(true_np)
numpy.save('../save/AE/yeWei_AE.npy',result_np)
numpy.save('../save/AE/yeWei_true_AE.npy',true_np)
torch.save(model.state_dict(), '../save/AE/aoutencoder_state_dict.pth')
