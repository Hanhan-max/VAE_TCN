import numpy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from Model.MyMseLoss import MyMseLoss, MutiLoss
from Model.TimePredicate.TrendPredicater import ATime
from tools.fit.train import trainer, test
from tools.getData import getOneData

path = '../data/processe_data/yeWei_4step.csv'
c_in =1
c_out =1
input_len = 720
decode_len = 120
predicte_len = 360
step_len = 1
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("输入长度:", input_len, "预测长度:", predicte_len, "滑动步长:", step_len)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_dataloader,eval_dataloader,test_dataloader,mystand = getOneData(path=path,input_len=input_len,predicte_len=predicte_len,step_len=step_len,batch_size=128,isstn =True)
# model = ATime(c_in,c_out,input_len,predicte_len,8,1,16,8,25,3)
model = ATime(c_in,c_out,input_len,predicte_len,8,1,16,8,25,3)

model = model.to(device)
criterion = MyMseLoss()
# criterion = MutiLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

## 非decoder训练
# num_epochs = 40
# trainer(num_epochs, train_dataloader, eval_dataloader,model,criterion,optimizer,input_len,device)
# result_np,true_np = test(model,test_dataloader,input_len,device)
## decoder训练
num_epochs = 100
trainer(num_epochs, train_dataloader, eval_dataloader,model,criterion,optimizer,input_len,device,scheduler)
result_np,true_np = test(model,test_dataloader,input_len,device)
torch.save(model.state_dict(), '../save/time/model_outlayer_80.pth')
if mystand != None:
    result_np = mystand.inverse_transform(result_np)
    true_np = mystand.inverse_transform(true_np)
numpy.save('../save/time/yeWei_4step_80.npy', result_np)
numpy.save('../save/time/yeWei_4steptrue_80.npy', true_np)
