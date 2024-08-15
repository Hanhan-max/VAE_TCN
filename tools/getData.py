import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tools.tools import getWindows, StandardScaler


def getOneData(path,input_len,predicte_len,step_len,batch_size=32,isstn = True):
    # 获取数据
    data = pd.read_csv(path)["值"]
    if isstn:
        mystn = StandardScaler()
        mystn.fit(data)
        data = mystn.transform(data)
    else:
        mystn = None
    # 划分窗口
    wind = getWindows(input_len, predicte_len, step_len, data)
    wind = np.expand_dims(wind, axis=-1)
    seg_train = int(len(wind) * 0.7)
    seg_test = int(len(wind) * 0.8)
    train_wind = torch.tensor(wind[:seg_train, :], dtype=torch.float32)
    eval_wind = torch.tensor(wind[seg_train:seg_test, :], dtype=torch.float32)
    test_wind = torch.tensor(wind[seg_test:, :], dtype=torch.float32)
    # train_wind = torch.tensor(wind[18000:38000, :], dtype=torch.float32)
    # test_wind = torch.tensor(wind[:18000, :], dtype=torch.float32)
    # eval_wind = torch.tensor(wind[38000:, :], dtype=torch.float32)
    # 创建 TensorDataset
    train_dataset = TensorDataset(train_wind)
    eval_dataset = TensorDataset(eval_wind)
    test_dataset = TensorDataset(test_wind)

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader,eval_dataloader,test_dataloader,mystn

def getEDData(path,input_len,predicte_len,step_len,batch_size=32,isstn = True):
    # 获取数据
    data = pd.read_csv(path)["值"]

    seg_train = int(len(data) * 0.7)
    seg_test = int(len(data) * 0.8)

    train_data = data[:seg_train]
    eval_data = data[seg_train:seg_test]
    test_data = data[seg_test:]

    if isstn:
        train_stn = StandardScaler()
        eval_stn = StandardScaler()
        test_stn = StandardScaler()

        train_stn.fit(train_data)
        eval_stn.fit(eval_data)
        test_stn.fit(test_data)

        train_data = train_stn.transform(train_data)
        eval_data = eval_stn.transform(eval_data)
        test_data = test_stn.transform(test_data)

    else:
        test_stn = None

    # 划分窗口
    train_wind = getWindows(input_len, predicte_len, step_len, train_data)
    train_wind = np.expand_dims(train_wind, axis=-1)

    eval_wind = getWindows(input_len, predicte_len, step_len, eval_data)
    eval_wind = np.expand_dims(eval_wind, axis=-1)

    test_wind = getWindows(input_len, predicte_len, step_len, test_data)
    test_wind = np.expand_dims(test_wind, axis=-1)

    train_wind = torch.tensor(train_wind, dtype=torch.float32)
    eval_wind = torch.tensor(eval_wind, dtype=torch.float32)
    test_wind = torch.tensor(test_wind, dtype=torch.float32)

    # 创建 TensorDataset
    train_dataset = TensorDataset(train_wind)
    eval_dataset = TensorDataset(eval_wind)
    test_dataset = TensorDataset(test_wind)

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, eval_dataloader, test_dataloader, test_stn

