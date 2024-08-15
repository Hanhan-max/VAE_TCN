import numpy as np
import pandas as pd


# 窗口划分，将数据划分为滑动窗口大小，输入为X，输出为Y,不足窗口的数据删除
# 参数 ： X输入窗口大小 Y输出窗口大小（预测长度） step滑动步长
# 输出：
import torch


def getWindows(X,Y,step,data):
    windows = []
    start = 0
    while start+X+Y <= data.shape[0]:
        window = data.iloc[start:start+X+Y]
        windows.append(window)
        start = start+step
    return np.array(windows)



class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



if __name__ == '__main__':
    data = pd.read_csv('../../Informer2020/data/yeWei/yeWei.csv')["值"]


    wind = getWindows(480,120,1,data)
    print(1)