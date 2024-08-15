import pandas as pd
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

data = pd.read_csv('../data/processe_data/yeWei.csv')['值']


# 将 DataFrame 转化为 NumPy 数组
numpy_array = data.values

# 将 NumPy 数组转化为 PyTorch 张量
x = torch.tensor(numpy_array, dtype=torch.float32)

# 定义池化的窗口大小和步长
kernel_size = 20
stride = 1

# 计算填充的大小，保持数据长度不变
padding = (kernel_size - 1) // 2

# 对数据进行填充
x_padded = F.pad(x, (padding, padding), mode='constant', value=0)

# 应用平均池化
x_pooled = F.avg_pool1d(x_padded.unsqueeze(0), kernel_size=kernel_size, stride=stride)

# 去掉多余的维度
x_pooled = x_pooled.squeeze(0)

# 绘制单次长预测结果对比
plt.plot(range(len(x_pooled)), x_pooled)
# 添加图例
plt.legend()
# 显示图表
plt.show()