import numpy as np
from matplotlib import pyplot as plt

prediction = np.load('yeWei_4step16_1_16_8_25_3_100.npy')
true = np.load('yeWei_4steptrue16_1_16_8_25_3_100.npy')
# prediction = np.load('../out/yaLi_result_withNorm_TCN&LSTM3.npy')
# true = np.load('../out/yaLi_true_withNorm.npy')
# 绘制单次长预测结果对比
plt.plot(range(360), prediction[5,:,0])
plt.plot(range(360), true[5,:,0])
# 添加图例
plt.legend()
# 显示图表
plt.show()

#绘制第n步结果对比
plt.plot(range(len(prediction)), prediction[:,0,0])
plt.plot(range(len(prediction)), true[:,0,0])
# 添加图例
plt.legend()
# 显示图表
plt.show()