import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import numpy as np


def extract_trend_with_varying_window(time_series, max_window_size=12):
    """
    Extracts the trend from a time series using a varying window size, starting from 1 up to max_window_size,
    then remaining constant, and finally decreasing again. No padding is used.
    :param time_series: numpy array of time series data.
    :param max_window_size: the maximum window size to compute the trend.
    :return: a numpy array containing the extracted trend.
    """
    # Ensure the time series is a numpy array
    time_series = np.asarray(time_series)
    # Check if max_window_size is valid
    if max_window_size <= 0 or max_window_size > len(time_series):
        raise ValueError(
            "Max window size must be greater than 0 and less than or equal to the length of the time series.")
    # Initialize the trend array
    trend = np.zeros_like(time_series)
    # Calculate the trend with a varying window size
    for i in range(len(time_series)):
        window_size = min(i + 1, max_window_size, len(time_series) - i)
        window = time_series[i - window_size + 1:i + 1]
        trend[i] = np.mean(window)

    return trend




if __name__ == '__main__':
    data  = pd.read_csv('../../data/processe_data/yeWei.csv')['值']
    trend = extract_trend_with_varying_window(data.values,12)
    trend_2  = extract_trend_with_varying_window(trend,4)
    trend_3 = extract_trend_with_varying_window(trend_2, 4)

 # 绘制第n步结果对比
    plt.plot(range(len(data)), data.values)
    plt.plot(range(len(trend)), trend)
    plt.plot(range(len(trend_2)), trend_2)
    # plt.plot(range(len(trend_3)), trend_3)


    # 添加图例
    plt.legend()
    # 显示图表
    plt.show()
    data['trend'] = trend_2
    data.to_csv('../../data/trend/yeWei_trend.csv')