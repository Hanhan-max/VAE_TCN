import pandas as pd

def data_chouxi(file_path):
    # 加载上传的数据'
    data = pd.read_csv(file_path)

    # 将'时间'列转换为日期时间格式
    data['时间'] = pd.to_datetime(data['时间'])

    # 创建一个副本并将时间精度设置为分钟
    minute_data = data.copy()
    minute_data['时间'] = minute_data['时间'].dt.floor('T')

    # 对相同时间的数据求平均值
    # 使用groupby和transform将相同时间的值替换为相同的值
    minute_data['值'] = minute_data.groupby('时间')['值'].transform('first')
    data['抽稀值'] = minute_data['值']
    # 获取列的列表
    cols = list(data.columns)

    # 交换列A和列B的位置
    cols[1], cols[2] = cols[2], cols[1]

    # 重新排列DataFrame的列
    data = data[cols]
    return data

def get_span_data(path,step):
    data = pd.read_csv(path)
    extracted_data = data.iloc[::step]
    return extracted_data
if __name__ == '__main__':
    path = '../data/processe_data/yeWei.csv'
    # data = data_chouxi(path)
    data = get_span_data(path,4)
    data.to_csv('../data/processe_data/yeWei_4step.csv', index=False)