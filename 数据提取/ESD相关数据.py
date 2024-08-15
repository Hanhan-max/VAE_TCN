import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取所有数据
yeWei_1 = pd.read_csv('../data/大数据分析数据/液位大于等于1000mm，触发ESD/CLZ_FCS0106_7低温分离器液位-1720533436963.csv')
yeWei_2 = pd.read_csv('../data/大数据分析数据/液位大于等于1000mm，触发ESD/CLZ_FCS0106_7低温分离器液位-1720533448419.csv')
yeWei_3 = pd.read_csv('../data/大数据分析数据/液位大于等于1000mm，触发ESD/CLZ_FCS0106_7低温分离器液位-1720533458241.csv')
wenDu_1 = pd.read_csv('../data/大数据分析数据/温度小于等于-5℃，出发ESD/CLZ_FCS0106_8脱甲烷塔底泵进口凝液温度-1720533213896.csv')
wenDu_2 = pd.read_csv('../data/大数据分析数据/温度小于等于-5℃，出发ESD/CLZ_FCS0106_8脱甲烷塔底泵进口凝液温度-1720533224525.csv')
wenDu_3 = pd.read_csv('../data/大数据分析数据/温度小于等于-5℃，出发ESD/CLZ_FCS0106_8脱甲烷塔底泵进口凝液温度-1720533248778.csv')
yaLi_1 = pd.read_csv('../data/大数据分析数据/PT-02023压力小于等于5.4MPa，触发ESD/CLZ_FCS0106_7F0201出口管线压力-1720532694762.csv')
yaLi_2 = pd.read_csv('../data/大数据分析数据/PT-02023压力小于等于5.4MPa，触发ESD/CLZ_FCS0106_7F0201出口管线压力-1720532707770.csv')
yaLi_3 = pd.read_csv('../data/大数据分析数据/PT-02023压力小于等于5.4MPa，触发ESD/CLZ_FCS0106_7F0201出口管线压力-1720532718762.csv')

yaLi_4 = pd.read_csv('../data/大数据分析数据/PT-03018压力小于等于1.5MPa,触发ESD/PT-03018压力小于等于1.5MPa,触发ESD/CLZ_FCS0106_7脱甲烷塔顶压力-1720666392073.csv')
yaLi_5 = pd.read_csv('../data/大数据分析数据/PT-03018压力小于等于1.5MPa,触发ESD/PT-03018压力小于等于1.5MPa,触发ESD/CLZ_FCS0106_7脱甲烷塔顶压力-1720666408095.csv')
yaLi_6 = pd.read_csv('../data/大数据分析数据/PT-03018压力小于等于1.5MPa,触发ESD/PT-03018压力小于等于1.5MPa,触发ESD/CLZ_FCS0106_7脱甲烷塔顶压力-1720666422398.csv')


# 数据合一
yeWei = pd.concat([yeWei_1,yeWei_2,yeWei_3],axis=0).drop(['对象名','点名','描述'],axis=1)
wenDu = pd.concat([wenDu_1,wenDu_2,wenDu_3],axis=0).drop(['对象名','点名','描述'],axis=1)
yaLi = pd.concat([yaLi_1,yaLi_2,yaLi_3],axis=0).drop(['对象名','点名','描述'],axis=1)
yaLi_03 = pd.concat([yaLi_4,yaLi_5,yaLi_6],axis=0).drop(['对象名','点名','描述'],axis=1)


# 使用 outer join 进行拼接
merged_df = pd.merge(yeWei.loc[:,["时间",'值']], wenDu.loc[:,["时间",'值']], on='时间', how='outer')
## 拼接数据，按照每一分钟12个点拼接
yaLi.iloc[:40000].to_csv('../data/processe_data/yali_train.csv',index=False)
yaLi.iloc[40000:].to_csv('../data/processe_data/yali_test.csv',index=False)
yaLi_03.to_csv('../data/processe_data/yali_03.csv',index=False)
wenDu.to_csv('../data/processe_data/wenDu.csv',index=False)
yeWei.to_csv('../data/processe_data/yeWei.csv',index=False)
merged_df.to_csv('../data/processe_data/merged_ywWai_wenDu.csv',index=False)