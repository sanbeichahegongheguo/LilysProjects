#import pandas as pd
#import numpy as np#########加列里面包含随机森林预测过程
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False
# plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
from os import error
import sys
#import matplotlib
import pandas as pd
from pandas.core import series
import statsmodels
import seaborn as sns
#import matplotlib.pyplot as plt
import xlrd
from scipy import stats
import matplotlib.pyplot as plt
from arimatry1 import error1
from arimatry1 import predict
from arimatry1 import data 
dataa=error1
arima_predict=predict
arima_real=data
#sys.path.append('c:\\anaconda\lib\site-packages')
import statsmodels
#数据读取和处理
#matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
df1 = pd.read_excel(r'C:\Users\tosal\Documents\nanjingtianfushijian\cangzhou20201213(2).xls',index_col=0, parse_dates=True)
df1 = pd.DataFrame(df1)
df_y=df1['沧州电厂_1A脱硝入口NOX折算浓度 ']
df_y1=df_y.shift(1)#向下移动是y（t-1）
df_y2=df_y.shift(2)
col_name=df1.columns.tolist()
col_name.insert(85,'y(t-1)')
df1 =df1.reindex(columns=col_name)
df1['y(t-1)']=df_y1
###################################
col_name.insert(86,'y(t-2)')
df1 =df1.reindex(columns=col_name)
df1['y(t-2)']=df_y2
###################################
col_name.insert(87,'偏差')
df1 =df1.reindex(columns=col_name)
df1['偏差']=dataa          #到这都没问题

df1.to_excel(r'C:\Users\tosal\Documents\nanjingtianfushijian\rf.xlsx')

import numpy as np
from numpy.lib.type_check import real

data_df=np.array(df1) #23386*79没问题
c=data_df.shape[1]#矩阵的列数，行数为[0]
print(c)
d=data_df.shape[0]#矩阵的行数，行数为[0]
print(d)
#data1 = pd.DataFrame(df)#作用在于把日期改了
#print(data1.shape[1])
##########################################################
x=data_df[3:d,0:(c-1)]
y=data_df[3:d,(c-1)] #把x和y导入清楚了
co=df1.corr()
#print('co',co) 
###################森林的训练集和测试集划分
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

# 训练随机森林解决回归问题
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200, random_state=0)
#x_f =(x/np.timedelta64(1, 'D')).astype(float)
#y_f =(y/np.timedelta64(1, 'D')).astype(float)
#x=x.astype(np.float)
#y=y.astype(np.float)
regressor.fit(x, y)
y_pred = regressor.predict(x)
#print(regressor.result)##这里有问题 怎么输出randomforest的预测结果呢？
# 评估回归性能 
# from sklearn import metrics
# print('score:',regressor.score(y,y_pred))
# print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred)) 
# print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
# print('Root Mean Squared Error:',
#       np.sqrt(metrics.mean_squared_error(y, y_pred)))
from sklearn import metrics
from sklearn.metrics import r2_score

r2=r2_score(y,y_pred)
print('randomforest_R2: %.3f' % r2 )
mes=metrics.mean_squared_error(y,y_pred)
print('randomforest_MSE: %.3f'% mes )

pp=predict[3:d] #是ARIMA的、反差分后的最终预测结果
rr=data[3:d] #真实值
y_end=pp+y_pred #两者结合预测出来的y


# from sklearn import metrics
# print('score:',regressor.score(rr,y_end))
# print('Mean Absolute Error:', metrics.mean_absolute_error(rr,y_end)) 
# print('Mean Squared Error:', metrics.mean_squared_error(rr,y_end))
# print('Root Mean Squared Error:',
#       np.sqrt(metrics.mean_squared_error(rr,y_end)))

hyr2 =r2_score(rr,y_end)
print('hybrid_R2: %.3f' % hyr2 )
hymes=metrics.mean_squared_error(rr,y_end)
print('hybrid_MSE: %.3f'% hymes )
# import sklearn.datasets as datasets
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.decomposition import PCA
# X_train, X_test, y_train, y_test = train_test_split(x,
#                                                     y,
#                                                     test_size=0.2,
#                                                     random_state=0)

# regr = RandomForestRegressor()
#RF效果图
#print('y_pred',np.size(y_pred))
#print('dataa',np.size(dataa))
plt.plot(y_pred,color='r',label="RF-error-predict")
plt.plot(dataa,color='g',label="errorreal")#dataa是arima后的真实偏差，
plt.xlabel("个数") #x轴命名表示
plt.ylabel("mg/m3") #y轴命名表示
plt.title("ariam模型残差的实际值与预测值的randomforest预测模型对比图") 
plt.legend()#增加图例
plt.show()
#混合效果图
#print('y_end',np.size(y_end))
#print('rr',np.size(rr))
plt.plot(y_end,color='b',label="hybrid_NOX_predict")
plt.plot(rr,color=(0,0,2),label="NOX_real")
plt.xlabel("个数") #x轴命名表示
plt.ylabel("mg/m3") #y轴命名表示
plt.title("SCR反应器入口NOX实际值与预测值的混合预测算法对比图") 
plt.legend()#增加图例
plt.show()
#print(result_fina)



df1['偏差']=error1