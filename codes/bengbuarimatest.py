# from os import error
# import sys
import pandas as pd
# from pandas.core import series
# import statsmodels
import numpy as np
from scipy import stats
# from numpy.lib.type_check import real
# import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# from scipy import stats
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']  # 用来显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来显示负号


# 加载读取数据
def get_data():
    # df = pd.read_csv(r'..\\datas\\211009.xls', encoding='unicode_escape')
    # df = pd.read_csv(r'..\\datas\\211009.xls')
    df = pd.read_excel(r'..\\datas\\211009.xlsx')

    # 最原始的data
    data = df['入口氮氧化物']

    # 数据数量，243000+
    n = np.size(data)

    # data_d取data数据的前80%，后面的ARIMA模型用的就是80%的data数据来训练
    # 后20%数据来预测
    # print("全部数据量： %d" % n)
    # print("准备提取80%%的训练数据量： %d" % (int(n * 0.8)))

    # 取训练数据
    data_d = data[:int(n * 0.8)]

    # 取测试数据
    data_test = data[int(n * 0.8):n]

    # 向下移动50，以此进行超前50步的预测，值从51开始
    data_down50 = data_d.shift(50)
    # print(data_d)
    # print(data_test)

    # 80%data difference前百分之八十数据的差分
    data_diff = data_d.diff(1).dropna()
    # print(data_diff)

    return data_d, data_diff


# boxcox transform俩图画在一张上observe#boxcox变换
def boxcox_transformation():
    _, data_diff = get_data()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    # x = stats.loggamma.rvs(5, size=500) + 5
    prob = stats.probplot(data_diff, dist=stats.norm, plot=ax1)

    ax1.set_xlabel('')
    ax1.set_title('Probplot against normal distribution')
    plt.show()

    ax2 = fig.add_subplot(212)
    data_diff_positive = data_diff + 100
    bc_diff, lamda = stats.boxcox(data_diff_positive)  # diff_data的boxcox变换
    prob = stats.probplot(bc_diff, dist=stats.norm, plot=ax2)
    ax2.set_title('Probplot after Box-Cox transformation')
    print('lamda', lamda)  # boxcox的lamda值
    plt.show()

    # np.reshape(bc_diff)
    # print(bc_diff)
    # print(data_d)


# 观察 数据的稳定性
def stationary_observation():
    data_d, data_diff = get_data()
    plt.plot(data_d, color='r', label="原数据")
    plt.plot(data_diff, color=(0, 0, 1), label="差分处理后数据")
    plt.xlabel("个数")  # x轴命名表示
    plt.ylabel("mg/m3")  # y轴命名表示
    plt.title("数据平稳性处理前后对比图")
    plt.legend()  # 增加图例
    plt.show()

    # data_d.plot()
    # data_diff.plot()
    # bc_diff.plot()
    # plt.show()


"""
def stationary_detect():
    from statsmodels.tsa.stattools import adfuller
    adfuller(data_diff)  #差分序列的ADF平稳性检验结果，重要，检测出拒绝原不平稳的假设
    ，差分后数据是平稳的

    from statsmodels.stats.diagnostic import acorr_ljungbox
    print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(data_diff, lags=1))

    from statsmodels.tsa.stattools import adfuller
    dwgtest = adfuller(data_diff)
    print(u'差分序列的单位根检验结果为：', dwgtest)

    from statsmodels.stats.diagnostic import acorr_ljungbox
    xgxtest = acorr_ljungbox(data_diff, lags=20)
    print(u'差分序列的相关性检验结果为：',
          xgxtest)  #第一个数统计值，第二个数p值，结果p较小，拒绝原假设（没有相关性），所以序列有相关性


stationary_detect()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


#画ACF、PACF图
def plot_acf_pacf(series, lags):
    pacf = plot_pacf(series, lags=lags)
    plt.title('PACF')  #画ACF和PACF图，判断AR(p)和MA(q)的阶数p、几节后截尾，值为几
    plt.show()
    acf = plot_acf(series, lags=lags)
    plt.title('ACF')
    plt.show()


series = data_diff
lags = 20
#plot_acf_pacf(series, lags)

#使用AIC、BIC最小准则确定p\\q,当pq阶数较小时，
# 可用这种遍历的暴力解法，返回一个元组，分别为pq值
import statsmodels.tsa.stattools as st


def pq_decide():
    #AIC评估统计模型复杂度和衡量统计模型“拟合”优良性的
    model = st.arma_order_select_ic(data_diff,
                                    max_ar=6,
                                    max_ma=6,
                                    ic=['aic', 'bic', 'hqic'])
    print(u'arima模型pq取值为：', model.bic_min_order)


#pq_decide()
#（1，2），80%（5，3）

#拟合ARIMA或ARMA模型
data_d = np.array(data_d)
import statsmodels.api as sm

data_d[np.isnan(data_d)] = 0
data_d[np.isinf(data_d)] = 0
from statsmodels.tsa.arima_model import ARMA

model_arma = ARMA(data_d, order=(5, 3))
#arima参数调整，寻优
#order = model.bic_min_order

result_arma = model_arma.fit(disp=-1, method='css')
print(u'输出result_arma：', result_arma.summary())
pred = result_arma.predict()
#当使用原始数据时，拟合ARIMA模型
#from statsmodels.tsa.arima_model import ARIMA
#model_arima = ARIMA(data_d, order=(6,1,3))#
#result_arima = model_arima.fit(disp = -1, method = 'css')
#print(u'输出result_arima：',result_arima.summary())
#pred = result_arima.predict(start=6,end= n+1 )
#残差检验
# 如果残差是白噪声序列，说明时间序列中有用的信息已经被提取完毕了，剩下的全是随机扰动，是无法预测和使用的。
# 残差序列如果通过了白噪声检验，则建模就可以终止了，因为没有信息可以继续提取。
# 如果残差如果未通过白噪声检验，说明残差中还有有用的信息，需要修改模型或者进一步提取。
#resid = result_arima.resid
predict_arma = pred
resid = result_arma.resid
from statsmodels.tsa.stattools import adfuller

dwgtest = adfuller(resid)
print(u'差分序列的单位根检验结果为：', dwgtest)


#print(resid)
#writer = pd.ExcelWriter()
#resid.to_excel(r'\\residue(5,3).xlsx')
#resid.to_excel(r'\\residue.xls')
#残差 QQ图？
def plot_resid_detect():
    from statsmodels.graphics.api import qqplot
    qqplot(resid, line='q', fit=True)
    plt.show()


plot_resid_detect
#PEP8规范写代码
# walk forward over time steps in test

#values = dataset.values

#history = [values[i] for i in range(len(values))]

predictions = list()

#test_values = test.values

#for t in range(len(test_values)):

# fit model

#model = ARIMA(history, order=(7,0,0))

#model_fit = model.fit(trend='nc', disp=0)

# make prediction

#yhat = model_fit.forecast()[0]

#predictions.append(yhat)

#history.append(test_values[t])

#rmse = sqrt(mean_squared_error(test_values, predictions))

#print('%s-%s (%d values) RMSE: %.3f' % (years[0], year, len(values), rmse))
# qq图中：如果是正态分布则为一条直线，即红线。结果大致符合白噪声，我们的结果就是这样，说明通过了残差检验，残差中没有可用信息
#ARIMA中对于残差的继续建模可以停止了？？？
# 白噪声检验除了qq图还可以使用DW检验法（DW：检验残差序列是否具有自相关性，只适用一一阶自相关；
# 多阶自相关可用LM检验）
#我的结果中和红线拟合情况很好，
import statsmodels.api as sm
#print(sm.stats.durbin_watson(resid.values))

#预测
#pred = result_arima.predict( )
#从训练集第start-1个开始预测，预测好整个训练集后，向后预测+n个
#print(u'pred长',len(pred))
#print(u'pred[-50:0]',pred[-50:])
# 第十二步：将预测的平稳值还原为非平稳序列
result_fina = predict_arma
#预测结果转化为差分前的原始值
#result_fina = data.shift(1)+pred[:]
#result_fina = data.shift(-1)+pred[:]
# 如果还取了对数，进行如下操作
# result_log_rev = np.exp(result_fina)
# result_log_rev.dropna(inplace=True)
#result_fina = np.array(pred[0:-50]) + (np.array(data.shift(1)))
#pred[1]=data_d[1]
#result_fina[2]=data_d[2]
#result_fina[3]=data_d[3]
#result_fina[4]=data_d[4]
# result_fina[1]=data[1]
# result_fina[2]=data[2]
# result_fina[3]=data[3]
# result_fina[4]=data[4]
#result_fina[23386]=data[23386]
predict = result_fina
real11 = data.shift(1) + data_diff
real = data_d
print('predict', np.size(predict))
print('real', np.size(real))
plt.plot(predict, color='r', label="predict")
plt.plot(real, color=(0, 0, 1), label="real")
plt.xlabel("个数")  #x轴命名表示
plt.ylabel("mg/m3")  #y轴命名表示
plt.title("SCR反应器入口NOX实际值与预测值的arima预测模型对比图")
plt.legend()  #增加图例
plt.show()
#print(result_fina)

error1 = real - predict
plt.plot(error1, color='r', label="error")
plt.xlabel("个数")  #x轴命名表示
plt.ylabel("mg/m3")  #y轴命名表示
plt.title("SCR反应器入口NOX实际值与预测值偏差图")
plt.legend()  #增加图例
plt.show()

plt.plot(resid, color='b', label="residue")
plt.xlabel("个数")  #x轴命名表示
plt.ylabel("mg/m3")  #y轴命名表示
plt.title("SCR反应器入口NOX实际值与预测值残差图")
plt.legend()  #增加图例
plt.show()

from collections import Counter
#print('residue: ',Counter(resid))

##决策树
import sklearn.tree as tree
from sklearn.model_selection import train_test_split

y = resid
x = df
nx = np.size(x)  #数据数量，243000+
print('x数量', nx)
x_d = data[:int(nx * 0.8)]  #data_d取data数据的前80%，
后面的ARIMA模型用的就是80%的data数据
print('80%x', x_d)

#多个变量x与残差y的相关性分析


#spearman相关度分析spearman person mda
def correlation_analyse():
    from scipy import stats
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(x, y)
    plt.grid()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(x, y)
    plt.grid()


# 负线性相关

#correlation_analyse()
#print(df)
# print(df.corr())
# print(df.corr('spearman'))
# print(df.corr('kendall'))
from sklearn import metrics
from sklearn.metrics import r2_score
#r2和mes
#predict2=predict[:int(n*0.8)]
r2 = r2_score(real, predict)
print('R2: %.3f' % r2)
mes = metrics.mean_squared_error(real, predict)
print('MSE: %.3f' % mes)

mes34 = metrics.mean_squared_error(real, predict)

"""

if __name__ == '__main__':
    # get_data()
    boxcox_transformation()
    stationary_observation()
