![Alt](https://repobeats.axiom.co/api/embed/455724986b31342ce99ca637fbb0b8b72f3e07d4.svg "Repobeats analytics image")

本研究用于进行时序数据的分析。

常见的时间序列模型有`AR自回归模型`，`MA移动平均模型`和`ARMA模型`等模型。本文主要介绍`ARMA模型`[^1]。


Box-Cox变换是Box和Cox在1964年提出的一种广义幂变换方法，是统计建模中常用的一种数据变换，用于连续的响应变量不满足正态分布的情况。

- Box-Cox变换之后，可以一定程度上减小不可观测的误差和预测变量的相关性。
- Box-Cox变换的主要特点是引入一个参数，通过数据本身估计该参数进而确定应采取的数据变换形式，Box-Cox变换可以明显地改善数据的正态性、对称性和方差相等性，对许多实际数据都是行之有效的。
- Box-Cox变换即将数据转换为满足正态分布的数据


[^1]: https://medium.com/auquan/time-series-analysis-for-finance-arma-models-21695e14c999


-----


#### sns.displot参数如下
```python`
sns.distplot(a, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None)
```

bins: ?
hist: 控制是否显示条形图, 默认为True.
kde: 控制是否显示核密度估计图, 默认为True.
rug: 控制是否显示观测的小细条（边际毛毯）默认为false.
fit: 设定函数图像, 与原图进行比较.
norm_hist：若为True, 则直方图高度显示密度而非计数(含有kde图像中默认为True).
通过hidt和kde参数调节是否显示直方图和核密度估计(hist, kde 默认均为True).
axlabel: 设置x轴的label.
label : 没有发现什么作用.
ax: 图片位置.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()  # 切换到sns的默认运行配置
import warnings
warnings.filterwarnings('ignore')

x=np.random.randn(100)

sns.distplot(x)
```

![](https://img-blog.csdnimg.cn/20210416155940508.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REeHVleGk=,size_16,color_FFFFFF,t_70)


```python
sns.distplot(x, kde=False)
```

![](https://img-blog.csdnimg.cn/20210416160036454.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REeHVleGk=,size_16,color_FFFFFF,t_70)


```python
# norm_hist
fig, axes=plt.subplots(1,2)
sns.distplot(x, norm_hist=True, kde=False, ax=axes[0]) # 左图
sns.distplot(x, kde=False, ax=axes[1]) # 右图
```

![](https://img-blog.csdnimg.cn/20210416160109226.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REeHVleGk=,size_16,color_FFFFFF,t_70)


```python
fig, axes = plt.subplots(1, 3) # 创建一个1行3列的图片
sns.distplot(x, ax=axes[0]) # ax=axex[0] 表示该图片在整个画板中的位置
sns.distplot(x, hist=False, ax=axes[1])  # 不显示直方图
sns.distplot(x, kde=False, ax=axes[2])  # 不显示核密度
````

![](https://img-blog.csdnimg.cn/20210416160126231.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REeHVleGk=,size_16,color_FFFFFF,t_70)

#### stats.probplot参数如下
```python
scipy.stats.probplot(x, sparams=(), dist='norm', fit=True, plot=None, rvalue=False)
```
- x：array_like,从哪个样本/响应数据probplot创建情节。

- sparams：tuple, 可选参数
Distribution-specific形状参数(形状参数加上位置和比例)。

- dist：str 或 stats.distributions instance, 可选参数。分发或分发函数名称。对于正常概率图，默认值为‘norm’。看起来像stats.distributions实例的对象(即它们具有一个ppf方法)也被接受。

- fit：bool, 可选参数。如果为True(默认值)，则将least-squares回归(best-fit)行拟合到样本数据。

- plot：object, 可选参数
