科学计算的几个包: scipy, numpy, panda
### panda
>可以读取不同的文件(csv, xlsm), 并且计算(类似数据库,在panda是DataFrame类)
以读取csv文件,和处理为例子
```python
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 读取数据
features = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']#定义需要的列, 可以是表中任意几列
dc_listings = pd.read_csv(r'D:\codes_jupyter\数据分析_learning\课件\05_K近邻\listings.csv', engine='python')#返回一个DataFrame对象
dc_listings = dc_listings[features]#返回一个只包含所选的列

# 对price列进行一定的处理，使其变成float型
dc_listings['price'] = dc_listings.astype(float)

# 对缺失值进行处理,删除有缺失值的数据
dc_listings = dc_listings.dropna()

# 归一化
dc_listings[features] = MinMaxScaler().fit_transform(dc_listings)

# 标准化
# dc_listings[features] = StandardScaler().fit_transform(dc_listings)

print(dc_listings.shape)
dc_listings.head()
```
```python

import pandas as pd
from sklearn.preprocessing import MinMaxScaler #归一化
from sklearn.preprocessing import StandardScaler #标准化

a=['a','b'] #定义列名
df = pd.read_csv('./test.csv',sep='\t',names=a)
df['a']=df['a'].astype(float)#转化数据类型
df = df.dropna() #删除确实数据的字段
df[a] = StandardScaler().fit_transform(df)#得到标准化结果
print(df)
```
### Matplotlib
- 使用pyplot包
	- 使用pyplot.plot([x1,x2...],[y1,y2...],color)#颜色编码可以百度, 如果多次调用会在原来的基础上多次绘制
	- title("string")
	- savefig(path)
### imageio 绘制动态图像
- 使用imageio.read(1.png)将图片转换为流,并且添加到列表哦li里
- 使用imageio.mimsace(path,li,'GIF',duration=0.5)将图片保存起来
```python
import imageio
from matplotlib import pyplot as pl

li=[]
pl.plot([1,2],[1,2],'.r')
pl.savefig('1.png')
li.append(imageio.imread('1.png'))
pl.plot([5,6],[5,6],'*b')
pl.savefig('1.png')
li.append(imageio.imread('1.png'))
imageio.mimsave('~/Desktop/test.gif',li,'GIF',duration=1)
```

