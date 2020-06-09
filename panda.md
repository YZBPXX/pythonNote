#### panda
read_csv 参数解析
- header=0 表示第0行为列索引, 这样names会替换原来的列索引
- names 设置列索引
- index_col 指定那一列为行索引,可以指定多列,默认为None 系统自动添加
- usecols 指定使用哪些列, 如usecols=[0,1,2,3,n-1] 使用全部
- nrows 指定读取的行
- parse_dates=True 将日期解析为日期格式
- sep 设置分割符
- thousands=',' 没3位一分割
----
pandas 小技巧
- 返回行: 使用pd.loc[] 返回一行类型为Series 可以用pd.loc[[]] 返回一个dateframe类型
- 按行读取:  且向字典添加键值对, 和行的值, row.index
```python 
for index,row in data.iterrows():
    dic.update({index:row.values})
```
- 转置: pd.T
