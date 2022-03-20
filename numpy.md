### numpy 实用操作
- 初始化矩阵 m_1 = np.matrix([[1,2],[2,2]])
- 用随机数创建一个举证 x=np.random.rand(10,1,28,28) 10个长28宽28深度为1的数据
- X.reshape(10,-1) 展开为10行,并自动计算列
- X.transpose(0,3,2,1) 改变轴的顺序，即：第一维还是第一维，第二维变第四维，第三维不变，第四维变第二维
- np.meshgrid(x, y) 根据x、y向量(画线）构建一个网格，返回网格点的x坐标，y坐标
- ravel 功能与flatten一样 只是ravel是改变索引，flatten重新生成一个
### numpy 小知识
- 访问元素时最好不用for语句，会很慢（局部性原理？）,使用im2col 将要处理的小矩阵变成一列
#### scipy-mode
>scipy是建立在numpy矩阵基础上对齐进行操作的库
----
解决线性规划问题(scipy.optimize(最优).linprog(线性规划))
问题描述为  
$$
\min_{x} c^tx \\
s.t \begin{cases}
A·x <=b \\
Aeq·x=beq \\
lb<=x<=ub
\end{cases}
$$
```
c表示x的n纬列向量, A, Aeq表示系数矩阵, lb, ub表示上下届
```
python求解方法
```python
#求的是最小值, 如果求最大值可以全体取反, 结果就是最大值
from scipy import optimize
import numpy as np

#确定c,A,b,Aeq,beq
c = np.array([2,3,-5])
A = np.array([[-2,5,-1],[1,3,1]])
b = np.array([-10,12])
Aeq = np.array([[1,1,1]])
beq = np.array([7])
X0_bounds = [0,None]
X1_bounds = [0,None]
X3_bounds = [0,None]

res = optimize.linprog(-c,A,b,Aeq,beq,bounds=(X0_bounds,X1_bounds,X3_bounds))

#求解
print(res)
```
返回值解析:
- fun:最优值
- x:最优解
- con:约束等式的残差
- slack:松弛变量的值
- status:表示算法退出状态的整数。
 	- 0 ：优化成功终止。
 	- 1 ：达到迭代限制。
 	- 2 ：问题似乎不可行。
 	- 3 ：问题似乎是无限的。
 	- 4 ：遇到数字困难。
- nit:执行的迭代总数
---
计算平均值, 方差, 标准差
numpy :
```python
x = [1,2,3,4]
mean = np.mean(x)#均值
var = np.var(x)#方差
std = np.std(x,ddof=1)#标准差
```
