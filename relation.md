#### pearson
> 研究两个数据集是否在一条线上, 用来衡量定矩变量间的*线性关系*

$$
\rho_{X,Y}=\frac{cov(X,Y)}{\sigma_{X}\sigma_{Y}}
$$

#### spearman 
$$
r_s=1-\frac{6\sum\limits_{i=1}^nd_i^2}{n(n^2-a)} $$
$$d_i表示数据排序后是第几位(可降可升)$$

pearson 和 spearman 对比
- 连续数据, 正态分布, 线性相关. 用pearson合适
- 以上条件不满足就用spearman
- 定序数据也用spearman(感觉和第一点重复了)如:优,良,差.

小写|大写|latex
-|-|-
α|A|\alpha
β|B|\beta
γ|Γ|\gamma
δ|Δ|\delta
ϵ|E|\epsilon
ζ|Z|\zeta
ν|N|\nu
ξ|Ξ|\xi
ο|O|\omicron
π|Π|\pi
ρ|P|\rho
σ|Σ|\sigma
η|H|\eta
θ|Θ|\theta
ι|I|\iota
κ|K|\kappa
λ|Λ|\lambda
μ|M|\mu
τ|T|\tau
υ|Υ|\upsilon
ϕ|Φ|\phi，（φφ：\varphi）
χ|X|\chi
ψ|Ψ|\psi
ω|Ω|\omega

将角度弧度化
```python
from math import radians
c = [30,60,90]
print(map(radians,c))#map将序列对应的函数做映射
#通过list(c)访问map元素
