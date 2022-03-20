### 使用参考
- tensor 相关计算
    - torch.mm(x,y) #乘
    - torch.cat(x,y,0) 按列拼接
    - torch.pow(x,y) 求$x_i^{y_i}$
    - torch.clamp(input,min,max,out==None) 将张量中的元素贾在min～max之间
    - torch.where(condition) 输出满足条件的元素下标(元祖里面n个tensor,n代表各个维度的下标)
- 改变tensor维度顺序 x.permute(2,0,1)
    - torch.unsqueeze(a,dim=-1) dim=1  表示在最后一维插入, 
- 断开拷贝的tensor和原来的tensor之间关系y=torch.transpose(x,0,1).contiguous()
- tensor 填充
    - x.fill_(0)  带_ 表示inplace操作
- tensor 遍历
    - x.gather([[1,3],[2,4]],[0,1]) # 类似numpy中的[:,[1,2,3]]  ，返回[1,4].T
- tensor 排序
    - torch.argsort(score, descending=True) 将tensor根据行排序, 返回下标
- tensor 改变形状
    - x.view() # reshape
    - x.repeat(x,y) 行扩展x倍，列扩展y倍
    - 降低维度 x.squeeze(0)  若第一维度的值为1 则去除第一维度
    - 将Tensor 合成
    ```python
        a=torch.stack((torch.Tensor([1,2,3]),torch.Tensor([4,5,6])))
        tensor([[1., 2., 3.],
                [4., 5., 6.]])
    ```
- tensor 与其他数据类型的转换
    - list 
        - torch.Tensor([])
        - list = tensor.numpy().tolist()
    - numpy 
        - torch.Tensor(n) 或者torch.from_numpy(n) 后者比较快
        - nump = tensor.numpy()
    - list-numpy
        - ndarray = np.array(list)
        - list = ndarray.tolist()
- 从文件中读取tensor 
    - torch.save(x,"x.pt") 使用pickle 将对象（包括模型、张量）序列化
    - x=torch.load("x.pt") 相反
- 保存加载模型(权重，偏置，学习率)
    - net.state_dict() 以键值对输出输出网络中权重名称和值
    - 保存加载参数
        - 保存: 
        ```python
        torch.save(model.state_dict(), PATH)
        ````
        -  加载
        ```python
        model = TheModelClass(*args, **kwargs)
        model.load_state_dict(torch.load(PATH))
        ```
    - 保存加载整个模型
        ```python
        torch.save(model, PATH)
        model = torch.load(PATH)
        ```
- 网络层相关 
    - nn.Sequential() 根据网络层顺序会自动forward
    - 求优化函数：x.relu() 
    - 初始化init ，注意不记录梯度(with torch.no_grad() 或者 .data)
        - `init.constant_(parameter,val=0) `  初始化为常数
        -  normal_(mean, std) 正太分布
        -  tensor.uniform_(-10, 10) 均匀分布
- 工具类
    - 封装数据集生成一个迭代器（这样可以使用torch的多线程，增广等加载功能）torch.utils.data.TensorDataset(train_feature,train_labels)
- 字典相关操作
    - update()新增参数
    - 使用keys()返回所有键值
    - 使用items()返回所有键值对
### 详解
- parameter 是torch的子类，不同的是如果是parameter 类型会被自动加入到网络参数中
    -   'self.weight1 = nn.Parameter(torch.rand(20, 20))' 这个weight_1 会添加到网络参数中
- 一个网络层对象可以作用于网络中的不同层，其中变量是共享的
### 模型构造
- Moudle 都不需要定义forback()
    - 无须定义反向传播，系统根据自动求梯度生成反向传播函数
    - net(x) 调用继承的__call__() 从里面调用forward()
    - 子类，注意可以大层套小层，比如外面无序，里面有序
        - sequential 生成过程多了判断输入网络层是否有序，无则改为有序的. 然后根据顺序 自动完成整个网络的计算
            - 不需要定义forward, 以下都要
            - 可以通过索引访问网络层
        - Moudlelist
            -  与sequential 相比 因为本身是无序的所以不需要相邻层维度匹配
            - 与list相比，它会将参数添加到整个网络中
        - MoudleDict

批量归一化
----
[讲的十分好 值得参考](https://blog.csdn.net/zlb872551601/article/details/103572027)
> 输入数据都会有预处理步骤，目的是将不同尺度的数据标准化，方便比较，便于训练模型
> 同理在经过多层处理后也会形成不同的分布，所以加个批量归一化
> 参考查到的一句话:即从大的方向上看，神经网络则需要在这多个分布中找到平衡点，从小的方向上看，由于每层网络输入数据分布在不断变化，这也会导致每层网络在找平衡点，显然，神经网络就很难收敛了。

- 为何要引入拉伸和偏移:
    - 只对输入进行标准化之后，必然导致上层学习到的数据分布发生变化，以sigmoid激活函数为例，如果BN之后数据经过一个sigmoid激活函数，由于标准化之后的数据在0-1，处于sigmoid函数的非饱和区域，相当于只经过只经过线性变换了，连sigmoid激活非线性变换的作用都没了。这样就破坏了BN之前学到的该有的分布，所以，需要恢复BN数据之前的分布，具体实现中引入了变换重构以及可学习参数γ，β，γ，β变成了该层的学习参数，与之前网络层的参数无关，从而更加有利于优化的过程。（γ，β变成了该层的学习参数，与之前网络层的参数无关，这句话是核心，说明了虽然我们要恢复到前一层学习到的特征分布，但怎么恢复，只与当前该层有关，与之前的各层已都没关系了）
    - 拉伸和偏置导致输出的方差和均值发生改变，但与原来层无关，这两个参数是通过后面层学到的
- 注意几点
    - 在训练数据是为了模型更有效需要使用批量归一化，而在 测试数据时为了得到准确的结果不需要这样（类似于丢弃层）可以使用训练时得到的移动平均值
    - 不适合当minibatch=1的情况
- BN层需要通道数来分配不同的拉伸、偏移参数
损失函数比较
---
- 差函数，误差较小是 导数还是很大，精确度不够
- 方差函数，因为有平方所以，如果存在少量的差值将会对预测结果有很大的影响。但是收敛更快
- smoothl1 结合了以上优点
cnn
-----
- 与全连接不一样，只需要定义输入输出的通道数，不需要定义输入输出的大小
- 矩阵形状
    - 为保持输出和输入的形状不变 应该添加行（列）s-1 s表示核的大小
    - 步幅为1的卷积运算 `Y[i, j] = (X[i: i + h, j: j + w] * K).sum()`
- 卷积层    
    - 深度学习中的卷积操作是互相关操作
    - 输出形状 $((n_h - k_h + 2 \times p_h )/s + 1 )$
        - h、w为高宽
        - n 表示输入大小
        - k 表示卷积核大小
        - s 表示步幅大小
        - p 表示填充大小
    -  1乘1的卷积核：降低模型复杂度(把通道数改为1维)，并且不改变位置信息（后面可以继续卷）
        - 全连接包括所有特征信息 可用于分类
- 池化层：卷积操作后边缘可能出现偏移

ResNet
---
> 随着网络的加深往往可以得到不同维度的特征，但是也会导致信息丢失（DPI）, 而残差网络输出加上输入，保证输出含有更多信息

- 下采样：当输入输出尺寸和维度不一致时需要下采样和升维
- 残差网络的两种基本结构
    -  basic block, 参数不是很多的时候，如resnet18, resnet34
        - 第一层为$3 \times 3 \times 64 \times 64$卷积
        - 第二层为$3 \times 3 \times 64 \times 64$卷积
    - 瓶颈结构 bottlenet(中间大两头小）(减少参数)
        - 第一层为$1 \times 1 \times 64 \times 256$ 卷积，压缩通道数
        - 第二层为$3 \times 3 \times 64 \times 64$ 卷积，提取图片特征
        - 第三层为$1 \times 1 \times 256 \times 64$ 卷积,扩张通道数
            - 残差边判断决定使用 哪个基本快
- 基本块：
    - identity block 残差边无操作
        - 无法改变输出维度
    - conv block 残差边有卷积和标准化
        - 可以改变输出特征的维度
图像增广
---
- 将数据类型转换为Tensor格式 torchvision.transforms.ToTensor()
- 上下翻转不改变识别类别
- 左右翻转：torchvision.transforms.RandomHorizontalFlip(img)
- 图片裁剪：torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
    - 裁剪面积为原面积的0.1~1
    - 高宽比随机取0.5~2
    - 然后高和宽缩放到200 像素
- 改变颜色:torchvision.transforms.ColorJitter(brightness=0.5)
    - brightness 亮度变化为原来的1-0.5～1+0.5之间
    - hue=0.5 色调
    - contrast=0.5 对比度
    - saturation 饱和度
- 叠加以上操作 'augs = torchvision.transforms.Compose([ torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])' 

神经网络的理解
----
- 多加一层 得到的解空间是原来的子集
- 卷积得到边界信息，那么边界信息在卷积得到一个文印，文印在卷积得到一个图案
- 标准化后利用relu函数才能发挥他的功能 否则如果全部大于0则和普通线性函数没区别
- 标准化后如果利用simoid函数激活 就会导致分布落在0-1之间 失去了学到的非线性分布
问题 
----
- 恒等映射为什么新模型可能得出更优的解来拟合训练数据集
- 残差网络和稠密连接网络的优点，为什么要这么设计
- 提高通道数有什么用
- 锚框 高宽比为什么加更号

训练锚框步骤
---
1. 生成n+m-1个框,锚框的宽和高将分别为$ws\sqrt{r}$和$hs/\sqrt{r}$ (也就是说当r为本身的高宽比时，框形成的面积是原来的s倍)
$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$
    - w(宽)
    - h（高)
    - s 与原图的比例
    - r（高宽比,意义是在原图的高宽比基础上伸缩多少)
    - 步骤
        1. 相对中心像素点的左下角、右上角，x，y的距离
        2. 生成像素点
        3. 将距离和像素点相加得到深度为像素点，行为框，列为坐标的张量
2. 求锚框和真实框交并比
    1. 求每个锚框和真实框的交集的左下脚，右上角（左下角求最大右上角求最小），注意上个步骤最后生成的是多个框，而真实框只有一个，所以会用到广播功能
    2. 将求出来的右上角减左下角，得到其宽、高。注意如果小于0则令其等于0（如果两个没有相交就会出现这种情况); 此时求出来的是行为多少框、列为宽，高。
    3. 将宽高相乘，得到行为框数，列为交集(一列)
    4. 求每个框的面积
    5. Iou=交集/(真实框+锚框-交集)
3. 训练锚框: 注意应为存在相对大小等分布问题，所以需要将偏移量转换才能衡量
    1. 在2步骤得到的框中，选取每个真实框对应Iou最大的锚框，并去除相应的行列(如果没被赋予真实框的锚框对应的真实框Iou达到一定值也会给他分配）,分配的结果以矩阵的形式保存(锚框数，1)
    2. 根据上步得到的对应关系,与真实框的相对中心和偏移量(所以需要将得到的左下角，右上角形式变为以中心点高宽衡量的方式) 计算方法如下:
        $$
        \left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
        \frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
        \frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
        \frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),
        $$
    3. 重复输入不同的真实框，改变锚框的中心和偏移量 得到三个变量
        - 每次训练的锚框位置和偏移量
        - 为了过滤背景的mask
        - 类别
4. 非极大值抑制
5. 

faster-rcnn
---
> 正负样本用来分类预测，正样本用来回归预测
- 区域提议网络(rpn)的生成
    1.  将输入图像进行backbone（resnet50,vgg16) 提取特征，生成特征图(feature map) shape(38,38,1024)
        - 在整个操作中conv 不改变大小,只改变通道数，pool层将大小减半
    2.  对特征图进行一次3乘3卷积(特征整合,t=1,p=1 不改变形状)，然后  
        - 进行一个18通道的$1\times 1$卷积生成(38,38,18)预测每个点是否是否包含物体,9个先验框($9\times 2$)
        - 进行一个36通道的$1\times 1$ 卷积生成(38，38，36)预测每个坐标的偏移、伸缩情况,9个预测框($9\times 4$)
            - $1\times 1$相当于全连接层进行分类
    3. 根据预测结果变换先验框，然后经过一下筛选,得到挑选后的建议框
        1. 概率大于p 的前k个目标
        2. 调整坐标，防止越界
        3. 删除过小的框
        4. nms
        5. 将剩下的框与真实框求交并比，挑选出每个真实框对呀Iou最大的锚框
            - Iou小于设定的最小值 分配-1
            - Iou大与最小值 分配-2
    4. 通过3的预测结果与真实框配对求损失
        - P,G 表示特征图，P_x 表示特征图上的锚框，G表示真实框，为了拟合真实宽，定义,P经过平移，伸缩后得到G 
    $$
    t_x = (G_x - P_x)/P_w\\
    t_y = (G_y - P_y)/P_h\\
    t_w = \log\frac{G_w}{P_w}\\
    t_h = \log\frac{G_h}{P_h}\\
    $$
        - 定义目标函数为$d(P)=w \Phi(P)$，因为假如锚框和真实框Iou比较大，可以通过线性拟合而来（误差较小） 即1乘1卷积, 所以损失函数为
        $$
        Loss = \sum\limits_{i}^{N}(t-w\Phi(P))^2
        $$

