# 题目：PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

## Tag

【点云直接操作】【3D目标分类】【3D部件分割】【3D场景分割】【3D目标检测】

## 贡献

- 直接对点云网络进行操作来提取特征。
- 
- 



# 1. 相关背景

**点云特征**

- **无序性**
- **稀疏性**
- **信息量有限**

> 以往学者用深度学习方法在处理点云时，往往将其转换为特定视角下的深度图像或者体素（Voxel）等更为规整的格式以便于定义权重共享的卷积操作等。

**欧式空间点云特征**

- **无序性：**点云的输入顺序不应影响结果；    => 构建对称函数
- **点之间的交互：**每个点不是独立的，而是与其周围的一些点共同蕴含了一些信息，因而模型应当能够抓住局部的结构和局部之间的交互；
- **转换不变性：**点云整体的旋转和平移不应该影响它的分类或者分割。


## 1.1 作者想法

- 
- 
- 

# 2. 论文特点

![image-20200903084855990](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012083259.png)

## 2.1 网络

​		**分类网络**对于输入的点云进行输入变换（input transform）和特征变换（feature transform），随后通过最大池化将特征整合在一起。**分割网络**是分类网络的延伸，其将整体和局部特征连接在一起得出每个点的分数。

​		其中，mlp是通过共享权重的卷积实现的，第一层卷积核大小是1x3（因为每个点的维度是xyz），之后的每一层卷积核大小都是1x1。即特征提取层只是把每个点连接起来而已。经过两个空间变换网络和两个mlp之后，对每一个点提取1024维特征，经过maxpool变成1x1024的全局特征。再经过一个mlp（代码中运用全连接）得到k个score。分类网络最后接的loss是softmax。



## 2.2 特点

### 2.2.1 点云数据无序性解决

- **构建对称函数解决**

**对称函数：**

- +与x是能处理两个输入的对称函数
- 点云排序是一个可能的对称函数

构建对称函数g：深度学习实际上是对复杂函数的拟合，g是由单变量函数与最大池化实现的。



### 2.2.2 点云数据转换不变性

- **T-Net**

​		作者在这里构建了T-Net网络，学习一个获得3x3变换矩阵的函数，并对初始点云应用这个变换矩阵，这一部分被称为**输入变换（input transform）**。随后通过一个mlp多层感知机后，再应用一次变换矩阵（**特征变换 feature transform**）和多层感知机，最后进行一次最大池化。	

以上阶段学习到的变换函数是如下图所表示的函数g和h，来保证了模型对特定空间转换的不变性。

理解：深度学习实际上是对复杂函数的拟合，g作为一个对称函数，是由单变量函数与最大池化实现的；h是mlp结构，代表了一个复杂函数（在图中是将一个3维向量映射成1024维向量的函数）。

![image-20200903213506768](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012085426.png)



### 2.2.3 点与点之间的交互性

- **整合局部和全局信息**

	​		对于点云分割任务，我们需要将局部跟全局信息结合起来。

	​        经过特征变换后的信息称作局部信息，它们是与每一个点紧密相关的；我们将局部信息和全局信息简单地连接起来，就得到用于分割的全部信息。

## 2.3 详细信息



## 2.4 理论分析

除了模型的介绍，作者还引入了两个相关的定理：

![](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012093804.png)

定理1证明了PointNet的网络结构能够拟合任意的连续集合函数。

![](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012093902.png)

定理2(a)说明对于任何输入数据集，都存在一个**关键集**和一个**最大集**，使得对任何集合，其网络输出都一样。这也就是说，模型对输入数据在有噪声和有数据损坏的情况都是鲁棒的。定理2(b)说明了关键集的数据多少由maxpooling操作输出数据的维度K给出上界（PointNet中为1024）。换个角度来讲，PointNet能够总结出表示某类物体形状的关键点，基于这些关键点PointNet能够判别物体的类别。这样的能力决定了PointNet对噪声和数据缺失的鲁棒性。

下图给出了一些关键集和最大集的样例：

![](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012093959.webp)


# 3. 实验

## 3.1 3D recognition tasks

### 3.1.1 3D目标分类

**benchmarks：** ModelNet40   shape classifification benchmark.

 有12,311个CAD模型来自40个人造物体类别，9843训练2468测试。

我们根据面面积对网格面上的1024个点进行了均匀采样，并将它们归一化为一个单位球体。 在训练中，我们通过随机旋转物体来增强点云  上轴和抖动每个点的位置由高斯噪声与零均值和0.02标准差。

![image-20201012103523057](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012103747.png)

### 3.1.2 3D零件分割

**benchmarks：**ShapeNet part data  set from [29], 其中包含16个类别的16,881个形状，总共标注了50个部分。 大多数对象类别被标记为两到五个部分。

我们将零件分割描述为一个每点分类问题。 评价度量是点上的mIoU。 对于C类的每个形状S，计算形状的mIoU：对于类别中的每个部件类型 奥利C，和预测之间计算IoU。 如果地面真相和预测点的结合是空的，那么将IoU部分计数为1。 然后，我们对C类到通用电气的所有部件类型进行平均IoUs 为了那个形状。 为了计算类别的mIoU，我们取该类别中所有形状的**mIoUs的平均值**。

![image-20201012105616947](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012105617.png)

![image-20201012110202707](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012110202.png)

### 3.1.3 3D场景分割

**benchmarks：**Stanford 3D semantic parsing data set

数据集包含来自Matterport扫描仪的3D扫描，在6个区域，包括271个房间。 扫描中的每个点都用13个类别(椅子、桌子、地板、墙)中的一个语义标签进行注释 其他）。每个点使用9维的向量表示。XYZ, RGB and 房间的归一化位置从0到1。 

为了准备训练数据，我们首先按房间分割点，然后将房间分成1m×1m的块。使用pointNet预测每一个块的每个点的类别，

 

![image-20201012111248726](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012111248.png)

我们将我们的方法与使用手工点特征的基线进行比较。 基线提取相同的9点局部特征和三个附加特征：局部点密度、局部曲率和法线 。 我们使用标准MLP作为分类器。

![image-20201012124611725](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012124611.png)

### 3.1.4 3D目标检测

在3D语义分割的基础上，进行3D目标检测。

![image-20201012111512174](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012125120.png)

## 3.2 validate our network design

基于3D目标分类网络来验证。

- **使用对称函数解决无序性的验证：**对称函数的作用

![image-20201012131959638](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012131959.png)

- T-Net 进行Input and Feature Transformations 的有效性

![image-20201012142351826](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012142351.png)

- 健壮性测试

![image-20201012142757576](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012142757.png)

## 3.3 visualize what the network learns 

![image-20201012145624296](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012145624.png)

## 3.4  analyze time and space complexity

![image-20201012145938289](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201012145938.png)

# 4. 代码分析(pytorch版)
> 代码来源： []()

## 4.1 model



## 4.2 预处理



## 4.3 训练





## 4.4 可视化





## 4.5 tips



# 5. Q&A

1. “mlp是通过共享权重的卷积实现”？ 如何进行共享？
2. 单变量函数是什么？
3. 

---


# 参考

1. [PointNet：基于深度学习的3D点云分类和分割模型 详解](https://www.jianshu.com/p/6a0fc51187c1)
2. [Momenta高级研究员陈亮论文解读](https://links.jianshu.com/go?to=https%3A%2F%2Fwww.leiphone.com%2Fnews%2F201708%2FehaRP2W7JpF1jG0P.html)
3. [美团无人配送的知乎专栏：PointNet系列论文解读](https://links.jianshu.com/go?to=https%3A%2F%2Fzhuanlan.zhihu.com%2Fp%2F44809266)
4. [hitrjj的CSDN博客:三维点云网络——PointNet论文解读](https://links.jianshu.com/go?to=https%3A%2F%2Fblog.csdn.net%2Fu014636245%2Farticle%2Fdetails%2F82755966)
5. [github源码](https://links.jianshu.com/go?to=https%3A%2F%2Fgithub.com%2Fcharlesq34%2Fpointnet%2Fblob%2Fmaster%2F)
6. [痛并快乐着呦西的CSDN博客：三维深度学习之pointnet系列详解](https://links.jianshu.com/go?to=https%3A%2F%2Fblog.csdn.net%2Fqq_15332903%2Farticle%2Fdetails%2F80224387)
7. []()
8. []()
9. []()
10. []()