# 题目：KPConv: Flexible and Deformable Convolution for Point Clouds-ICCV2019

## Tag

【点卷积】【可变形卷积】【点云分类】【点云分割】【核点】

## 摘要

- 直接对点云进行卷及操作而不需要转换成中间表示(如晶格表示)。
- ~~提出了核点的概念？~~
- 提出rigid KPconv解决简单任务更适合：分类或者小的分割数据集
- 提出deformable KPConv=>对输入点进行局部的转换从而更好的适应点云的几何特征=>解决复杂任务更适合：大场景多样的场景分割、实例分割
- 分类和分割上达到了最优

## 贡献

- 提出了KPConv核点卷积，比固定网格卷积具有更大的灵活性。

## 思想

- 直接对点进行卷积提取特征，但在提取特征过程中要解决如何构建卷积的问题，

-  KPConv使用半径邻域内的点作为输入，并使用核点的空间权重对输入点进行处理。

  

# 1. 背景

​		通常的点云卷积操作需要将点云转换成晶格表示形式，晶格表示是规则的，所以可以容易进行3D-CNN操作，也能很好的保留局部邻域特征，但是需要转换成这种晶格表示这一中间形式，效率会大大折扣。

​		而直接处理点云，不需要进行转换成中间表示，效率会提高，但是缺少了局部邻域特征，导致使得聚合局部邻域特征成为一个难点。

​		本文使用核点，构建起整个空间结构，在rigid KPconv中固定核点，而在deformable KPConv对核点周围的输入点进行微调，使其更适应点云的几何结构。如此一来便可以再次使用3D-CNN操作，来聚合邻域特征而不需转换成中间表示形式，提升了效率的同时，增强了空间特征聚合能力。



# 2. 特点

- 3D filter
- 我们使用一组核点来定义区域，每个核点上有相应的weight值。
- 核点的数量不固定。
- 需要正则化来使可变形核适应点云结构并且避免空的空间。
- 使用半径邻域，而不是使用KNN邻域。
- KNN在非均匀采样中不具有鲁棒性，对密度不敏感，此处采用半径邻域+均匀采样。
- 点卷积和图卷积区别：图卷积聚合了局部表面块的特征在欧式空间中，无法捕捉住欧式空间中的表面块的变形；点卷积聚合了3D几何结构特征，能够抓住表面的变形。
- KPConv的下采样策略，正则化
- 使用线性相关性而不是高斯相关性，更有利于反向传播

## 2.1 核函数的定义

- 核函数g（关键所在）![image-20200925141446558](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200925141455.png)
- h是两者的相关性函数
- ![image-20200925141523768](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200925141523.png)

## 2.2 Rigid or Deformable Kernel

- 放置K个核点，核点的位置需要学习
- 每个点之间具有排斥力
- 点由于吸引了被限制在球体中，有一个点被限制在球心。
-  周围点被重新标度为15σ的平均半径，确保了每个核点影响区域之间的小重叠和良好的空间覆盖。
- ![image-20200925142425128](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200925142425.png)
- 

## 2.3




# 3. 实验

## 3.1 ModelNet40  for classification

## 3.2 ShapenetPart for part segmentation

## 3.3 3D Scene Segmentation

### 3.3.1 数据集：Scannet、S3DIS、Semantic3D、Paris-Lille-3D




### 3.3.2 结果对比(mIOU)

![image-20200915095213606](img/image-20200915095213606.png)

rigid KPConv在Scannet与Semantic3D中表现比deformable KPConv好，分析原因是因为Semantic3D缺乏多样性，类别较少。即表明deformable KPConv在多样性上作用更明显。

![image-20200915105216227](img/image-20200915105216227.png)

![image-20200915105235597](img/image-20200915105235597.png)

![image-20200915105256259](img/image-20200915105256259.png)




## 3.4 消融实验

- rigid KPConv
- deformable KPConv



# 4. 代码分析(pytorch版)
> 代码来源： []()

## 4.1 model



## 4.2 预处理



## 4.3 训练





## 4.4 可视化





## 4.5 tips



# 5. 我的想法

- 是否可以不使用核点，直接对input点做变形，来聚合特征。
- 
- 
- 
- 

---


# 参考

1. [Project page]( https:// github.com/ HuguesTHOMAS/ KPConv) from github.
2. [KPConv：点云核心点卷积 (ICCV 2019)](https://zhuanlan.zhihu.com/p/92244933) from 知乎.
3. []()
4. []()
5. []()

