<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](#randla-net-efficient-semantic-segmentation-of-large-scale-point-clouds)
  - [[TOC]](#toc)
- [2. 相关工作](#2-%E7%9B%B8%E5%85%B3%E5%B7%A5%E4%BD%9C)
- [3. RandLA-Net](#3-randla-net)
  - [3.1 随机采样的有效性](#31-%E9%9A%8F%E6%9C%BA%E9%87%87%E6%A0%B7%E7%9A%84%E6%9C%89%E6%95%88%E6%80%A7)
  - [3.2 局部特征聚合模块](#32-%E5%B1%80%E9%83%A8%E7%89%B9%E5%BE%81%E8%81%9A%E5%90%88%E6%A8%A1%E5%9D%97)
- [4. Experiments](#4-experiments)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

CVPR2020   牛津大学、中山大学、国防科技大学

---
[TOC]
---

问题：

- 采用复杂的点采样，计算效率低。
- 繁重的预处理和后处理。

作用：可以进行高效轻量的可以直接进行大规模点云的逐点语义分割。

**特点：** 

- 采用**随机点采样**，而不是复杂的点采样方法。计算和内存效率非常高，但随机抽样可能会偶然丢弃关键特征。
- 引入了一种新的局部特征聚合模块来逐步增加每个三维点的感受野，从而有效地保留了几何细节。
- 能够部署到实时的应用中。
- （基于简单的随机采样和有效的局部特征聚合器的原则）
- 可进行端到端训练，不需要额外的预处理和后处理步骤。

效果：

- RandLA-Net一次可以处理100万个点，比现有的方法快200倍
- RandLA-Net在两个大规模基准数据集Semantic3D 和 SemanticKITTI上state-of-the-art.
- ![image-20200702214631280](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200703000723.png)
- 图1.PointNet++[44]，**SPG**[26]的语义切分结果和我们在SemancKITTI[3]上的方法。我们的RandLA-Net只需要0.04s就可以在三维空间中直接处理150×130×10米上$10^5$个点的大型点云，速度比**SPG**快200倍。红色圆圈突出了我们方法的卓越分割精度。

贡献：

- 对现有的采样方法进行了分析和比较，确定随机采样是对大规模点云进行有效学习的最合适的组成部分。
- 提出了一个有效的局部特征聚合模块，通过逐步增加每个点的感受野来保持复杂的局部结构。
- 在多个大规模基准测试中，我们表现出显著的内存和计算增益，并且超过了最先进的语义分割方法。



**RandLA-Net 特点：**  **<font color='red'>如何实现？</font>**

- 该网络对点云处理时仅依赖于网络内的随机采样，故所需内存和计算消耗较少。
- 提出一个局部特征聚合模块，通过考虑局部空间关系和点的特征，来获得较大的感受野。    
- 参照point net及point net++的共享MLP，不使用图构造和核化。   效率高。    **<font color='red'>共享MLP为什么效率高？</font>** 

---

**<font color='red'>补充：</font>**

[点云采样的方法]: https://yongqi.blog.csdn.net/article/details/105608948

- 最远点采样、反密度采样 常用作小尺度点云的方法。

2. 实时处理怎么能够实现？

3. 局部空间编码（LocSE) ?

4. attentive pooling作用及用法？

5. pointnet中的共享MLP

6. SPG

7. 点与点特征有什么区别？在locSE中

8. 点的感受野，如何增加感受野？

   

PointNet++比PointNet的优势在于，前者可以更好的捕捉点云的局部结构



https://cloud.tencent.com/developer/article/1594448   完整的中文翻译

https://www.sohu.com/a/398379788_715754   直击重点的讲解

https://blog.csdn.net/qq_43058685/article/details/105089579   代码阅读

https://github.com/QingyongHu/RandLA-Net   官方代码





---

3D点云语义分割挑战：

- 深度传感器获取的原始点云通常是不规则采样的、无结构的和无序的。是非结构化数据。

**以往论文做法：**

- PointNet：使用共享的MLP(多重感知机)直接处理3D点云。计算上有效，但无法捕获每个点的更广泛的上下文信息。<font color='red'>问题1：如何学习更丰富的局部结构？</font>

<font color='red'>问题1：如何学习更丰富的局部结构？</font>

- neighbouring feature pooling：相邻特征池化    [44, 32, 21, 70, 69]
- graph message passing：图形消息传递   [57, 48, 55, 56, 5, 22, 34]
- kernel-based convolution ：核卷积   [49, 20, 60, 29, 23, 24, 54, 38]
- attention-based aggregation：基于注意力的聚合   [61, 68, 66, 42]

<font color='red'>以上未考虑大规模点云，全是4k点或1×1米的块，如果没有块划分等预处理步骤，就不能直接扩展到更大的点云(例如，数百万个点，最大可达200×200米)</font>

**分析小规模点云分割为什么无法进行大规模点云分割：**

1.  这些网络常用的**点采样方法**要么计算量大，要么内存效率低；例如，广泛使用的**最远点抽样[44]**需要200秒以上才能抽样100万个点中的10%。
2.  大多数现有的**局部特征学习模块**通常依赖于计算代价高昂的核化或图构造，因此无法处理大量的点。 
3.  对于通常由数百个对象组成的大规模点云，现有的局部特征学习器由于其接受场的大小有限，要么无法捕获复杂的结构，要么效率低下。

大规模点云与小规模点云区别：

​	

| 比较方面 | 大规模点云 |  小规模点云  |
| :------: | :--------: | :----------: |
| 对象组成 | 数百个对象 | 对象数量较少 |
|          |            |              |
|          |            |              |

**大规模点云语义分割的一些处理方法：**

- SPG[26]在应用神经网络学习每个超点语义之前，将大的点云作为超图进行预处理。
- FCPN[45]和PCT[7]都结合了体素化和点级网络来处理海量的点云。<font color="red">虽然它们获得了不错的分割精度，但预处理和体素化步骤的计算量太大，无法部署到实时应用中。</font>

**目标：**

​	设计一种存储和计算高效的神经结构，它可以在一次遍历中直接处理大规模的三维点云，而不需要体素化、块划分或图形构造等任何前后处理步骤。

**面对的挑战：**

- 1)内存和计算效率高的采样方法，以逐步对大规模点云进行下采样，以适应当前GPU的限制； **=> 随机点采样**
- 2)有效的局部特征学习器，以逐渐增加感受野大小，以保持复杂的几何结构。大多数现有的**局部特征学习模块**通常依赖于计算代价高昂的核化或图构造，因此无法处理大量的点。  **随机采样造成丢失关键点信息 => 构建了一个局部特征聚合模型来捕获逐渐变小的点击上的复杂的局部结构(通过逐渐增加每个神经层的接收场大小来有效地学习复杂的局部结构。)**     在这块理解： 只是保证一定的局部结构，而不会是向小规模点云分割中一样，保证更微小的局部结构。



**3.2节中几种采样策略：**

- **最远点采样**和**反密度采样**是最常用于小比例尺点云的方法[44，60，33，70，15]。 
- **常用的采样方法限制了对大型点云的缩放，并成为实时处理的重要瓶颈。**  理解： 因为关注小型点云的更精细的局部结构特征，所以不必进行太多的缩放。那处理大型点云时，就必须进行缩放，来保证其感受野。随机抽样的优势：

**随机抽样的优势：**

- 速度快，缩放效率高。
- 代价：一些关键的特征可能会意外的被drop掉，在现有的网络中使用性能会损失。 => 新的局部特征聚合模块，通过逐渐增加每个神经层的接收场大小来有效地学习复杂的局部结构。

**对于每个3D点**：

1. **引入局部空间编码（LocSE）单元以显式保留局部几何结构。**

2. 利用 **attentive pooling** 来自动保留有用的局部特征。

3. 将多个LocSE单元和注意池堆叠成一个扩张的残差块，大大增加了每个点的有效感受野。

这些神经组件都以**共享MLP**的形式实现，因此具有显著的内存和计算效率。



# 2. 相关工作

要从三维点云中提取特征，传统方法通常依赖于手工创建的特征[11，47，25，18]。最近的基于学习的方法[16，43，37]主要包括基于投影的、基于体素的和基于点的方案

**如何从三维点云中提取特征：**

1. 基于投影的网络

   将三维点云投影/展平到二维图像上，在投影过程中可能会丢失几何细节。

2. 基于体素的网络

   将点云体素化为3D网格，应用3D CNN，造成很大的计算量，尤其在处理大规模点云时。

3. 基于点的网络

   引入复杂的神经模块学习逐点的局部特征。

   - neighbouring feature pooling：相邻特征池化    [44, 32, 21, 70, 69]
   - graph message passing：图形消息传递   [57, 48, 55, 56, 5, 22, 34]
   - kernel-based convolution ：核卷积   [49, 20, 60, 29, 23, 24, 54, 38]
   - attention-based aggregation：基于注意力的聚合   [61, 68, 66, 42]

   计算、存储成本大，无法直接扩展到大规模点云。

---

**最近的大规模点云语义分割：**

- SPG [26]将大点云作为超点图进行预处理，以学习每个超点语义；     <font color='red'>图划分计算上消耗大。</font>

- FCPN[45]和PCT[7]都应用了基于体素和基于点的网络来处理海量的点云；   <font color='red'>体素化消耗大。</font>

---



# 3. RandLA-Net

## 3.1 随机采样的有效性

<img src="https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200704095501.png" alt="image-20200704095456711" style="zoom:80%;" />

- 在RandLA-Net的每一层中，大规模的点云被显著地下采样，但仍能够保留精确分割所需的特征。
- 给定一个具有数百万个点、跨越数百米的大型点云，要使用深度神经网络对其进行处理，必然需要在每个神经层对这些点进行渐进式高效的下采样，而不会丢失有用的点特征。
- 使用简单快速的随机抽样方法来大大降低点密度，同时使用精心设计的局部特征聚合器来保留显著的特征。这使得整个网络能够在效率和有效性之间实现极好的权衡。

**高效的采样：** [44、33、15、12、1、60]可以分为启发式方法和基于学习的方法，没有适用于大规模点云的标准采样策略。

**为了从具有N个点的大规模点云P中采样K个点，**

1. **启发式采样：**

- FPS(Farthest Point Sampling)最远点采样：

  ​       返回一个重新排序的度量空间$\left\{p_{1} \cdots p_{k} \cdots p_{K}\right\}$，使得每个$p_{k}$是离前k-1个点最远的点。常用于小规模点云，计算复杂度为$\mathcal{O}\left(N^{2}\right)$，对于大型点云($N-10^6$)，在单个gpu上处理最多需要200秒。

- IDIS(Inverse Density Importance Sampling)逆密度重要性采样：

  ​		IDIS根据每个点的密度对所有N个点重新排序，然后选择密度高的前K个点[15]。计算复杂度为$\mathcal{O}\left(N\right)$，处理$10^6$点需要10秒，

与FPS算法相比，IDIS算法效率更高，但对异常值也更敏感。在实时系统中这种效果仍旧太慢。

- RS(Random Sampling)随机采样：

  ​		从原始的N个点中均匀的读取K个点，计算复杂度为$\mathcal{O}\left(1\right)$，与输入点的总数无关，即它的时间恒定。与FPS和IDIS相比，无论输入点云的规模大小，随机采样的计算效率都是最高的。处理个$10^6$点只需要0.004秒。满足实时性的需求。

2. **基于学习的采样：**

- (GS)Generator-based Sampling 基于生成器的采样：

  ​		学习生成一个小的点集来近似表示原始的大点集。在推理阶段使用FPS将生成的子集与原始集进行匹配，这会带来额外的计算。在我们的实验中，对$10^6$ 个点中的10%进行采样需要长达1200秒的时间。

- (CRS)Continuous Relaxation based Sampling 基于连续松弛的采样：   对空间消耗大，对时间消耗小。

  ​		CRS方法[1，66]使用重新参数化技巧将采样操作放宽到一个连续的域以进行**端到端训练**。具体地说，每个采样点都是基于整个点云的加权总和来学习的。当使用一次通过矩阵乘法同时对所有新点进行采样时，它会导致较大的权重矩阵，从而导致无法承受的存储成本。例如，估计对$10^6$个点进行10%的采样需要300  GB以上的内存。

- (PGS)Policy Gradient based Sampling 基于策略梯度的采样：

  ​		PGS将采样操作公式化为马尔可夫决策过程[62]。它顺序地学习概率分布来采样。当点云较大时，由于探测空间非常大，学习概率具有较大的方差。对$10^6$点进行10%的抽样，探索空间为$\mathrm{C}_{10^{6}}^{10^{5}}$，不太可能学习到有效的采样策略。实验发现，如果将PGS用于大型点云，则网络很难收敛。

**高效采样的总结：**

​		总体而言，FPS，IDIS和GS在计算上过于昂贵，无法应用于大型点云。  CRS方法的内存占用过多，PGS很难学习。相比之下，**随机采样具有以下两个优点：1）因为与输入点总数无关，，因此其计算效率非常高；  2）不需要额外的存储空间即可进行计算。**因此，我们可以得出结论：与所有现有替代方法相比，**随机采样是迄今为止最适合处理大规模点云的方法（就实时性而言)。**但是，随机采样可能会导致许多有用的点特征丢失。为了克服这个问题，我们提出了一个功能强大的局部特征聚合模块，如下一节所述。



## 3.2 局部特征聚合模块

![image-20200704144118285](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200704144118.png)



局部特征聚合模块由三个神经单元组成：

1) 局部空间编码单元(LocSE)

2) attentive pooling

3) 扩张残差块

1. **局部空间编码模块(LocSE)**     对接收到的点云特征进行编码，从而使得更好的观察局部几何图案。

   ​		接收原始的点云或者中间学习到的点云特征，该局部空间编码单元显示地嵌入所有相邻点的x-y-z坐标，使得对应的点特征总是知道它们的相对空间位置。这使得LocSE单元能够显式地观察局部几何图案，从而最终有利于整个网络有效地学习复杂的局部结构。

![image-20200704150909117](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200704150909.png)

**(1) 局部空间编码(Local Spatial Encoding)**

​		此模块用于显式地对输入的点云的三维坐标信息进行编码。不同于直接将各个点的三维坐标作为一个普通的通道特征输入到网络中，LocSE模块旨在显式地去编码三维点云的空间几何形状信息，从而使得网络能够从各个点的相对位置以及距离信息中更好地学习到空间的几何结构。具体来说分为以下步骤：

- 首先，我们用 ![[公式]](https://www.zhihu.com/equation?tex=K) 最近邻搜索算法为每一个点 ![[公式]](https://www.zhihu.com/equation?tex=p_i) 找到欧氏空间中最近的![[公式]](https://www.zhihu.com/equation?tex=K)个邻域点
- 对于 ![[公式]](https://www.zhihu.com/equation?tex=p_i) 的![[公式]](https://www.zhihu.com/equation?tex=K)个最近邻点 ![[公式]](https://www.zhihu.com/equation?tex=\{p_i^1+\cdots+p_i^k+\cdots+p_i^K\}) , 我们显式地对点的相对位置进行编码，将中心点的三维坐标 ![[公式]](https://www.zhihu.com/equation?tex=p_i) , 邻域点的三维坐标 ![[公式]](https://www.zhihu.com/equation?tex=p_i^k) , 相对坐标 ![[公式]](https://www.zhihu.com/equation?tex=+(p_i-p_i^k)) 以及欧式距离 ![[公式]](https://www.zhihu.com/equation?tex=||p_i-p_i^k||) 连接(concatenation)到一起。如下所示：$\mathbf{r}_{i}^{k}=M L P\left(p_{i} \oplus p_{i}^{k} \oplus\left(p_{i}-p_{i}^{k}\right) \oplus\left\|p_{i}-p_{i}^{k}\right\|\right)$
- 最后我们将邻域点![[公式]](https://www.zhihu.com/equation?tex=p_i^k) 对应的点特征 ![[公式]](https://www.zhihu.com/equation?tex=\mathbf{f}_i^k) 与编码后的相对点位置 ![[公式]](https://www.zhihu.com/equation?tex=\mathbf{r}_{i}^{k}) 连接到一起，得到新的点特征 ![[公式]](https://www.zhihu.com/equation?tex=\mathbf{\hat{f}}_i^k)。

**(2) Attentive pooling**

![image-20200705115843299](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200705122318.png)

​		此模块用于将上述单元输出的邻域点特征集聚合到一起。现有的大多数算法通常采用启发式的max/mean/sum pooling来hard integrate邻域点特征集，这样做有可能导致许多有用的信息被丢失。不同于此，我们希望通过attention mechanism来自动学习和聚合邻域点特征集中有用的信息。具体来说，对于一个邻域特征点集合 ![[公式]](https://www.zhihu.com/equation?tex=\mathbf{\hat{F}}_i+%3D+\{\mathbf{\hat{f}}_i^1+\cdots+\mathbf{\hat{f}}_i^k+\cdots+\mathbf{\hat{f}}_i^K+\}) ，我们首先设计一个共享函数 ![[公式]](https://www.zhihu.com/equation?tex=g(\cdot)) 来为每一个点学习一个单独的attention score，其中:$\mathbf{s}_{i}^{k}=g\left(\hat{\mathbf{f}}_{i}^{k}, \boldsymbol{W}\right)$，![[公式]](https://www.zhihu.com/equation?tex=\boldsymbol{W}) 是共享MLP的可学习参数。然后，我们将学习到的attention score视作一个能够自动选择重要特征的soft mask，最终得到的特征是这些邻域特征点集的加权求和，如下所示:

$\tilde{\mathbf{f}}_{i}=\sum_{k=1}^{K}\left(\hat{\mathbf{f}}_{i}^{k} \cdot \mathbf{s}_{i}^{k}\right)$

**(3)扩张残差块(Dilated Residual Block)**：增大感受野    **这部分不详细**

![image-20200705122309379](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200705122309.png)

​		考虑到输入的点云会被持续大幅度的降采样，因此显著地增加每个点的感受野是非常有必要的。换句话来说也就是，我们希望即便RandLA-Net随机地丢弃某些点的特征，输入点云的整体的几何细节也能够被保留下来。基于这样一个想法，我们**将多个LocSE，Attentive Pooling以及skip connection连接在一起组成扩张残差块(Dilated Residual Block)**。下图进一步说明了扩展残差块的作用，可以看到: 红色的点在第一次LocSE/Attentive Pooling操作后的有效感受野是与之相邻的 $K$个相邻点，然后在第二次聚合以后最多能够将感受野扩展到个$K^{2}$邻域点。相比于直接增大K最近搜索中的K值而言，这是一种更加廉价高效的方式来增大每个点的感受野以及促进邻域点之间的feature propogation。通过后面的ablation实验，我们的扩张残差块最终使用两组LocSE和attentive pooling单元，以平衡最终的分割性能以及计算效率。

<font color='red'>如何增加感受野</font>

![image-20200705123928590](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200706005009.png)



**整体结构：**

![image-20200705164329912](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200706005010.png)

RandLA-Net的网络结构. (N, D)分别表示点的个数和特征维数。FC:全连通层，LFA:局部特征聚合，RS:随机采样，MLP:共享多层感知器，US:上采样，DP: Dropout		

​		最后，我们将随机采样以及局部特征聚合模块组合到一起，基于标准的encoder-decoder结构组建了RandLA-Net。网络的详细结构如下图所示，可以看到，输入的点云在RandLA-Net中持续地进行降采样以节约计算资源及内存开销。此外，RandLA-Net中的所有模块都由简单高效的feed-forward MLP组成，因此具有非常高的计算效率。最后，在解码器中的上采样阶段，不同于广泛采用的三线性插值(trilinear interpolation)，我们选择了更加高效的最近邻插值(nearest interpolation)，进一步提升了算法的效率。



**损失函数用的哪个？**

​	交叉熵损失

# 4. Experiments

**实验一：**验证随机采样的有效性

连续5次降采样，每次采样仅保留原始点云中25%的点。

![image-20200705190131341](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200705190131.png)

**结果分析：**

- 对于小规模的点云$~10^3$, 上述采样方法在计算时间和内存消耗的差距并不明显, 总体来说都是可接受的
- 对于大规模点云$~10^6$, FPS/IDIS/GS所需要的计算时间显著增加, 而CRS需要占用大量的GPU内存(图b虚线)。
- 相比之下，RS在计算时间和内存消耗方面都有着显著的优势，因此非常适合处理大规模点云。这个结果也进一步说明了为什么大多数算法选择在小规模点云上进行处理和优化，主要是因为它们依赖于昂贵的采样方法



**实验二：**验证RandLA-Net的有效性

在SemanticKITTI数据集的验证集(序列8：一共4071帧)进行对比测试。主要评估以下三个方面的指标：总时间，模型参数以及网络最多可处理点数。公平起见，我们在每一帧中将相同数量的点(81920)输入到baseline基准模型以及我们RandLA-Net中。



![image-20200705192627600](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200705192627.png)

​								表1.语义切分的不同方法在SemancKITTI[3]数据集的序列08上的计算时间、网络参数和最大输入点数量。

**结果分析：**

- SPG[23]的模型参数最少，但耗时最长。主要原因是几何划分(geometrical partitioning)和超图构建(super-graph construction)等步骤的计算代价较高；
- PointNet++和PointCNN的耗时也很长，主要原因是FPS在处理大场景点云时比较耗时；
- PointNet和KPConv无法一次性处理非常大规模的点云 ![[公式]](https://www.zhihu.com/equation?tex=%28%5Csim+10%5E6%29) ，主要原因是没有降采样操作(PointNet)或者模型较为复杂。
- 得益于简单的随机采样以及基于MLP的高效的局部特征聚合模块，RandLA-Net的耗时最少(22帧/每秒)，并且能够一次处理总数高达$10^6$的点云。



**实验三：**基准数据集评估结果

​		Semantic3D由30个大规模的户外场景点云组成，包含真实三维空间中160×240×30米的场景，总量高达40亿个点。其中每个点包含3D坐标、RGB信息以及强度信息。RandLA-Net只用了三维坐标以及对应的颜色信息进行处理。从表中可以看出我们的方法达到了非常好的效果，相比于SPG, KPConv等方法都有较明显的提升。

![image-20200705230320304](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200705230320.png)

​																表 2. 不同方法对Semantic3D (reduced-8)的定量结果对比

​		SemanticKITTI数据集由21个序列, 43552帧点云组成。每一帧的点云由$~10^5$个点组成，包含真实三维空间中160×160×20 米的场景。我们按照官方的train-validation-test进行分类，其中序列00-07以及09-10(19130帧)作为训练集，序列08(4071帧)作为验证集，序列11~21(20351帧)用于在线测试。需要注意的是，这个数据集中的点云仅包含各个点的三维坐标，而没有相应的颜色信息。实验结果如下表所示，可以看出：RandLA-Net相比于基于点的方法(表格上半部分)有着显著的提升，同时也优于大部分基于投影的方法，并且在模型参数方面相比于DarKNet53Seg等有着比较明显的优势。

![image-20200705231735317](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200706005011.png)

​														     表 3.   不同方法对SemanticKITTI数据集的定量结果对比

![image-20200706002149239](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200706005012.png)

S3DIS数据集由6个区域的271个房间组成。每个点云包含真实三维空间中20×15×5米的室内场景。6-fold的交叉验证实验结果也进一步证实了我们方法的有效性。

![image-20200706000333844](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200706000333.png)



**消融实验：**

![image-20200706000644974](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20200706000859.png)