# 汇总3D点云语义分割论文（ALL） [![Awesome](img/README/badge.svg)](https://awesome.re)



参考来源：

1、https://github.com/Yochengliu/awesome-point-cloud-analysis

2、[awesome-point-cloud-analysis-2021: A list of papers and datasets about point cloud analysis (processing) since 2017.](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2021#2021)

```diff
- Recent papers (from 2017)
```

# Table of Contents

- [2017](#2017)
- [2018](#2018)
- [2019](#2019)
- [2020](#2020) [CVPR: 70 papers; ECCV: 39 papers]
- [2021](#2021) [CVPR: 60 paper]



![image-20201024220352989](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201124110455.png)

> 1. [[3D点云语义分割综述](3D-point-cloud-review.md)] **from by** [胡庆永]([Deep Learning for 3D Point Clouds：A Survey_20200727版](https://github.com/lizhangjie316/3D-Point-Cloud-Semantic-Segement-Paper/blob/master/papers/Deep%20Learning%20for%203D%20Point%20Clouds%EF%BC%9AA%20Survey_20200727%E7%89%88.pdf))
> 2. [Yochengliu/awesome-point-cloud-analysis from 2017](https://github.com/Yochengliu/awesome-point-cloud-analysis)
> 3. [NUAAXQ/awesome-point-cloud-analysis-2020](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2020#2020)
> 4. [NUAAXQ/awesome-point-cloud-analysis-2021: A list of papers and datasets about point cloud analysis (processing) since 2017.](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2021#2021)



### Keywords

__`dat.`__: dataset &emsp; | &emsp; __`cls.`__: classification &emsp; | &emsp; __`rel.`__: retrieval &emsp; | &emsp; __`seg.`__: segmentation     
__`det.`__: detection &emsp; | &emsp; __`tra.`__: tracking &emsp; | &emsp; __`pos.`__: pose &emsp; | &emsp; __`dep.`__: depth     
__`reg.`__: registration &emsp; | &emsp; __`rec.`__: reconstruction &emsp; | &emsp; __`aut.`__: autonomous driving     
__`oth.`__: other, including normal-related, correspondence, mapping, matching, alignment, compression, generative model...

Statistics: :fire: code is available & stars >= 100 &emsp;|&emsp; :star: citation >= 50



### 2017

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)] PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. [[tensorflow](https://github.com/charlesq34/pointnet)][[pytorch](https://github.com/fxia22/pointnet.pytorch)] [__`cls.`__ __`seg.`__ __`det.`__] :fire: :star:

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yi_SyncSpecCNN_Synchronized_Spectral_CVPR_2017_paper.pdf)] SyncSpecCNN: Synchronized Spectral CNN for 3D Shape Segmentation. [[torch](https://github.com/ericyi/SyncSpecCNN)] [__`seg.`__ __`oth.`__] :star:

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Dai_ScanNet_Richly-Annotated_3D_CVPR_2017_paper.pdf)] ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes. [[project](http://www.scan-net.org/)][[git](http://www.scan-net.org/)] [__`dat.`__ __`cls.`__ __`rel.`__ __`seg.`__ __`oth.`__] :fire: :star:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Riegler_OctNet_Learning_Deep_CVPR_2017_paper.pdf)] OctNet: Learning Deep 3D Representations at High Resolutions. [[torch](https://github.com/griegler/octnet)] [__`cls.`__ __`seg.`__ __`oth.`__] :fire: :star:
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Klokov_Escape_From_Cells_ICCV_2017_paper.pdf)] Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models. [[pytorch](https://github.com/fxia22/kdnet.pytorch)] [__`cls.`__ __`rel.`__ __`seg.`__] :star:
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_3DCNN-DQN-RNN_A_Deep_ICCV_2017_paper.pdf)] 3DCNN-DQN-RNN: A Deep Reinforcement Learning Framework for Semantic Parsing of Large-scale 3D Point Clouds. [[code](https://github.com/CKchaos/scn2pointcloud_tool)] [__`seg.`__]
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qi_3D_Graph_Neural_ICCV_2017_paper.pdf)] 3D Graph Neural Networks for RGBD Semantic Segmentation. [[pytorch](https://github.com/yanx27/3DGNN_pytorch)] [__`seg.`__]
- [[NeurIPS](https://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space)] PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. [[tensorflow](https://github.com/charlesq34/pointnet2)][[pytorch](https://github.com/erikwijmans/Pointnet2_PyTorch)] [__`cls.`__ __`seg.`__] :fire: :star:
- [[ICRA](https://ieeexplore.ieee.org/document/7989591)] Fast segmentation of 3D point clouds: A paradigm on LiDAR data for autonomous vehicle applications. [[code](https://github.com/VincentCheungM/Run_based_segmentation)] [__`seg.`__ __`aut.`__]
- [[ICRA](https://ieeexplore.ieee.org/document/7989618)] SegMatch: Segment based place recognition in 3D point clouds. [__`seg.`__ __`oth.`__]
- [[3DV](http://segcloud.stanford.edu/segcloud_2017.pdf)] SEGCloud: Semantic Segmentation of 3D Point Clouds. [[project](http://segcloud.stanford.edu/)] [__`seg.`__ __`aut.`__] :star:



### 2018

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Su_SPLATNet_Sparse_Lattice_CVPR_2018_paper.pdf)] SPLATNet: Sparse Lattice Networks for Point Cloud Processing. [[caffe](https://github.com/NVlabs/splatnet)] [__`seg.`__] :fire:

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xie_Attentional_ShapeContextNet_for_CVPR_2018_paper.pdf)] Attentional ShapeContextNet for Point Cloud Recognition. [__`cls.`__ __`seg.`__]

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Mining_Point_Cloud_CVPR_2018_paper.pdf)] Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling. [[code](http://www.merl.com/research/license#KCNet)] [__`cls.`__ __`seg.`__]

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hua_Pointwise_Convolutional_Neural_CVPR_2018_paper.pdf)] Pointwise Convolutional Neural Networks. [[tensorflow](https://github.com/scenenn/pointwise)] [__`cls.`__ __`seg.`__]

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.pdf)] SO-Net: Self-Organizing Network for Point Cloud Analysis. [[pytorch](https://github.com/lijx10/SO-Net)] [__`cls.`__ __`seg.`__] :fire: :star:

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Recurrent_Slice_Networks_CVPR_2018_paper.pdf)] Recurrent Slice Networks for 3D Segmentation of Point Clouds. [[pytorch](https://github.com/qianguih/RSNet)] [__`seg.`__]

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Graham_3D_Semantic_Segmentation_CVPR_2018_paper.pdf)] 3D Semantic Segmentation with Submanifold Sparse Convolutional Networks. [[pytorch](https://github.com/facebookresearch/SparseConvNet)] [__`seg.`__] :fire:

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Deep_Parametric_Continuous_CVPR_2018_paper.pdf)] Deep Parametric Continuous Convolutional Neural Networks. [__`seg.`__ __`aut.`__]

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SGPN_Similarity_Group_CVPR_2018_paper.pdf)] SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation. [[tensorflow](https://github.com/laughtervv/SGPN)] [__`seg.`__] :fire:

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Landrieu_Large-Scale_Point_Cloud_CVPR_2018_paper.pdf)] Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs. [[pytorch](https://github.com/loicland/superpoint_graph)] [__`seg.`__] :fire:

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lawin_Density_Adaptive_Point_CVPR_2018_paper.pdf)] Density Adaptive Point Set Registration. [[code](https://github.com/felja633/DARE)] [__`reg.`__]

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Birdal_A_Minimalist_Approach_CVPR_2018_paper.pdf)] A Minimalist Approach to Type-Agnostic Detection of Quadrics in Point Clouds. [__`seg.`__]

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Le_PointGrid_A_Deep_CVPR_2018_paper.pdf)] PointGrid: A Deep Network for 3D Shape Understanding. [[tensorflow](https://github.com/trucleduc/PointGrid)] [__`cls.`__ __`seg.`__]

- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tatarchenko_Tangent_Convolutions_for_CVPR_2018_paper.pdf)] Tangent Convolutions for Dense Prediction in 3D. [[tensorflow](https://github.com/tatarchm/tangent_conv)] [__`seg.`__ __`aut.`__]

- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaoqing_Ye_3D_Recurrent_Neural_ECCV_2018_paper.pdf)] 3D Recurrent Neural Networks with Context Fusion for Point Cloud Semantic Segmentation. [__`seg.`__]

- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chu_Wang_Local_Spectral_Graph_ECCV_2018_paper.pdf)] Local Spectral Graph Convolution for Point Set Feature Learning. [[tensorflow](https://github.com/fate3439/LocalSpecGCN)] [__`cls.`__ __`seg.`__]

- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yifan_Xu_SpiderCNN_Deep_Learning_ECCV_2018_paper.pdf)] SpiderCNN: Deep Learning on Point Sets with Parameterized Convolutional Filters. [[tensorflow](https://github.com/xyf513/SpiderCNN)] [__`cls.`__ __`seg.`__]

- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Dario_Rethage_Fully-Convolutional_Point_Networks_ECCV_2018_paper.pdf)] Fully-Convolutional Point Networks for Large-Scale Point Clouds. [[tensorflow](https://github.com/drethage/fully-convolutional-point-network)] [__`seg.`__ __`oth.`__]

- [[ECCVW](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Zeng_3DContextNet_K-d_Tree_Guided_Hierarchical_Learning_of_Point_Clouds_Using_ECCVW_2018_paper.pdf)] 3DContextNet: K-d Tree Guided Hierarchical Learning of Point Clouds Using Local and Global Contextual Cues. [__`cls.`__ __`seg.`__]

- [[NeurIPS](https://papers.nips.cc/paper/7362-pointcnn-convolution-on-x-transformed-points)] PointCNN: Convolution On X-Transformed Points. [[tensorflow](https://github.com/yangyanli/PointCNN)][[pytorch](https://github.com/hxdengBerkeley/PointCNN.Pytorch)] [__`cls.`__ __`seg.`__] :fire:

- [[TOG](https://dl.acm.org/ft_gateway.cfm?id=3201301&ftid=1991771&dwn=1&CFID=155708095&CFTOKEN=598df826a5b545a7-3E7CE91C-DE12-F588-FAEEF2551115E64E)] Point Convolutional Neural Networks by Extension Operators. [[tensorflow](https://github.com/matanatz/pcnn)] [__`cls.`__ __`seg.`__]

- [[SIGGRAPH Asia](https://arxiv.org/abs/1806.01759)] Monte Carlo Convolution for Learning on Non-Uniformly Sampled Point Clouds. [[tensorflow](https://github.com/viscom-ulm/MCCNN)] [__`cls.`__ __`seg.`__ __`oth.`__]

- [[SIGGRAPH](https://arxiv.org/abs/1706.04496)] Learning local shape descriptors from part correspondences with multi-view convolutional networks. [[project](https://people.cs.umass.edu/~hbhuang/local_mvcnn/index.html)] [__`seg.`__ __`oth.`__] 

- [[MM](https://arxiv.org/abs/1806.02952)] RGCNN: Regularized Graph CNN for Point Cloud Segmentation. [[tensorflow](https://github.com/tegusi/RGCNN)] [__`seg.`__]

- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460837)] Multi-View 3D Entangled Forest for Semantic Segmentation and Mapping. [__`seg.`__ __`oth.`__]

- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8462926)] SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud. [[tensorflow](https://github.com/priyankanagaraj1494/Squeezseg)] [__`seg.`__ __`aut.`__]

- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594356)] Extracting Phenotypic Characteristics of Corn Crops Through the Use of Reconstructed 3D Models. [__`seg.`__ __`rec.`__]

- [[ACCV](https://arxiv.org/abs/1803.07289)] Flex-Convolution (Million-Scale Point-Cloud Learning Beyond Grid-Worlds). [[tensorflow](https://github.com/cgtuebingen/Flex-Convolution)] [__`seg.`__]

  

---

### 2019

- [[CVPR](http://export.arxiv.org/abs/1904.07601)] Relation-Shape Convolutional Neural Network for Point Cloud Analysis. [[pytorch](https://github.com/Yochengliu/Relation-Shape-CNN)] [__`cls.`__ __`seg.`__ __`oth.`__] :fire:
- [[CVPR](https://raoyongming.github.io/files/SFCNN.pdf)] Spherical Fractal Convolutional Neural Networks for Point Cloud Recognition. [__`cls.`__ __`seg.`__]
- [[CVPR](https://arxiv.org/abs/1904.03375v1)] Modeling Point Clouds with Self-Attention and Gumbel Subset Sampling. [__`cls.`__ __`seg.`__]
- [[CVPR](http://export.arxiv.org/abs/1904.08017)] A-CNN: Annularly Convolutional Neural Networks on Point Clouds. [[tensorflow](https://github.com/artemkomarichev/a-cnn)] [__`cls.`__ __`seg.`__]
- [[CVPR](https://arxiv.org/abs/1811.07246)] PointConv: Deep Convolutional Networks on 3D Point Clouds. [[tensorflow](https://github.com/DylanWusee/pointconv)] [__`cls.`__ __`seg.`__] :fire:
- [[CVPR](https://arxiv.org/abs/1812.11647)] Path-Invariant Map Networks. [[tensorflow](https://github.com/zaiweizhang/path_invariance_map_network)] [__`seg.`__ __`oth.`__]
- [[CVPR](https://arxiv.org/abs/1812.02713)] PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding. [[code](https://github.com/daerduoCarey/partnet_dataset)] [__`dat.`__ __`seg.`__]
- [[CVPR](https://arxiv.org/abs/1902.09852)] Associatively Segmenting Instances and Semantics in Point Clouds. [[tensorflow](https://github.com/WXinlong/ASIS)] [__`seg.`__] :fire:
- [[CVPR](https://arxiv.org/abs/1903.00343)] Octree guided CNN with Spherical Kernels for 3D Point Clouds. [[extension](https://arxiv.org/pdf/1909.09287.pdf)] [[code](https://github.com/hlei-ziyan/SPH3D-GCN)] [__`cls.`__ __`seg.`__]
- [[CVPR](https://arxiv.org/abs/1904.00699v1)] JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds with Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields. [[pytorch](https://github.com/pqhieu/JSIS3D)] [__`seg.`__]
- [[CVPR](https://arxiv.org/abs/1904.02113)] Point Cloud Oversegmentation with Graph-Structured Deep Metric Learning. [__`seg.`__]
- [[CVPR](https://arxiv.org/abs/1903.00709)] PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation. [[pytorch](https://github.com/FoggYu/PartNet)] [__`dat.`__ __`seg.`__] 
- [[CVPR](http://export.arxiv.org/abs/1904.08755)] 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks. [[pytorch](https://github.com/StanfordVL/MinkowskiEngine)] [__`seg.`__] :fire:
- [[CVPR](https://arxiv.org/abs/1811.10136)] FilterReg: Robust and Efficient Probabilistic Point-Set Registration using Gaussian Filter and Twist Parameterization. [[code](https://bitbucket.org/gaowei19951004/poser/src/master/)] [__`reg.`__]
- [[CVPR](http://jiaya.me/papers/pointweb_cvpr19.pdf)] PointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing. [[pytorch](https://github.com/hszhao/PointWeb)] [__`cls.`__ __`seg.`__]
- [[CVPR](https://arxiv.org/abs/1812.03320)] GSPN: Generative Shape Proposal Network for 3D Instance Segmentation in Point Cloud. [__`seg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Graph_Attention_Convolution_for_Point_Cloud_Semantic_Segmentation_CVPR_2019_paper.pdf)] Graph Attention Convolution for Point Cloud Semantic Segmentation. [__`seg.`__]
- [[CVPR](https://arxiv.org/pdf/1904.03498.pdf)] LP-3DCNN: Unveiling Local Phase in 3D Convolutional Neural Networks. [[project](https://sites.google.com/view/lp-3dcnn/home)] [__`cls.`__ __`seg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Duan_Structural_Relational_Reasoning_of_Point_Clouds_CVPR_2019_paper.pdf)] Structural Relational Reasoning of Point Clouds. [__`cls.`__ __`seg.`__]
- [[ICCV](https://arxiv.org/abs/1904.03751)] DeepGCNs: Can GCNs Go as Deep as CNNs? [[tensorflow](https://github.com/lightaime/deep_gcns)] [[pytorch]](https://github.com/lightaime/deep_gcns_torch) [__`seg.`__] :fire:
- [[ICCV](https://arxiv.org/abs/1904.08889)] KPConv: Flexible and Deformable Convolution for Point Clouds. [[tensorflow](https://github.com/HuguesTHOMAS/KPConv)] [__`cls.`__ __`seg.`__] :fire:
- [[ICCV](https://arxiv.org/pdf/1908.06295.pdf)] ShellNet: Efficient Point Cloud Convolutional Neural Networks using Concentric Shells Statistics. [[project](https://hkust-vgd.github.io/shellnet/)] [__`seg.`__]
- [[ICCV](https://arxiv.org/abs/1909.03669)] DensePoint: Learning Densely Contextual Representation for Efficient Point Cloud Processing. [[pytorch](https://github.com/Yochengliu/DensePoint)] [__`cls.`__ __`seg.`__ __`oth.`__]
- [[ICCV](https://arxiv.org/pdf/1909.10469.pdf)] Hierarchical Point-Edge Interaction Network for Point Cloud Semantic Segmentation. [__`seg.`__]
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Mao_Interpolated_Convolutional_Networks_for_3D_Point_Cloud_Understanding_ICCV_2019_paper.pdf)] Interpolated Convolutional Networks for 3D Point Cloud Understanding. [__`cls.`__ __`seg.`__]
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Dynamic_Points_Agglomeration_for_Hierarchical_Point_Sets_Learning_ICCV_2019_paper.pdf)] Dynamic Points Agglomeration for Hierarchical Point Sets Learning. [[pytorch](https://github.com/yuyi1005/DPAM)] [__`cls.`__ __`seg.`__]
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Hassani_Unsupervised_Multi-Task_Feature_Learning_on_Point_Clouds_ICCV_2019_paper.pdf)] Unsupervised Multi-Task Feature Learning on Point Clouds. [__`cls.`__ __`seg.`__]
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Meng_VV-Net_Voxel_VAE_Net_With_Group_Convolutions_for_Point_Cloud_ICCV_2019_paper.pdf)] VV-NET: Voxel VAE Net with Group Convolutions for Point Cloud Segmentation. [[tensorflow](https://github.com/xianyuMeng/VV-Net-Voxel-VAE-Net-with-Group-Convolutions-for-Point-Cloud-Segmentation)] [__`seg.`__]
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_MeteorNet_Deep_Learning_on_Dynamic_3D_Point_Cloud_Sequences_ICCV_2019_paper.pdf)] MeteorNet: Deep Learning on Dynamic 3D Point Cloud Sequences. [[code](https://github.com/xingyul/meteornet)] [__`cls.`__ __`seg.`__ __`oth.`__]
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lahoud_3D_Instance_Segmentation_via_Multi-Task_Metric_Learning_ICCV_2019_paper.pdf)] 3D Instance Segmentation via Multi-Task Metric Learning. [[code](https://sites.google.com/view/3d-instance-mtml)] [__`seg.`__]
- [[NeurIPS](https://arxiv.org/abs/1906.01140)] Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds. [[tensorflow](https://github.com/Yang7879/3D-BoNet)] [__`det.`__ __`seg.`__]
- [[NeurIPS](http://papers.nips.cc/paper/8706-exploiting-local-and-global-structure-for-point-cloud-semantic-segmentation-with-contextual-point-representations.pdf)] Exploiting Local and Global Structure for Point Cloud Semantic Segmentation with Contextual Point Representations. [[tensorflow](https://github.com/fly519/ELGS)] [__`seg.`__]
- [[NeurIPS](https://arxiv.org/pdf/1907.03739.pdf)] Point-Voxel CNN for Efficient 3D Deep Learning. [__`det.`__ __`seg.`__ __`aut.`__]
- [[AAAI](https://arxiv.org/abs/1811.02565)] Point2Sequence: Learning the Shape Representation of 3D Point Clouds with an Attention-based Sequence to Sequence Network. [[tensorflow](https://github.com/liuxinhai/Point2Sequence)] [__`cls.`__ __`seg.`__]
- [[TOG](https://arxiv.org/abs/1801.07829)] Dynamic Graph CNN for Learning on Point Clouds. [[tensorflow](https://github.com/WangYueFt/dgcnn)][[pytorch](https://github.com/WangYueFt/dgcnn)] [__`cls.`__ __`seg.`__] :fire: :star:
- [[SIGGRAPH Asia](https://dl.acm.org/doi/10.1145/3355089.3356573)] RPM-Net: recurrent prediction of motion and parts from point cloud. [[tensorflow](https://github.com/Salingo/RPM-Net)] [__`seg.`__]
- [[SIGGRAPH Asia](https://arxiv.org/abs/1908.00575v1)] StructureNet: Hierarchical Graph Networks for 3D Shape Generation. [__`seg.`__ __`oth.`__]
- [[MM](https://dl.acm.org/citation.cfm?id=3351042)] SRINet: Learning Strictly Rotation-Invariant Representations for Point Cloud Classification and Segmentation. [[tensorflow](https://github.com/tasx0823/SRINet)] [__`cls.`__ __`seg.`__]
- [[MM](https://dl.acm.org/citation.cfm?id=3351076)] Ground-Aware Point Cloud Semantic Segmentation for Autonomous Driving. [[code](https://github.com/Jaiy/Ground-aware-Seg)] [__`seg.`__ __`aut.`__]
- [[ICRA](https://arxiv.org/abs/1809.08495)] SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud. [[tensorflow](https://github.com/xuanyuzhou98/SqueezeSegV2)] [__`seg.`__ __`aut.`__]
- [[ICRA](https://arxiv.org/abs/1905.02553)] Oriented Point Sampling for Plane Detection in Unorganized Point Clouds. [__`det.`__ __`seg.`__]
- [[ICRA](https://arxiv.org/abs/1809.06267)] PointNetGPD: Detecting Grasp Configurations from Point Sets. [[pytorch](https://github.com/lianghongzhuo/PointNetGPD)] [__`det.`__ __`seg.`__]
- [[ICRA](https://ras.papercept.net/conferences/conferences/ICRA19/program/ICRA19_ContentListWeb_3.html)] Robust 3D Object Classification by Combining Point Pair Features and Graph Convolution. [__`cls.`__ __`seg.`__]
- [[ICRA](https://ras.papercept.net/conferences/conferences/ICRA19/program/ICRA19_ContentListWeb_3.html)] Hierarchical Depthwise Graph Convolutional Neural Network for 3D Semantic Segmentation of Point Clouds. [__`seg.`__]
- [[IROS](https://arxiv.org/pdf/1909.01643v1.pdf)] PASS3D: Precise and Accelerated Semantic Segmentation for 3D Point Cloud. [__`seg.`__ __`aut.`__]
- [[IV](https://arxiv.org/abs/1906.10964)] End-to-End 3D-PointCloud Semantic Segmentation for Autonomous Driving. [__`seg.`__] [__`aut.`__]
- [[Eurographics Workshop](https://arxiv.org/abs/1904.02375)] Generalizing Discrete Convolutions for Unstructured Point Clouds. [[pytorch](https://github.com/aboulch/ConvPoint)] [__`cls.`__ __`seg.`__]
- [[3DV](https://arxiv.org/pdf/1908.06297.pdf)] Rotation Invariant Convolutions for 3D Point Clouds Deep Learning. [[project](https://hkust-vgd.github.io/riconv/)] [__`cls.`__ __`seg.`__]
- [[3DV](https://arxiv.org/abs/1906.11555)] Effective Rotation-invariant Point CNN with Spherical Harmonics kernels. [[tensorflow](https://github.com/adrienPoulenard/SPHnet)] [__`cls.`__ __`seg.`__ __`oth.`__]



---

### 2020 

- [[AAAI](https://arxiv.org/abs/1912.10775)] Point2Node: Correlation Learning of Dynamic-Node for Point Cloud Feature Modeling. [__`seg.`__ __`cls.`__]
- [[AAAI](https://arxiv.org/abs/1811.09361)] PRIN: Pointwise Rotation-Invariant Network. [__`seg.`__ __`cls.`__]
- [[AAAI](https://arxiv.org/pdf/1912.09654.pdf)] JSNet: Joint Instance and Semantic Segmentation of 3D Point Clouds. [[tensorflow](https://github.com/dlinzhao/JSNet)][__`seg.`__][__`seg.`__] 
- [[CVPR](https://arxiv.org/pdf/1911.11236.pdf)] RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds. [[tensorflow](https://github.com/QingyongHu/RandLA-Net)] [__`seg.`__] :fire:
- [[CVPR](https://arxiv.org/pdf/2003.06233.pdf)] Fusion-Aware Point Convolution for Online Semantic 3D Scene Segmentation. [[pytorch](https://github.com/jzhzhang/FusionAwareConv)] [__`seg.`__] 
- [[CVPR](https://arxiv.org/pdf/1903.10297.pdf)] AdaCoSeg: Adaptive Shape Co-Segmentation with Group Consistency Loss. [__`seg.`__]
- [[CVPR](https://arxiv.org/pdf/2003.01251.pdf)] Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud. [[tensorflow](https://github.com/WeijingShi/Point-GNN)][__`det.`__] :fire:
- [[CVPR](https://arxiv.org/pdf/2003.13035.pdf)] Multi-Path Region Mining For Weakly Supervised 3D Semantic Segmentation on Point Clouds. [__`seg.`__] 
- [[CVPR](https://arxiv.org/pdf/2003.13479.pdf)] RPM-Net: Robust Point Matching using Learned Features. [[code](https://github.com/yewzijian/RPMNet)] [__`seg.`__] 
- [[CVPR](https://arxiv.org/pdf/2003.14032.pdf)] PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation. [[pytorch](https://github.com/edwardzhou130/PolarSeg)] [__`seg.`__] 
- [[CVPR](https://arxiv.org/pdf/2003.13867.pdf)] 3D-MPA: Multi Proposal Aggregation for 3D Semantic Instance Segmentation. [__`seg.`__]
- [[CVPR](https://arxiv.org/pdf/2003.06537.pdf)] OccuSeg: Occupancy-aware 3D Instance Segmentation. [__`seg.`__]
- [[CVPR](https://arxiv.org/pdf/2003.05593.pdf)] Learning to Segment 3D Point Clouds in 2D Image Space. [[pytorch](https://github.com/Zhang-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space)] [__`seg`__]
- [[CVPR](https://arxiv.org/pdf/2004.01658.pdf)] PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation. [__`seg.`__]
- [[CVPR](https://arxiv.org/abs/2004.02869)] DualSDF: Semantic Shape Manipulation using a Two-Level Representation. [[code](https://github.com/zekunhao1995/DualSDF)] [__`seg`__]
- **[[CVPR](https://arxiv.org/abs/2005.01939)] From Image Collections to Point Clouds with Self-supervised Shape and Pose Networks. [[tensorflow](https://github.com/val-iisc/ssl_3d_recon)] ['image-to-point cloud.']**
- [[CVPR](https://arxiv.org/abs/1911.12676)] xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation. [__`Segmentation`__]
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_End-to-End_3D_Point_Cloud_Instance_Segmentation_Without_Detection_CVPR_2020_paper.pdf)] End-to-End 3D Point Cloud Instance Segmentation Without Detection. [__`Segmentation`__]
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Weakly_Supervised_Semantic_Point_Cloud_Segmentation_Towards_10x_Fewer_Labels_CVPR_2020_paper.pdf)] Weakly Supervised Semantic Point Cloud Segmentation: Towards 10x Fewer Labels. [__`Segmentation`__]
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2020/papers/Lei_SegGCN_Efficient_3D_Point_Cloud_Segmentation_With_Fuzzy_Spherical_Kernel_CVPR_2020_paper.pdf)] SegGCN: Efficient 3D Point Cloud Segmentation With Fuzzy Spherical Kernel. [__`Segmentation`__]
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_SpSequenceNet_Semantic_Segmentation_Network_on_4D_Point_Clouds_CVPR_2020_paper.pdf)] SpSequenceNet: Semantic Segmentation Network on 4D Point Clouds. [__`Segmentation`__]
- **[[ECCV](https://arxiv.org/abs/2002.10277)] PUGeo-Net: A Geometry-centric Network for 3D Point Cloud Upsampling. [`Upsampling`]**
- **[[ECCV](https://arxiv.org/abs/2007.02578)] Learning Graph-Convolutional Representations for Point Cloud Denoising. [`Denoising`]**
- [[ECCV](https://arxiv.org/pdf/2007.06888.pdf)] JSENet: Joint Semantic Segmentation and Edge Detection Network for 3D Point Clouds. [[code](https://github.com/hzykent/JSENet)] [__`Segmentation`__]
- [[ECCV](https://arxiv.org/pdf/2007.13344.pdf)] Self-Prediction for Joint Instance and Semantic Segmentation of Point Clouds. [__`Segmentation`__]
- [[ECCV](https://arxiv.org/pdf/2007.13138.pdf)] Virtual Multi-view Fusion for 3D Semantic Segmentation. [__`Segmentation`__]
- [[ECCV](https://arxiv.org/pdf/2007.16100.pdf)] Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution. [__`Segmentation`__]
- **[[ECCV](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650460.pdf)] Rotation-robust Intersection over Union for 3D Object Detection. [`3D IOU`]**
- [[ECCV](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720494.pdf)] Efficient Outdoor 3D Point Cloud Semantic Segmentation for Critical Road Objects and Distributed Contexts. [__`Segmentation`__]
- [[ECCV](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690630.pdf)] Deep FusionNet for Point Cloud Semantic Segmentation. [[code](https://github.com/feihuzhang/LiDARSeg)] [__`Segmentation`__]
- [[ECCV](https://arxiv.org/abs/2004.01803)] SqueezeSegV3: Spatially-Adaptive Convolution for Efficient Point-Cloud Segmentation. [[code](https://github.com/chenfengxu714/SqueezeSegV3)] [__`seg.`__]

- [[IROS](https://arxiv.org/abs/2011.00988)] PBP-Net: Point Projection and Back-Projection Network for 3D Point Cloud Segmentation. [__`Segmentation `__]
- [[IROS](http://ras.papercept.net/images/temp/IROS/files/0144.pdf)] RegionNet: Region-feature-enhanced 3D Scene Understanding Network with Dual Spatial-aware Discriminative Loss. [__`Segmentation `__]
- [[IROS](https://arxiv.org/pdf/2007.15488.pdf)] Cascaded Non-local Neural Network for Point Cloud Semantic Segmentation. [__`Segmentation`__]
- [[ACM MM](https://arxiv.org/pdf/2008.04968.pdf)] Campus3D: A Photogrammetry Point Cloud Benchmark for Hierarchical Understanding of Outdoor Scene. [__`Understanding`__]
- [[WACV](https://arxiv.org/pdf/1912.08487.pdf)] FuseSeg: LiDAR Point Cloud Segmentation Fusing Multi-Modal Data. [__`seg.`__ __`aut.`__]
- [[WACV](http://openaccess.thecvf.com/content_WACV_2020/papers/Ma_Global_Context_Reasoning_for_Semantic_Segmentation_of_3D_Point_Clouds_WACV_2020_paper.pdf)] Global Context Reasoning for Semantic Segmentation of 3D Point Clouds. [__`seg.`__]
- [[BMVC](https://arxiv.org/pdf/2008.05149.pdf)] ASAP-Net: Attention and Structure Aware Point Cloud Sequence Segmentation. [__`Segmentation`__]
- [[ICRA](https://arxiv.org/pdf/2003.08624.pdf)] DeepTemporalSeg: Temporally Consistent Semantic Segmentation of 3D LiDAR Scans. [__`seg.`__]
- [[Master Thesis](https://pdfs.semanticscholar.org/4303/8a62b3e3b2f44d7a9cc50ff69e7586a758cc.pdf)] Neighborhood Pooling in Graph Neural Networks for 3D and 4D Semantic Segmentation. [__'seg.'__]



---

### 2021

- [[CVPR](https://arxiv.org/abs/2009.03137)] Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges. [[code](https://github.com/QingyongHu/SensatUrban)] [__`Segmentation`__]
- [[CVPR](https://arxiv.org/abs/2103.07969)] Monte Carlo Scene Search for 3D Scene Understanding. [__`Understanding`__]
- [[CVPR](https://arxiv.org/abs/2102.04530)] AF2-S3Net: Attentive Feature Fusion with Adaptive Feature Selection for Sparse Semantic Segmentation Network. [__`Segmentation`__]
- [[CVPR](https://arxiv.org/abs/2103.14147)] Equivariant Point Network for 3D Point Cloud Analysis. [__`Analysis`__]
- [[CVPR](https://arxiv.org/abs/2103.14635)] PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds. [[code](https://github.com/CVMI)] [__`Convolution`__]
- [[CVPR](http://arxiv.org/abs/1912.00145)] Point Cloud Instance Segmentation using Probabilistic Embeddings. [__`Segmentation`__]
- [[CVPR](https://arxiv.org/abs/2103.14962)] Panoptic-PolarNet: Proposal-free LiDAR Point Cloud Panoptic Segmentation. [__`Segmentation`__]
- [[CVPR oral](https://hehefan.github.io/pdfs/p4transformer.pdf)] Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos. [[pytorch](https://github.com/hehefan/P4Transformer)] [__`Transformer`__]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Qiu_Semantic_Segmentation_for_Real_Point_Cloud_Scenes_via_Bilateral_Augmentation_CVPR_2021_paper.pdf)] Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion. [__`Segmentation`__]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Yi_Complete__Label_A_Domain_Adaptation_Approach_to_Semantic_Segmentation_CVPR_2021_paper.pdf)] Complete & Label: A Domain Adaptation Approach to Semantic Segmentation of LiDAR Point Clouds. [__`Segmentation`__]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Qian_PU-GCN_Point_Cloud_Upsampling_Using_Graph_Convolutional_Networks_CVPR_2021_paper.pdf)] PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks. [[code](https://github.com/guochengqian/PU-GCN)] [__`Upsampling`__]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_SCF-Net_Learning_Spatial_Contextual_Features_for_Large-Scale_Point_Cloud_Segmentation_CVPR_2021_paper.pdf)] SCF-Net: Learning Spatial Contextual Features for Large-Scale Point Cloud Segmentation. [__`Segmentation`__]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Eckart_Self-Supervised_Learning_on_3D_Point_Clouds_by_Learning_Discrete_Generative_CVPR_2021_paper.pdf)] Self-Supervised Learning on 3D Point Clouds by Learning Discrete Generative Models. [__`Self-Supervised`__]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/He_DyCo3D_Robust_Instance_Segmentation_of_3D_Point_Clouds_Through_Dynamic_CVPR_2021_paper.pdf)] DyCo3D: Robust Instance Segmentation of 3D Point Clouds Through Dynamic Convolution. [[code](https://git.io/DyCo3D)] [__`Segmentation`__]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Pang_TearingNet_Point_Cloud_Autoencoder_To_Learn_Topology-Friendly_Representations_CVPR_2021_paper.pdf)] TearingNet: Point Cloud Autoencoder To Learn Topology-Friendly Representations. [__`Autoencoder`__]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Lu_CGA-Net_Category_Guided_Aggregation_for_Point_Cloud_Semantic_Segmentation_CVPR_2021_paper.pdf)] CGA-Net: Category Guided Aggregation for Point Cloud Semantic Segmentation. [__`Segmentation`__]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Point_Cloud_Upsampling_via_Disentangled_Refinement_CVPR_2021_paper.pdf)] Point Cloud Upsampling via Disentangled Refinement. [[code](https://github.com/liruihui/Dis-PU)] [__`Upsampling`__]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Few-Shot_3D_Point_Cloud_Semantic_Segmentation_CVPR_2021_paper.pdf)] Few-shot 3D Point Cloud Semantic Segmentation. [[code](https://github.com/Na-Z/attMPTI)] [__`Segmentation`__]



---

### arXiv

#### 2018

- [[arXiv](https://arxiv.org/abs/1807.00652)] PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation. [[tensorflow](https://github.com/MVIG-SJTU/pointSIFT)] [__`seg.`__] :fire:
- [[arXiv](https://arxiv.org/abs/1811.11209)] Iterative Transformer Network for 3D Point Cloud. [__`cls.`__ __`seg.`__ __`pos.`__]
- [[arXiv](https://arxiv.org/abs/1812.11029)] Multi-column Point-CNN for Sketch Segmentation. [__`seg.`__]
- [[arXiv](https://arxiv.org/abs/1810.01151)] Know What Your Neighbors Do: 3D Semantic Segmentation of Point Clouds. [__`seg.`__]

#### 2019

- [[arxiv](https://arxiv.org/abs/1901.08396)] Context Prediction for Unsupervised Deep Learning on Point Clouds. [__`cls.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/abs/1902.05247)] 3D Graph Embedding Learning with a Structure-aware Loss Function for Point Cloud Semantic Instance Segmentation. [__`seg.`__]
- [[arXiv](https://arxiv.org/abs/1904.00230)] MortonNet: Self-Supervised Learning of Local Features in 3D Point Clouds. [__`cls.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/pdf/1904.10014.pdf)] Linked Dynamic Graph CNN: Learning on Point Cloud via Linking Hierarchical Features. [__`cls.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/abs/1905.08705)] GAPNet: Graph Attention based Point Neural Network for Exploiting Local Feature of Point Cloud. [[tensorflow](https://github.com/FrankCAN/GAPNet)] [__`cls.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/abs/1906.01140)] Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds. [[tensorflow](https://github.com/Yang7879/3D-BoNet)] [__`det.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/abs/1906.10887)] Spatial Transformer for 3D Points. [__`seg.`__]
- [[arXiv](https://arxiv.org/abs/1907.03739)] Point-Voxel CNN for Efficient 3D Deep Learning. [__`seg.`__ __`det.`__ __`aut.`__]
- [[arXiv](https://arxiv.org/pdf/1905.07650v1.pdf)] SAWNet: A Spatially Aware Deep Neural Network for 3D Point Cloud Processing. [[tensorflow](https://github.com/balwantraikekutte/SAWNet)] [__`cls.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/pdf/1906.03299.pdf)] PyramNet: Point Cloud Pyramid Attention Network and Graph Embedding Module for Classification and Segmentation. [__`cls.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/abs/1907.09798)] PointAtrousGraph: Deep Hierarchical Encoder-Decoder with Point Atrous Convolution for Unorganized 3D Points. [[tensorflow](https://github.com/paul007pl/PointAtrousGraph)] [__`cls.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/pdf/1908.11026.pdf)] Point2SpatialCapsule: Aggregating Features and Spatial Relationships of Local Regions on Point Clouds using Spatial-aware Capsules. [__`cls.`__ __`rel.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/pdf/1907.13079.pdf)] Deformable Filter Convolution for Point Cloud Reasoning. [__`seg.`__ __`det.`__ __`aut.`__]
- [[arXiv](https://arxiv.org/pdf/1912.02984v1.pdf)] Grid-GCN for Fast and Scalable Point Cloud Learning. [__`seg.`__ __`cls.`__]
- [[arXiv](https://arxiv.org/pdf/1911.10150.pdf)] PointPainting: Sequential Fusion for 3D Object Detection. [__`seg.`__ __`det.`__]
- [[arXiv](https://arxiv.org/pdf/1912.10644.pdf)] Geometry Sharing Network for 3D Point Cloud Classification and Segmentation. [[pytorch](https://github.com/MingyeXu/GS-Net)] [__`cls.`__ __`seg.`__]
- [[arvix](https://arxiv.org/abs/1912.12033)] Deep Learning for 3D Point Clouds: A Survey. [[code](https://github.com/QingyongHu/SoTA-Point-Cloud)] [__`cls.`__ __`det.`__ __`tra.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/pdf/1909.12663.pdf)] Point Attention Network for Semantic Segmentation of 3D Point Clouds. [__`seg.`__]

#### 2020

- [[arXiv](https://arxiv.org/pdf/2003.03653.pdf)] SalsaNext: Fast Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving. [[code](https://github.com/TiagoCortinhal/SalsaNext)] [__`seg.`__]
- [[arXiv](https://arxiv.org/pdf/2003.06233.pdf)] Feature Fusion Network Based on Attention Mechanism for 3D Semantic Segmentation of Point Clouds. [__`seg.`__]
- [[arXiv](https://www.sciencedirect.com/science/article/abs/pii/S0925231220304070)] Multi-view Semantic Learning Network for Point Cloud Based 3D Object Detection. [__`seg.`__]
- [[arXiv](https://arxiv.org/pdf/2003.08284.pdf)] Toronto-3D: A Large-scale Mobile LiDAR Dataset for Semantic Segmentation of Urban Roadways. [[code](https://github.com/WeikaiTan/Toronto-3D)] [__`seg.`__]
- [[arXiv](https://arxiv.org/pdf/2003.12841.pdf)] A Benchmark for Point Clouds Registration Algorithms. [[code](https://github.com/iralabdisco/point_clouds_registration_benchmark)] [__`seg.`__]
- [[arXiv](https://arxiv.org/pdf/2003.05420.pdf)] Bi-Directional Attention for Joint Instance and Semantic Segmentation in Point Clouds. [[pytorch](https://github.com/pumpkinnan/BAN)] [__`seg.`__]
- [[arXiv](https://arxiv.org/pdf/2003.13926.pdf)] Scene Context Based Semantic Segmentation for 3D LiDAR Data in Dynamic Scene. [__`seg.`__]
- [[arXiv](https://arxiv.org/abs/2004.02724)] Reconfigurable Voxels: A New Representation for LiDAR-Based Point Clouds. [__`seg.`__]
- [[arXiv](https://arxiv.org/abs/2004.03401)] MNEW: Multi-domain Neighborhood Embedding and Weighting for Sparse Point Clouds Segmentation. [__`seg.`__]
- **[[arXiv](https://arxiv.org/abs/2004.05224)] Deep Learning for Image and Point Cloud Fusion in Autonomous Driving: A Review. [`review.`]**
- [[arXiv](https://arxiv.org/abs/2004.11784)] DPDist : Comparing Point Clouds Using Deep Point Cloud Distance. [__`seg.`__]
- [[arXiv](https://arxiv.org/abs/2004.12498)] Weakly Supervised Semantic Segmentation in 3D Graph-Structured Point Clouds of Wild Scenes. [__`seg.`__]
- [[arXiv](https://arxiv.org/abs/2005.06734)] Dense-Resolution Network for Point Cloud Classification and Segmentation.[[code](https://github.com/ShiQiu0419/DRNet)] [__`segmentation.`__]
- [[arXiv](https://arxiv.org/abs/2005.06667)] Exploiting Multi-Layer Grid Maps for Surround-View Semantic Segmentation of Sparse LiDAR Data. [__`segmentation.`__]
- [[arXiv](https://arxiv.org/abs/2005.09830)] Deep Learning for LiDAR Point Clouds in Autonomous Driving: A Review. [__`Review.`__]
- [[arXiv](https://arxiv.org/abs/2006.04307)] Are We Hungry for 3D LiDAR Data for Semantic Segmentation? [__`Segmentation.`__]
- [[arXiv](https://arxiv.org/abs/2007.08488)] Complete & Label: A Domain Adaptation Approach to Semantic Segmentation of LiDAR Point Clouds. [__`Segmentation.`__]
- **[[arXiv](https://arxiv.org/abs/2007.08501)] Accelerating 3D Deep Learning with PyTorch3D. [`PyTorch3D.`]**
- [[arXiv](https://arxiv.org/abs/2008.01550)] Cylinder3D: An Effective 3D Framework for Driving-scene LiDAR Semantic Segmentation. [[code](https://github.com/xinge008/Cylinder3D)] [__`Segmentation.`__]
- **[[arXiv](https://arxiv.org/abs/2008.02986)] Global Context Aware Convolutions for 3D Point Cloud Understanding. [`Understanding.`]**
- [[arXiv](https://arxiv.org/abs/2008.03928)] Projected-point-based Segmentation: A New Paradigm for LiDAR Point Cloud Segmentation. [__`Segmentation.`__]
- [[arXiv](https://arxiv.org/pdf/2009.08924.pdf)] Multi-Resolution Graph Neural Network for Large-Scale Pointcloud Segmentation. [__`Segmentation`__]
- **[[arXiv](https://arxiv.org/pdf/2009.08920.pdf)] Deep Learning for 3D Point Cloud Understanding: A Survey. [[code]( https://github.com/SHI-Labs/3D-Point-Cloud-Learning)] [`Survey`]**
- [[arXiv](https://arxiv.org/pdf/2009.10569.pdf)] Improving Point Cloud Semantic Segmentation by Learning 3D Object Proposal Generation. [__`Segmentation`__]
- **[[arXiv](https://arxiv.org/pdf/2009.13727.pdf)] Graph-based methods for analyzing orchard tree structure using noisy point cloud data. [` `]**
- **[[arXiv](https://arxiv.org/pdf/2010.04642.pdf)] Torch-Points3D: A Modular Multi-Task Framework for Reproducible Deep Learning on 3D Point Clouds.[[torch]( https://github.com/nicolas-chaulet/torch-points3d)] [`Framework`]**
- [[arXiv](https://arxiv.org/pdf/2010.09582.pdf)] Learning to Reconstruct and Segment 3D Objects. [__` Segmentation；Reconstruction`__]
- [[arXiv](https://arxiv.org/pdf/2010.08744.pdf)] Generating Large Convex Polytopes Directly on Point Clouds. [__` Segmentation `__]
- [[arxiv]( https://arxiv.org/pdf/2010.08092.pdf)] Human Segmentation with Dynamic LiDAR Data. [__` Segmentation`__]
- [[arXiv](https://arxiv.org/abs/2011.00923)] MARNet: Multi-Abstraction Refinement Network for 3D Point Cloud Analysis. [[code](https://github.com/ruc98/MARNet)][__`Analysis`__]
- **[[arXiv](https://arxiv.org/abs/2011.00931)] Point Transformer. [`Analysis`]**
- **[[arXiv](https://arxiv.org/abs/2010.05501)] BiPointNet: Binary Neural Network for Point Clouds. [`Analysis`]**

- [[arXiv](https://arxiv.org/abs/2011.12745)] Deep Magnification-Arbitrary Upsampling over 3D Point Clouds. [__`Upsampling`__]
- [[arXiv](https://arxiv.org/abs/2011.13784)] Spherical Interpolated Convolutional Network with Distance-Feature Density for 3D Semantic Segmentation of Point Clouds.[__` Segmentation`__]
- [[arXiv](https://arxiv.org/abs/2011.13328)] DyCo3D: Robust Instance Segmentation of 3D Point Clouds through Dynamic Convolution.[[code](https://github.com/aim-uofa/DyCo3D)] [__`Segmentation `__]
- **[[arXiv](https://arxiv.org/pdf/2011.14285.pdf)] Deeper or Wider Networks of Point Clouds with Self-attention?[`Networks`]**
- [[arXiv](https://arxiv.org/abs/2012.04934)] AMVNet: Assertion-based Multi-View Fusion Network for LiDAR Semantic Segmentation.[__`Segmentation`__]
- [[arXiv](https://arxiv.org/abs/2012.05018)] vLPD-Net: A Registration-aided Domain Adaptation Network for 3D Point Cloud Based Place Recognition.[__`Registration`__]
- [[arXiv](https://arxiv.org/abs/2012.04439)] SPU-Net: Self-Supervised Point Cloud Upsampling by Coarse-to-Fine Reconstruction with Self-Projection Optimization.[__`Upsampling`__]

#### 2021

- [[arXiv](https://arxiv.org/abs/2011.13328)] DyCo3D: Robust Instance Segmentation of 3D Point Clouds through Dynamic Convolution. [__`Segmentation.`__]
- [[arXiv](https://arxiv.org/abs/2012.10217)] SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation. [__`Segmentation.`__]
- [[arXiv](https://arxiv.org/ftp/arxiv/papers/2012/2012.10192.pdf)] LGENet: Local and Global Encoder Network for Semantic Segmentation of Airborne Laser Scanning Point Clouds. [__`Segmentation.`__]
- [[arXiv](https://arxiv.org/abs/2012.09688)] PCT: Point Cloud Transformer. [__`Transformer.`__]
- [[arXiv](https://arxiv.org/abs/2012.09793)] SceneFormer: Indoor Scene Generation with Transformers. [__`Transformer.`__]
- [[arXiv](https://arxiv.org/abs/2101.02691)] Self-Supervised Pretraining of 3D Features on any Point-Cloud.[[pytorch](https://github.com/facebookresearch/DepthContrast)] [__`Self-Supervised.`__]
- [[arXiv](https://arxiv.org/abs/2102.10788)] Attention Models for Point Clouds in Deep Learning: A Survey. [__`Survey.`__]
