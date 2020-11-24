# 汇总3D点云语义分割论文




参考来源：https://github.com/Yochengliu/awesome-point-cloud-analysis

```diff
- Recent papers (from 2017)
```



![image-20201024220352989](https://cdn.jsdelivr.net/gh/lizhangjie316/img/2020/20201124110455.png)

> 1. [[3D点云语义分割综述](3D-point-cloud-review.md)] **from by** [胡庆永]([Deep Learning for 3D Point Clouds：A Survey_20200727版](https://github.com/lizhangjie316/3D-Point-Cloud-Semantic-Segement-Paper/blob/master/papers/Deep%20Learning%20for%203D%20Point%20Clouds%EF%BC%9AA%20Survey_20200727%E7%89%88.pdf))
> 2. [Yochengliu/awesome-point-cloud-analysis from 2017](https://github.com/Yochengliu/awesome-point-cloud-analysis)
> 3. [NUAAXQ/awesome-point-cloud-analysis-2020](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2020#2020)





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
- [[arXiv](https://arxiv.org/abs/1807.00652)] PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation. [[tensorflow](https://github.com/MVIG-SJTU/pointSIFT)] [__`seg.`__] :fire:
- [[arXiv](https://arxiv.org/abs/1811.11209)] Iterative Transformer Network for 3D Point Cloud. [__`cls.`__ __`seg.`__ __`pos.`__]
- [[arXiv](https://arxiv.org/abs/1812.11029)] Multi-column Point-CNN for Sketch Segmentation. [__`seg.`__]

---
- [[arXiv](https://arxiv.org/abs/1810.01151)] Know What Your Neighbors Do: 3D Semantic Segmentation of Point Clouds. [__`seg.`__]



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

### 2020 

- [[AAAI](https://arxiv.org/abs/1912.10775)] Point2Node: Correlation Learning of Dynamic-Node for Point Cloud Feature Modeling. [__`seg.`__ __`cls.`__]
- [[AAAI](https://arxiv.org/abs/1811.09361)] PRIN: Pointwise Rotation-Invariant Network. [__`seg.`__ __`cls.`__]
- [[CVPR](https://arxiv.org/pdf/1911.11236.pdf)] RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds. [[tensorflow](https://github.com/QingyongHu/RandLA-Net)] [__`seg.`__] 
- [[WACV](https://arxiv.org/pdf/1912.08487.pdf)] FuseSeg: LiDAR Point Cloud Segmentation Fusing Multi-Modal Data. [__`seg.`__ __`aut.`__]
- [[ECCV](https://arxiv.org/pdf/2007.10985.pdf)] PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding. [__`cls.`__ __`seg.`__ __`det.`__]



### Datasets

- [[KITTI](http://www.cvlibs.net/datasets/kitti/)] The KITTI Vision Benchmark Suite. [__`det.`__]
- [[ModelNet](http://modelnet.cs.princeton.edu/)] The Princeton ModelNet . [__`cls.`__]
- [[ShapeNet](https://www.shapenet.org/)]  A collaborative dataset between researchers at Princeton, Stanford and TTIC. [__`seg.`__]
- [[PartNet](https://shapenet.org/download/parts)] The PartNet dataset provides fine grained part annotation of objects in ShapeNetCore. [__`seg.`__]
- [[PartNet](http://kevinkaixu.net/projects/partnet.html)] PartNet benchmark from Nanjing University and National University of Defense Technology. [__`seg.`__]
- [[S3DIS](http://buildingparser.stanford.edu/dataset.html#Download)] The Stanford Large-Scale 3D Indoor Spaces Dataset. [__`seg.`__]
- [[ScanNet](http://www.scan-net.org/)] Richly-annotated 3D Reconstructions of Indoor Scenes. [__`cls.`__ __`seg.`__]
- [[Stanford 3D](https://graphics.stanford.edu/data/3Dscanrep/)] The Stanford 3D Scanning Repository. [__`reg.`__]
- [[UWA Dataset](http://staffhome.ecm.uwa.edu.au/~00053650/databases.html)] . [__`cls.`__ __`seg.`__ __`reg.`__]
- [[Princeton Shape Benchmark](http://shape.cs.princeton.edu/benchmark/)] The Princeton Shape Benchmark.
- [[SYDNEY URBAN OBJECTS DATASET](http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml)] This dataset contains a variety of common urban road objects scanned with a Velodyne HDL-64E LIDAR, collected in the CBD of Sydney, Australia. There are 631 individual scans of objects across classes of vehicles, pedestrians, signs and trees. [__`cls.`__ __`match.`__]
- [[ASL Datasets Repository(ETH)](https://projects.asl.ethz.ch/datasets/doku.php?id=home)] This site is dedicated to provide datasets for the Robotics community with the aim to facilitate result evaluations and comparisons. [__`cls.`__ __`match.`__ __`reg.`__ __`det`__]
- [[Large-Scale Point Cloud Classification Benchmark(ETH)](http://www.semantic3d.net/)] This benchmark closes the gap and provides a large labelled 3D point cloud data set of natural scenes with over 4 billion points in total. [__`cls.`__]
- [[Robotic 3D Scan Repository](http://asrl.utias.utoronto.ca/datasets/3dmap/)] The Canadian Planetary Emulation Terrain 3D Mapping Dataset is a collection of three-dimensional laser scans gathered at two unique planetary analogue rover test facilities in Canada.  
- [[Radish](http://radish.sourceforge.net/)] The Robotics Data Set Repository (Radish for short) provides a collection of standard robotics data sets.
- [[IQmulus & TerraMobilita Contest](http://data.ign.fr/benchmarks/UrbanAnalysis/#)] The database contains 3D MLS data from a dense urban environment in Paris (France), composed of 300 million points. The acquisition was made in January 2013. [__`cls.`__ __`seg.`__ __`det.`__]
- [[Oakland 3-D Point Cloud Dataset](http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/doc/)] This repository contains labeled 3-D point cloud laser data collected from a moving platform in a urban environment.
- [[Robotic 3D Scan Repository](http://kos.informatik.uni-osnabrueck.de/3Dscans/)] This repository provides 3D point clouds from robotic experiments，log files of robot runs and standard 3D data sets for the robotics community.
- [[Ford Campus Vision and Lidar Data Set](http://robots.engin.umich.edu/SoftwareData/Ford)] The dataset is collected by an autonomous ground vehicle testbed, based upon a modified Ford F-250 pickup truck. 
- [[The Stanford Track Collection](https://cs.stanford.edu/people/teichman/stc/)] This dataset contains about 14,000 labeled tracks of objects as observed in natural street scenes by a Velodyne HDL-64E S2 LIDAR.
- [[PASCAL3D+](http://cvgl.stanford.edu/projects/pascal3d.html)] Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild. [__`pos.`__ __`det.`__]
- [[3D MNIST](https://www.kaggle.com/daavoo/3d-mnist)] The aim of this dataset is to provide a simple way to get started with 3D computer vision problems such as 3D shape recognition. [__`cls.`__]
- [[WAD](http://wad.ai/2019/challenge.html)] [[ApolloScape](http://apolloscape.auto/tracking.html)] The datasets are provided by Baidu Inc. [__`tra.`__ __`seg.`__ __`det.`__]
- [[nuScenes](https://d3u7q4379vrm7e.cloudfront.net/object-detection)] The nuScenes dataset is a large-scale autonomous driving dataset.
- [[PreSIL](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/projects/precise-synthetic-image-and-lidar-presil-dataset-autonomous)] Depth information, semantic segmentation (images), point-wise segmentation (point clouds), ground point labels (point clouds), and detailed annotations for all vehicles and people. [[paper](https://arxiv.org/abs/1905.00160)] [__`det.`__ __`aut.`__]
- [[3D Match](http://3dmatch.cs.princeton.edu/)] Keypoint Matching Benchmark, Geometric Registration Benchmark, RGB-D Reconstruction Datasets. [__`reg.`__ __`rec.`__ __`oth.`__]
- [[BLVD](https://github.com/VCCIV/BLVD)] (a) 3D detection, (b) 4D tracking, (c) 5D interactive event recognition and (d) 5D intention prediction. [[ICRA 2019 paper](https://arxiv.org/abs/1903.06405v1)] [__`det.`__ __`tra.`__ __`aut.`__ __`oth.`__]
- [[PedX](https://arxiv.org/abs/1809.03605)] 3D Pose Estimation of Pedestrians, more than 5,000 pairs of high-resolution (12MP) stereo images and LiDAR data along with providing 2D and 3D labels of pedestrians. [[ICRA 2019 paper](https://arxiv.org/abs/1809.03605)] [__`pos.`__ __`aut.`__]
- [[H3D](https://usa.honda-ri.com/H3D)] Full-surround 3D multi-object detection and tracking dataset. [[ICRA 2019 paper](https://arxiv.org/abs/1903.01568)] [__`det.`__ __`tra.`__ __`aut.`__]
- [[Argoverse BY ARGO AI]](https://www.argoverse.org/) Two public datasets (3D Tracking and Motion Forecasting) supported by highly detailed maps to test, experiment, and teach self-driving vehicles how to understand the world around them.[[CVPR 2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.html)][__`tra.`__ __`aut.`__]
- [[Matterport3D](https://niessner.github.io/Matterport/)] RGB-D: 10,800 panoramic views from 194,400 RGB-D images. Annotations: surface reconstructions, camera poses, and 2D and 3D semantic segmentations. Keypoint matching, view overlap prediction, normal prediction from color, semantic segmentation, and scene classification. [[3DV 2017 paper](https://arxiv.org/abs/1709.06158)] [[code](https://github.com/niessner/Matterport)] [[blog](https://matterport.com/blog/2017/09/20/announcing-matterport3d-research-dataset/)]
- [[SynthCity](https://arxiv.org/abs/1907.04758)] SynthCity is a 367.9M point synthetic full colour Mobile Laser Scanning point cloud. Nine categories. [__`seg.`__ __`aut.`__]
- [[Lyft Level 5](https://level5.lyft.com/dataset/?source=post_page)] Include high quality, human-labelled 3D bounding boxes of traffic agents, an underlying HD spatial semantic map. [__`det.`__ __`seg.`__ __`aut.`__]
- [[SemanticKITTI](http://semantic-kitti.org)] Sequential Semantic Segmentation, 28 classes, for autonomous driving. All sequences of KITTI odometry labeled. [[ICCV 2019 paper](https://arxiv.org/abs/1904.01416)] [__`seg.`__ __`oth.`__ __`aut.`__]
- [[NPM3D](http://npm3d.fr/paris-lille-3d)] The Paris-Lille-3D  has been produced by a Mobile Laser System (MLS) in two different cities in France (Paris and Lille). [__`seg.`__] 
- [[The Waymo Open Dataset](https://waymo.com/open/)] The Waymo Open Dataset is comprised of high resolution sensor data collected by Waymo self-driving cars in a wide variety of conditions. [__`det.`__]
- [[A*3D: An Autonomous Driving Dataset in Challeging Environments](https://github.com/I2RDL2/ASTAR-3D)] A*3D: An Autonomous Driving Dataset in Challeging Environments. [__`det.`__]
- [[PointDA-10 Dataset](https://github.com/canqin001/PointDAN)] Domain Adaptation for point clouds.
- [[Oxford Robotcar](https://robotcar-dataset.robots.ox.ac.uk/)] The dataset captures many different combinations of weather, traffic and pedestrians. [__`cls.`__ __`det.`__ __`rec.`__]
- [[PandaSet](https://scale.com/open-datasets/pandaset)] Public large-scale dataset for autonomous driving provided by Hesai & Scale. It enables researchers to study challenging urban driving situations using the full sensor suit of a real self-driving-car. [__`det.`__ __`seg.`__]
- [[3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future)] [Alibaba] 3D-FRONT contains 10,000 houses (or apartments) and ~70,000 rooms with layout information. 3D-FUTURE contains 20,000+ clean and realistic synthetic scenes in 5,000+ diverse rooms which contain 10,000+ unique high quality 3D instances of furniture.



