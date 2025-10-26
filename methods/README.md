![](https://img.shields.io/badge/Abbreviation-blue) The abbreviation of the work.

![](https://img.shields.io/badge/Modality-green) The main explored modality of the work.

![](https://img.shields.io/badge/Pre_Training-orange) Pre-training method of the work.


主要根据涉及模态分类：
- 单一模态的预训练
  - 纯图像
  - 纯点云
- 涉及不同模态交互的预训练
  - 点云辅助图像预训练：增强图像感知几何的能力，如[ViDAR](https://github.com/OpenDriveLab/ViDAR) (可以以3D Object Detection任务为主列出一个表格作为benchmark)
  - 图像辅助点云预训练：增强点云感知语义的能力，如[SLidR](https://github.com/valeoai/SLidR) (可以以LiDAR Segmentation任务为主列出一个表格作为benchmark)
  - 点云图像联合的预训练，如[UniPAD](https://github.com/Nightmare-n/UniPAD)
  - 更多模态的预训练技术：引入文本信息，获得开放世界的感知能力，如[CLIP2Scene](https://github.com/runnanchen/CLIP2Scene)


## 介绍代理任务和下游任务

不同的代理任务，参考([Forge_VFM4AD](https://github.com/zhanghm1995/Forge_VFM4AD))，简要介绍：
- 基于对比学习（including with text，构建正负特征样本对）
- 基于知识蒸馏（使用现有的Image基础模型蒸馏给LiDAR为主）
- 基于重建（Masked Signal Modeling）
- 基于渲染（输出渲染回2D相机平面including RGB、Depth、Semantics）
- 基于预测未来帧（时序Forecasting，使用现有观测预测未来帧）

不同的下游任务：
- 感知类：3D的检测和分割、BEV地图构建、占用网络
- 预测规划类：运动预测（Forecasting）、motion planning、e2e driving

## 根据模态分类介绍方法

### 单模态

#### 纯图像
- 有监督预训练
2D (ImageNet-cls、Coco-Det)，3D (FCOS3D-3D Det),用于初始化图像backbone权重

- 无监督预训练
通用视觉领域：早期基于数据增强的方法([image permutation](https://arxiv.org/pdf/1603.09246), [rotation prediction](https://arxiv.org/pdf/1803.07728)), 对比学习（Moco系列等），MIM/MAE，DINO/DINOv2等
自动驾驶和机器人场景：

[RA-L 2023][INoD: Injected Noise Discriminator for Self-Supervised Representation Learning in Agricultural Fields](https://ieeexplore.ieee.org/document/10202201) ![](https://img.shields.io/badge/INoD-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Data_Aug-orange)

[RA-L 2024][Self-Supervised Representation Learning From Temporal Ordering of Automated Driving Sequences](https://ieeexplore.ieee.org/document/10400840) ![](https://img.shields.io/badge/TempO-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Data_Aug-orange)

[ECCV 2024][LetsMap: Unsupervised Representation Learning for Label-Efficient Semantic BEV Mapping](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07537.pdf) ![](https://img.shields.io/badge/LetsMap-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Construction-orange)

[ECCV 2024][NeRF-MAE: Masked AutoEncoders for Self-Supervised 3D Representation Learning for Neural Radiance Fields](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12306.pdf) ![](https://img.shields.io/badge/NeRF_MAE-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Construction-orange)

[arXiv 2024][VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving](https://arxiv.org/pdf/2411.14716) ![](https://img.shields.io/badge/VisionPAD-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Rendering-orange)


#### 纯点云

[ECCV 2020][PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding](https://github.com/facebookresearch/PointContrast) ![](https://img.shields.io/badge/PointContrast-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)

[CVPR 2021][Exploring Data-Efficient 3D Scene Understanding with Contrastive Scene Contexts](https://github.com/facebookresearch/ContrastiveSceneContexts) ![](https://img.shields.io/badge/ContrastiveSceneContexts-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)

[ICCV 2021][Self-Supervised Pretraining of 3D Features on Any Point-Cloud](https://github.com/facebookresearch/DepthContrast) ![](https://img.shields.io/badge/DepthContrast-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)

[RA-L 2021][SegContrast: 3D Point Cloud Feature Representation Learning through Self-supervised Segment Discrimination](https://github.com/PRBonn/segcontrast) ![](https://img.shields.io/badge/SegContrast-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)

[ECCV 2022][ProposalContrast: Unsupervised Pre-training for LiDAR-Based 3D Object Detection](https://github.com/yinjunbo/ProposalContrast) ![](https://img.shields.io/badge/ProposalContrast-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)

[T-IV 2023][Occupancy-MAE: Self-supervised Pre-training Large-scale LiDAR Point Clouds with Masked Occupancy Autoencoders](https://github.com/chaytonmin/Occupancy-MAE) ![](https://img.shields.io/badge/Occupancy_MAE-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Construction-orange)

[CVPR 2023][Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting](https://github.com/tarashakhurana/4d-occ-forecasting) ![](https://img.shields.io/badge/4D_Occ-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Forecasting-orange)

[CVPR 2023][ALSO: Automotive Lidar Self-supervision by Occupancy estimation](https://github.com/valeoai/ALSO) ![](https://img.shields.io/badge/ALSO-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Forecasting-orange)

[CVPR 2023][Generative Decoder for MAE Pre-training on LiDAR Point Clouds](https://github.com/Nightmare-n/GD-MAE) ![](https://img.shields.io/badge/GD_MAE-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange)

[NeurIPS 2023][AD-PT: Autonomous Driving Pre-Training with Large-scale Point Cloud Dataset](https://github.com/PJLab-ADG/3DTrans) ![](https://img.shields.io/badge/AD_PT-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)

[arXiv 2023][PatchContrast: Self-Supervised Pre-training for 3D Object Detection](https://arxiv.org/pdf/2308.06985) ![](https://img.shields.io/badge/PatchContrast-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)

[WACV 2024][MAELi: Masked Autoencoder for Large-Scale LiDAR Point Clouds](https://openaccess.thecvf.com/content/WACV2024/papers/Krispel_MAELi_Masked_Autoencoder_for_Large-Scale_LiDAR_Point_Clouds_WACV_2024_paper.pdf) ![](https://img.shields.io/badge/MAELi-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange)

[AAAI 2024][BEV-MAE: Bird's Eye View Masked Autoencoders for Point Cloud Pre-training in Autonomous Driving Scenarios](https://github.com/VDIGPKU/BEV-MAE) ![](https://img.shields.io/badge/MAELi-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange)

[CVPR 2024][UnO: Unsupervised Occupancy Fields for Perception and Forecasting](https://openaccess.thecvf.com/content/CVPR2024/papers/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.pdf) ![](https://img.shields.io/badge/UnO-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Forecasting-orange)

[3DV 2024][BEVContrast: Self-Supervision in BEV Space for Automotive Lidar Point Clouds](https://github.com/valeoai/BEVContrast) ![](https://img.shields.io/badge/BEVContrast-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)

[ICLR 2024][Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion](https://arxiv.org/abs/2311.01017) ![](https://img.shields.io/badge/Copilot4D-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Forecasting-orange)

[ECCV 2024][T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning](https://github.com/codename1995/t-mae) ![](https://img.shields.io/badge/T_MAE-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange)

[ECCV 2024][Equivariant Spatio-Temporal Self-Supervision for LiDAR Object Detection](https://arxiv.org/pdf/2404.11737) ![](https://img.shields.io/badge/E_SSL-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Scene_Flow-orange)

[ACM MM 2024][Point Cloud Reconstruction Is Insufficient to Learn 3D Representations](https://openreview.net/pdf?id=DM5eaZ6Ect) ![](https://img.shields.io/badge/PICTURE-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange)

[IV 2024][Rethinking Masked-Autoencoder-Based 3D Point Cloud Pretraining](https://ieeexplore.ieee.org/abstract/document/10588770) ![](https://img.shields.io/badge/LSV_MAE-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange)

[arXiv 2024][UNIT: Unsupervised Online Instance Segmentation through Time](https://csautier.github.io/unit/) ![](https://img.shields.io/badge/UINT-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Instance_Seg-orange)

[arXiv 2024][Sense Less, Generate More: Pre-training LiDAR Perception with Masked Autoencoders for Ultra-Efficient 3D Sensing](https://github.com/sinatayebati/R-MAE) ![](https://img.shields.io/badge/R_MAE-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange)


### 多模态交互

#### 图像辅助点云预训练
[arXiv 2022][Self-Supervised Pre-Training of 3D Point Cloud Networks with Image Data](https://arxiv.org/pdf/2211.11801) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)

[CVPR 2022][Image-to-Lidar Self-Supervised Distillation for Autonomous Driving Data](https://github.com/valeoai/SLidR) ![](https://img.shields.io/badge/SLidR-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[CVPR 2023][Self-Supervised Image-to-Point Distillation via Semantically Tolerant Contrastive Loss](https://openaccess.thecvf.com/content/CVPR2023/papers/Mahmoud_Self-Supervised_Image-to-Point_Distillation_via_Semantically_Tolerant_Contrastive_Loss_CVPR_2023_paper.pdf) ![](https://img.shields.io/badge/ST_SLidR-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[CVPR 2023][Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders](https://github.com/ZrrSkywalker/I2P_MAE) ![](https://img.shields.io/badge/I2P-MAE-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Reconstruction-orange)

[CVPR 2023][Unsupervised 3D Point Cloud Representation Learning by Triangle Constrained Contrast for Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2023/papers/Pang_Unsupervised_3D_Point_Cloud_Representation_Learning_by_Triangle_Constrained_Contrast_CVPR_2023_paper.pdf) ![](https://img.shields.io/badge/TriCC-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[NeurIPS 2023][Segment Any Point Cloud Sequences by Distilling Vision Foundation Models](https://github.com/youquanl/Segment-Any-Point-Cloud) ![](https://img.shields.io/badge/Seal-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[NeurIPS 2023][PRED: Pre-training via Semantic Rendering on LiDAR Point Clouds](https://arxiv.org/abs/2311.04501) ![](https://img.shields.io/badge/PRED-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Rendering-orange)

[arXiv 2023][360° from a Single Camera: A Few-Shot Approach for LiDAR Segmentation](https://arxiv.org/pdf/2309.06197) ![](https://img.shields.io/badge/ImageTo360-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[CVPR 2024][Three Pillars improving Vision Foundation Model Distillation for Lidar](https://github.com/ZrrSkywalker/ScaLR) ![](https://img.shields.io/badge/ScaLR-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[CVPR 2024][Building a Strong Pre-Training Baseline for Universal 3D Large-Scale Perception](https://github.com/chenhaomingbob/CSC) ![](https://img.shields.io/badge/CSC-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[ICLR 2024][Pre-Training LiDAR-Based 3D Object Detectors Through Colorization](https://arxiv.org/abs/2310.14592) ![](https://img.shields.io/badge/GPC-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Colorization-orange)

[IROS 2024][Cross-Modal Self-Supervised Learning with Effective Contrastive Units for LiDAR Point Clouds](https://github.com/qcraftai/cross-modal-ssl) ![](https://img.shields.io/badge/cross_modal_ssl-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)

[ECCV 2024][4D Contrastive Superflows are Dense 3D Representation Learners](https://github.com/Xiangxu-0103/SuperFlow) ![](https://img.shields.io/badge/SuperFlow-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)

[ECCV 2024][Image-to-Lidar Relational Distillation for Autonomous Driving Data](https://arxiv.org/pdf/2409.00845) ![](https://img.shields.io/badge/Rel-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[IJCV 2024][HVDistill: Transferring Knowledge from Images to Point Clouds via Unsupervised Hybrid-View Distillation](https://link.springer.com/article/10.1007/s11263-023-01981-w) ![](https://img.shields.io/badge/HVDistill-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[DCOSS-IoT 2024][Self-Supervised Contrastive Learning for Camera-to-Radar Knowledge Distillation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10621525) ![](https://img.shields.io/badge/RadarContrast-blue) ![](https://img.shields.io/badge/Radar-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[arXiv 2024][Shelf-Supervised Cross-Modal Pre-Training for 3D Object Detection](https://arxiv.org/pdf/2406.10115) ![](https://img.shields.io/badge/CM3D-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[arXiv 2024][Fine-grained Image-to-LiDAR Contrastive Distillation with Visual Foundation Models](https://github.com/Eaphan/OLIVINE) ![](https://img.shields.io/badge/OLIVINE-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[arXiv 2024][Exploring the Untouched Sweeps for Conflict-Aware 3D Segmentation Pretraining](https://arxiv.org/pdf/2407.07465) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Contrastive_Learning-orange)


#### 点云辅助图像预训练

[ICCV 2021][Is Pseudo-Lidar needed for Monocular 3D Object detection?](https://github.com/TRI-ML/dd3d) ![](https://img.shields.io/badge/DD3D-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Depth_Distillation-orange)

[arXiv 2022][Delving into the Pre-training Paradigm of Monocular 3D Object Detection](https://arxiv.org/pdf/2206.03657) ![](https://img.shields.io/badge/DEPT-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Depth_Distillation-orange)

[ICCV 2023][Scene as Occupancy](https://github.com/OpenDriveLab/OccNet) ![](https://img.shields.io/badge/OpenOcc-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Forecasting-orange)

[ICCV 2023][GeoMIM: Towards Better 3D Knowledge Transfer via Masked Image Modeling for Multi-view 3D Understanding](https://github.com/Sense-X/GeoMIM) ![](https://img.shields.io/badge/GeoMIM-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange)

[arXiv 2023][Geometric-aware Pretraining for Vision-centric 3D Object Detection](https://github.com/OpenDriveLab/Birds-eye-view-Perception) ![](https://img.shields.io/badge/GAPretrain-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Distillation-orange)

[RA-L 2024][Multi-Camera Unified Pre-training via 3D Scene Reconstruction](https://github.com/chaytonmin/UniScene) ![](https://img.shields.io/badge/UniScene-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Forecasting-orange)

[CVPR 2024][SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction](https://github.com/huang-yh/SelfOcc) ![](https://img.shields.io/badge/SelfOcc-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Rendering-orange)

[CVPR 2024][Visual Point Cloud Forecasting enables Scalable Autonomous Driving](https://github.com/OpenDriveLab/ViDAR) ![](https://img.shields.io/badge/ViDAR-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Forecasting-orange)

[CVPR 2024][DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving](https://arxiv.org/abs/2405.04390) ![](https://img.shields.io/badge/DriveWorld-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Forecasting-orange)

[CVPRW 2024][OccFeat: Self-supervised Occupancy Feature Prediction for Pretraining BEV Segmentation Networks](https://openaccess.thecvf.com/content/CVPR2024W/WAD/papers/Sirko-Galouchenko_OccFeat_Self-supervised_Occupancy_Feature_Prediction_for_Pretraining_BEV_Segmentation_Networks_CVPRW_2024_paper.pdf) ![](https://img.shields.io/badge/OccFeat-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Distillation-orange)

[ECCV 2024][OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](https://github.com/wzzheng/OccWorld) ![](https://img.shields.io/badge/OccWorld-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Occ-green) ![](https://img.shields.io/badge/Forecasting-orange)

[IJCNN 2024][Focus on your Geometry: Exploiting the Potential of Multi-Frame Stereo Depth Estimation Pre-training for 3D Object Detection](https://ieeexplore.ieee.org/abstract/document/10650924) ![](https://img.shields.io/badge/MVS3D-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Depth_Distillation-orange)

[arXiv 2024][OccSora: 4D Occupancy Generation Models as World Simulators for Autonomous Driving](https://github.com/wzzheng/OccSora) ![](https://img.shields.io/badge/OccSora-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Occ-green) ![](https://img.shields.io/badge/Forecasting-orange)

[arXiv 2024][MIM4D: Masked Modeling with Multi-View Video for Autonomous Driving Representation Learning](https://github.com/hustvl/MIM4D) ![](https://img.shields.io/badge/MIM4D-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange) ![](https://img.shields.io/badge/Rendering-orange)

[arXiv 2024][GaussianPretrain: A Simple Unified 3D Gaussian Representation for Visual Pre-training in Autonomous Driving](https://arxiv.org/pdf/2411.12452) ![](https://img.shields.io/badge/GaussianPretrain-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange) ![](https://img.shields.io/badge/Rendering-orange)


#### 点云图像联合预训练

[arXiv 2023][PonderV2: Pave the Way for 3D Foundationn Model with A Universal Pre-training Paradigm](https://github.com/OpenGVLab/PonderV2) ![](https://img.shields.io/badge/PonderV2-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Rendering-orange)

[CVPR 2024][UniPAD: A niversal Pre-training Paradigm for Autonomous Driving](https://github.com/Nightmare-n/UniPAD) ![](https://img.shields.io/badge/UniPAD-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Rendering-orange) ![](https://img.shields.io/badge/Reconstruction-orange)

[ECCV 2024][UniM2AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving](https://github.com/hollow-503/UniM2AE) ![](https://img.shields.io/badge/UniM2AE-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange)

[ECCV 2024][ConDense: Consistent 2D/3D Pre-training for Dense and Sparse Features from Multi-View Images](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07038.pdf) ![](https://img.shields.io/badge/ConDense-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Distillation-orange)

[arXiv 2024][Learning Shared RGB-D Fields: Unified Self-supervised Pre-training for Label-efficient LiDAR-Camera 3D Perception](https://github.com/Xiaohao-Xu/Unified-Pretrain-AD/) ![](https://img.shields.io/badge/Unified_Pretrain_AD-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Reconstruction-orange) ![](https://img.shields.io/badge/Rendering-orange)

[arXiv 2024][BEVWorld: A Multimodal World Model for Autonomous Driving via Unified BEV Latent Space](https://github.com/zympsyche/BevWorld) ![](https://img.shields.io/badge/BEVWorld-blue) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Forecasting-orange) ![](https://img.shields.io/badge/Rendering-orange)

#### 引入更多模态

[CVPR 2023][CLIP2Scene: Towards Label-efficient 3D Scene Understanding by CLIP](https://github.com/runnanchen/CLIP2Scene) ![](https://img.shields.io/badge/CLIP2Scene-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Text-green) ![](https://img.shields.io/badge/Distillation-orange)

[CVPR 2023][OpenScene: 3D Scene Understanding with Open Vocabularies](https://openaccess.thecvf.com/content/CVPR2023/papers/Peng_OpenScene_3D_Scene_Understanding_With_Open_Vocabularies_CVPR_2023_paper.pdf) ![](https://img.shields.io/badge/OpenScene-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Text-green) ![](https://img.shields.io/badge/Distillation-orange)

[ACM MM 2023][Transferring CLIP’s Knowledge into Zero-Shot Point Cloud Semantic Segmentation](https://arxiv.org/pdf/2312.07221) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Text-green) ![](https://img.shields.io/badge/Distillation-orange)

[ICCVW 2023][CLIP-FO3D: Learning Free Open-world 3D Scene Representations from 2D Dense CLIP](https://openaccess.thecvf.com/content/ICCV2023W/OpenSUN3D/papers/Zhang_CLIP-FO3D_Learning_Free_Open-World_3D_Scene_Representations_from_2D_Dense_ICCVW_2023_paper.pdf) ![](https://img.shields.io/badge/CLIP_FO3D-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Text-green) ![](https://img.shields.io/badge/Distillation-orange)

[NeurIPS 2023][POP-3D: Open-Vocabulary 3D Occupancy Prediction from Images](https://vobecant.github.io/POP3D/) ![](https://img.shields.io/badge/POP3D-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Text-green) ![](https://img.shields.io/badge/Distillation-orange)

[ITSC 2023][Road Condition Anomaly Detection using Self-Supervised Learning from Audio](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10421899) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Audio-green) ![](https://img.shields.io/badge/Reconstruction-orange)

[AAAI 2024][VLM2Scene: Self-Supervised Image-Text-LiDAR Learning with Foundation Models for Autonomous Driving Scene Understanding](https://ojs.aaai.org/index.php/AAAI/article/view/28121) ![](https://img.shields.io/badge/VLM2Scene-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Text-green) ![](https://img.shields.io/badge/Distillation-orange)

[CVPR 2024][Hierarchical Intra-modal Correlation Learning for Label-free 3D Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2024/papers/Kang_Hierarchical_Intra-modal_Correlation_Learning_for_Label-free_3D_Semantic_Segmentation_CVPR_2024_paper.pdf) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Text-green) ![](https://img.shields.io/badge/Distillation-orange)

[ECCV 2024][Better Call SAL: Towards Learning to Segment Anything in Lidar](https://github.com/nv-dvl/segment-anything-lidar) ![](https://img.shields.io/badge/SAM_LiDAR-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Text-green) ![](https://img.shields.io/badge/Distillation-orange)

[ACM MM 2024][Affinity3D: Propagating Instance-Level Semantic Affinity for Zero-Shot Point Cloud Semantic Segmentation](https://openreview.net/pdf?id=5ielGTd21u) ![](https://img.shields.io/badge/Affinity3D-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Text-green) ![](https://img.shields.io/badge/Distillation-orange)

[arXiv 2024][3D Unsupervised Learning by Distilling 2D Open-Vocabulary Segmentation Models for Autonomous Driving](https://github.com/sbysbysbys/UOV) ![](https://img.shields.io/badge/UOV-blue) ![](https://img.shields.io/badge/LiDAR-green) ![](https://img.shields.io/badge/Camera-green) ![](https://img.shields.io/badge/Text-green) ![](https://img.shields.io/badge/Distillation-orange)

