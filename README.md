<div align=center>

# Forging Spatial Intelligence
### A Survey on Multi-Modal Pre-Training for Autonomous Systems

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![arXiv](https://img.shields.io/badge/arXiv-25xx.xxxxx-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/25xx.xxxxx)
![Visitors](https://komarev.com/ghpvc/?username=worldbench&repo=awesome-spatial-intelligence&label=Visitors&color=yellow&style=social)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-red.svg?style=flat)](https://github.com/worldbench/awesome-spatial-intelligence/pulls)

| <img width="100%" src="docs/figures/teaser.png" alt="Taxonomy of Spatial Intelligence"> |
|:-:|
| *Figure 1: Taxonomy of Multi-Modal Representation Learning for Spatial Intelligence.* |

</div>

This repository serves as the official resource collection for the survey paper **"Forging Spatial Intelligence: A Survey on Multi-Modal Pre-Training for Autonomous Systems"**.

In this work, we establish a systematic taxonomy for the field, unifying terminology, scope, and evaluation benchmarks. We organize existing methodologies into three complementary paradigms based on information flow and abstraction level:

* 📷 **Single-Modality Pre-Training** *The Bedrock of Perception.* Focuses on extracting foundational features from individual sensor streams (Camera or LiDAR) via self-supervised learning techniques, such as Contrastive Learning, Masked Modeling, and Forecasting. This paradigm establishes the fundamental representations for sensor-specific tasks.
* 🔄 **Multi-Modality Pre-Training** *Bridging the Semantic-Geometric Gap.* Leverages cross-modal synergy to fuse heterogeneous sensor data. This category includes **LiDAR-Centric** (distilling visual semantics into geometry), **Camera-Centric** (injecting geometric priors into vision), and **Unified** frameworks that jointly learn modality-agnostic representations.
* 🌍 **Spatial Intelligence & World Models** *The Frontier of Embodied Autonomy.* Represents the evolution from passive perception to active decision-making. This paradigm encompasses **Generative World Models** (e.g., video/occupancy generation), **Embodied Vision-Language-Action (VLA)** models, and systems capable of **Open-World** reasoning.

📄 **Paper:** [arXiv:25xx.xxxxx](https://arxiv.org/abs/25xx.xxxxx)


---

### Citation

If you find this work helpful for your research, please kindly consider citing our paper:

```bibtex
@article{wang2025forging,
    title={Forging Spatial Intelligence: A Survey on Multi-Modal Pre-Training for Autonomous Systems},
    author={Song Wang and Lingdong Kong and Xiaolu Liu and Hao Shi and Wentong Li and Jianke Zhu and Steven C. H. Hoi},
    journal={arXiv preprint arXiv:25xx.xxxxx},
    year={2025}
}
````

-----

### Table of Contents

  - [**1. Benchmarks & Datasets**](#1-benchmarks--datasets)
      - [Vehicle-Based Datasets](#vehicle-based-datasets)
      - [Drone-Based Datasets](#drone-based-datasets)
      - [Other Datasets](#other-datasets)
  - [**2. Single-Modality Pre-Training**](#2-single-modality-pre-training)
      - [LiDAR-Only](#lidar-only)
      - [Camera-Only](#camera-only)
  - [**3. Multi-Modality Pre-Training**](#3-multi-modality-pre-training)
      - [LiDAR-Centric (Vision-to-LiDAR)](#lidar-centric-pre-training)
      - [Camera-Centric (LiDAR-to-Vision)](#camera-centric-pre-training)
      - [Unified Frameworks](#unified-pre-training)
  - [**4. Spatial Intelligence & World Models**](#4-spatial-intelligence--world-models)
      - [Generative World Models](#generative-world-models)
      - [Vision-Language-Action (VLA)](#vision-language-action-vla)
      - [Open-World Perception](#open-world-perception)
  - [**5. Acknowledgements**](#5-acknowledgements)

-----

# 1. Benchmarks & Datasets

### Vehicle-Based Datasets


| Dataset | Venue | Sensor | Task | Download |
| :-: | :-: | :-: | :-: | :-: |
| `KITTI` | [CVPR'12](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf) | 2 Cam(RGB), 2 Cam(Gray), 1 LiDAR(64) | 3D Det, Stereo, Optical Flow, SLAM | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.cvlibs.net/datasets/kitti/index.php) |
| `Argoverse` | [CVPR'19](https://arxiv.org/pdf/1911.02620) | 7 Cam(RGB), 2 LiDAR(32) | 3D Tracking, Forecasting, Map | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.argoverse.org/) |
| `nuScenes` | [CVPR'20](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf) | 6 Cam(RGB), 1 LiDAR(32), 5 Radar | 3D Det, Seg, Occ, Map | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.nuscenes.org/nuscenes#download) |
| `Waymo` | [CVPR'20](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf) | 5 Cam(RGB), 5 LiDAR | Perception (Det, Seg, Track), Motion | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://waymo.com/open/licensing/?continue=%2Fopen%2Fdownload%2F) |
| `Lyft L5` | [CoRL'20](https://arxiv.org/pdf/2006.14480) | 7 Cam(RGB), 3 LiDAR, 5 Radar | 3D Det, Motion Forecasting/Planning | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://self-driving.lyft.com/level5/data) |
| `ONCE` | [NeurIPS'21](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/67c6a1e7ce56d3d6fa748ab6d9af3fd7-Paper-round1.pdf) | 7 Cam(RGB), 1 LiDAR(40) | 3D Det (Self-supervised/Semi-supervised) | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://once-for-auto-driving.github.io/download.html) |
| `PandaSet` | [ITSC'21](https://ieeexplore.ieee.org/abstract/document/9565009) | 6 Cam(RGB), 2 LiDAR | 3D Det, LiDAR Seg | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://scale.com/open-av-datasets/pandaset) |

### Drone-Based Datasets

| Dataset | Venue | Sensor | Task | Download |
| :-: | :-: | :-: | :-: | :-: |
| `M3ED` | CVPRW'23 | Cam (RGB/Gray), LiDAR, Event | 2D/3D Seg, Depth, Optical Flow | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://m3ed.io/download/) |
| `CDrone` | GCPR'24 | Camera (Carla) | Monocular 3D Det | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://cvg.cit.tum.de/webshare/g/cdrone/data/) |
| `VisDrone` | 2019 | Aerial Camera | Detection, Tracking | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/VisDrone/VisDrone-Dataset) |
| `UAVid` | 2020 | Slanted Camera | Semantic Segmentation | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://uavid.nl) |
| `BioDrone` | 2024 | Bionic Camera | Tracking | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](http://biodrone.aitestunion.com/) |

### Other Datasets

| Dataset | Platform | Sensors | Website |
|:-:|:-:|:-|:-:|
| `RailSem19` | Railway | Camera | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.wilddash.cc/railsem19) |
| `WaterScenes` | USV (Water) | Camera, Radar | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/WaterScenes/WaterScenes) |
| `Han et al.` | Legged Robot | Cam, LiDAR | - |

-----

# 2. Single-Modality Pre-Training

### LiDAR-Only

> *Methods utilizing Point Cloud Contrastive Learning, Masked Autoencoders (MAE), or Forecasting.*

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `PointContrast` | [![arXiv](https://img.shields.io/badge/arXiv-2007.10985-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2007.10985)<br>Unsupervised Pre-training for 3D Point Cloud Understanding | ECCV 2020 | [![GitHub](https://img.shields.io/github/stars/facebookresearch/PointContrast)](https://github.com/facebookresearch/PointContrast) |
| `DepthContrast` | [![arXiv](https://img.shields.io/badge/arXiv-2101.02691-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2101.02691)<br>Self-supervised Pretraining of 3D Features on any Point-Cloud | ICCV 2021 | [![GitHub](https://img.shields.io/github/stars/facebookresearch/DepthContrast)](https://github.com/facebookresearch/DepthContrast) |
| `SegContrast` | 3D Point Cloud Feature Representation Learning through Self-supervised Segment Discrimination | RA-L 2021 | [](https://github.com/PRBonn/segcontrast) |
| `ProposalContrast` | Unsupervised Pre-training for LiDAR-Based 3D Object Detection | ECCV 2022 | [](https://github.com/yinjunbo/ProposalContrast) |
| `Occupancy-MAE` | Self-supervised Pre-training Large-scale LiDAR Point Clouds with Masked Occupancy Autoencoders | T-IV 2023 | [](https://github.com/chaytonmin/Occupancy-MAE) |
| `ALSO` | Automotive LiDAR Self-supervision by Occupancy Estimation | CVPR 2023 | [](https://github.com/valeoai/ALSO) |
| `GD-MAE` | Generative Decoder for MAE Pre-training on LiDAR Point Clouds | CVPR 2023 | [](https://www.google.com/search?q=https://github.com/OpenDriveLab/GD-MAE) |
| `AD-PT` | Autonomous Driving Pre-Training with Large-scale Point Cloud Dataset | NeurIPS 2023 | [](https://github.com/PJLab-ADG/3DTrans) |
| `PatchContrast` | Self-Supervised Pre-training for 3D Object Detection | arXiv 2023 | [](https://www.google.com/search?q=) |
| `MAELi` | Masked Autoencoder for Large-Scale LiDAR Point Clouds | WACV 2024 | [](https://www.google.com/search?q=) |
| `BEV-MAE` | Bird's Eye View Masked Autoencoders for Point Cloud Pre-training | AAAI 2024 | [](https://github.com/VDIGPKU/BEV-MAE) |
| `UnO` | Unsupervised Occupancy Fields for Perception and Forecasting | CVPR 2024 | [](https://www.google.com/search?q=) |
| `BEVContrast` | Self-Supervision in BEV Space for Automotive Lidar Point Clouds | 3DV 2024 | [](https://github.com/valeoai/BEVContrast) |
| `Copilot4D` | Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion | ICLR 2024 | [](https://www.google.com/search?q=) |
| `T-MAE` | Temporal Masked Autoencoders for Point Cloud Representation Learning | ECCV 2024 | [](https://github.com/codename1995/t-mae) |
| `PICTURE` | Point Cloud Reconstruction Is Insufficient to Learn 3D Representations | ACM MM 2024 | [](https://www.google.com/search?q=) |
| `LSV-MAE` | Rethinking Masked-Autoencoder-Based 3D Point Cloud Pretraining | IV 2024 | [](https://www.google.com/search?q=) |
| `UNIT` | Unsupervised Online Instance Segmentation through Time | arXiv 2024 | [](https://csautier.github.io/unit/) |
| `R-MAE` | Sense Less, Generate More: Pre-training LiDAR Perception with Masked Autoencoders | arXiv 2024 | [](https://github.com/sinatayebati/R-MAE) |

### Camera-Only

> *Self-supervised learning from image sequences for driving/robotics.*

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `INoD` | Injected Noise Discriminator for Self-Supervised Representation | RA-L 2023 | [](https://www.google.com/search?q=) |
| `TempO` | Self-Supervised Representation Learning From Temporal Ordering | RA-L 2024 | [](https://www.google.com/search?q=) |
| `LetsMap` | Unsupervised Representation Learning for Label-Efficient Semantic BEV Mapping | ECCV 2024 | [](https://www.google.com/search?q=) |
| `NeRF-MAE` | Masked AutoEncoders for Self-Supervised 3D Representation Learning | ECCV 2024 | [](https://www.google.com/search?q=) |
| `VisionPAD` | A Vision-Centric Pre-training Paradigm for Autonomous Driving | arXiv 2024 | [](https://www.google.com/search?q=) |

-----

# 3. Multi-Modality Pre-Training

### LiDAR-Centric Pre-Training

> *Enhancing LiDAR representations using Vision foundation models (Knowledge Distillation).*

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `SLidR` | Image-to-Lidar Self-Supervised Distillation | CVPR 2022 | [](https://github.com/valeoai/SLidR) |
| `ST-SLidR` | Self-Supervised Image-to-Point Distillation via Semantically Tolerant Contrastive Loss | CVPR 2023 | [](https://www.google.com/search?q=) |
| `I2P-MAE` | Learning 3D Representations from 2D Pre-trained Models via Image-to-Point MAE | CVPR 2023 | [](https://github.com/ZrrSkywalker/I2P_MAE) |
| `TriCC` | Unsupervised 3D Point Cloud Representation Learning by Triangle Constrained Contrast | CVPR 2023 | [](https://www.google.com/search?q=) |
| `Seal` | Segment Any Point Cloud Sequences by Distilling Vision FMs | NeurIPS 23 | [](https://github.com/youquanl/Segment-Any-Point-Cloud) |
| `PRED` | Pre-training via Semantic Rendering on LiDAR Point Clouds | NeurIPS 23 | [](https://www.google.com/search?q=) |
| `ImageTo360` | 360° from a Single Camera: A Few-Shot Approach for LiDAR Segmentation | arXiv 2023 | [](https://www.google.com/search?q=) |
| `ScaLR` | Three Pillars improving Vision Foundation Model Distillation for Lidar | CVPR 2024 | [](https://github.com/ZrrSkywalker/ScaLR) |
| `CSC` | Building a Strong Pre-Training Baseline for Universal 3D Large-Scale Perception | CVPR 2024 | [](https://github.com/chenhaomingbob/CSC) |
| `GPC` | Pre-Training LiDAR-Based 3D Object Detectors Through Colorization | ICLR 2024 | [](https://www.google.com/search?q=) |
| `Cross-Modal SSL` | Cross-Modal Self-Supervised Learning with Effective Contrastive Units | IROS 2024 | [](https://github.com/qcraftai/cross-modal-ssl) |
| `SuperFlow` | 4D Contrastive Superflows are Dense 3D Representation Learners | ECCV 2024 | [](https://github.com/Xiangxu-0103/SuperFlow) |
| `Rel` | Image-to-Lidar Relational Distillation for Autonomous Driving Data | ECCV 2024 | [](https://www.google.com/search?q=) |
| `HVDistill` | Transferring Knowledge from Images to Point Clouds via Unsupervised Hybrid-View Distillation | IJCV 2024 | [](https://www.google.com/search?q=) |
| `RadarContrast` | Self-Supervised Contrastive Learning for Camera-to-Radar Knowledge Distillation | DCOSS-IoT 2024 | [](https://www.google.com/search?q=) |
| `CM3D` | Shelf-Supervised Cross-Modal Pre-Training for 3D Object Detection | arXiv 2024 | [](https://www.google.com/search?q=) |
| `OLIVINE` | Fine-grained Image-to-LiDAR Contrastive Distillation with Visual Foundation Models | arXiv 2024 | [](https://github.com/Eaphan/OLIVINE) |

### Camera-Centric Pre-Training

> *Learning 3D Geometry from Camera inputs using LiDAR supervision.*

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `DD3D` | Is Pseudo-Lidar needed for Monocular 3D Object detection? | ICCV 2021 | [](https://github.com/TRI-ML/dd3d) |
| `DEPT` | Delving into the Pre-training Paradigm of Monocular 3D Object Detection | arXiv 2022 | [](https://www.google.com/search?q=) |
| `OccNet` | Scene as Occupancy | ICCV 2023 | [](https://github.com/OpenDriveLab/OccNet) |
| `GeoMIM` | Towards Better 3D Knowledge Transfer via Masked Image Modeling | ICCV 2023 | [](https://github.com/Sense-X/GeoMIM) |
| `GAPretrain` | Geometric-aware Pretraining for Vision-centric 3D Object Detection | arXiv 2023 | [](https://github.com/OpenDriveLab/Birds-eye-view-Perception) |
| `UniScene` | Multi-Camera Unified Pre-training via 3D Scene Reconstruction | RA-L 2024 | [](https://github.com/chaytonmin/UniScene) |
| `SelfOcc` | Self-Supervised Vision-Based 3D Occupancy Prediction | CVPR 2024 | [](https://github.com/huang-yh/SelfOcc) |
| `ViDAR` | Visual Point Cloud Forecasting enables Scalable Autonomous Driving | CVPR 2024 | [](https://github.com/OpenDriveLab/ViDAR) |
| `DriveWorld` | 4D Pre-trained Scene Understanding via World Models | CVPR 2024 | [](https://www.google.com/search?q=) |
| `OccFeat` | Self-supervised Occupancy Feature Prediction for Pretraining BEV Segmentation | CVPRW 2024 | [](https://www.google.com/search?q=) |
| `OccWorld` | Learning a 3D Occupancy World Model for Autonomous Driving | ECCV 2024 | [](https://github.com/wzzheng/OccWorld) |
| `MVS3D` | Exploiting the Potential of Multi-Frame Stereo Depth Estimation Pre-training | IJCNN 2024 | [](https://www.google.com/search?q=) |
| `OccSora` | 4D Occupancy Generation Models as World Simulators | arXiv 2024 | [](https://github.com/wzzheng/OccSora) |
| `MIM4D` | Masked Modeling with Multi-View Video for Autonomous Driving | arXiv 2024 | [](https://github.com/hustvl/MIM4D) |
| `GaussianPretrain` | A Simple Unified 3D Gaussian Representation for Visual Pre-training | arXiv 2024 | [](https://www.google.com/search?q=) |

### Unified Pre-Training

> *Joint optimization of multi-modal encoders for unified representations.*

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `PonderV2` | Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm | arXiv 2023 | [](https://github.com/OpenGVLab/PonderV2) |
| `UniPAD` | A Universal Pre-training Paradigm for Autonomous Driving | CVPR 2024 | [](https://github.com/Nightmare-n/UniPAD) |
| `UniM2AE` | Multi-Modal Masked Autoencoders with Unified 3D Representation | ECCV 2024 | [](https://github.com/hollow-503/UniM2AE) |
| `ConDense` | Consistent 2D/3D Pre-training for Dense and Sparse Features | ECCV 2024 | [](https://www.google.com/search?q=) |
| `Unified Pretrain` | Learning Shared RGB-D Fields: Unified Self-supervised Pre-training | arXiv 2024 | [](https://github.com/Xiaohao-Xu/Unified-Pretrain-AD/) |
| `BEVWorld` | A Multimodal World Simulator via Unified BEV Latent Space | arXiv 2024 | [](https://github.com/zympsyche/BevWorld) |

-----

# 4. Spatial Intelligence & World Models

### Open-World Perception

> *Text-Grounded and Open-Vocabulary Understanding.*

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `CLIP2Scene` | Towards Label-efficient 3D Scene Understanding by CLIP | CVPR 2023 | [](https://github.com/runnanchen/CLIP2Scene) |
| `OpenScene` | 3D Scene Understanding with Open Vocabularies | CVPR 2023 | [](https://www.google.com/search?q=) |
| `CLIP-FO3D` | Learning Free Open-world 3D Scene Representations from 2D Dense CLIP | ICCVW 2023 | [](https://www.google.com/search?q=) |
| `POP-3D` | Open-Vocabulary 3D Occupancy Prediction from Images | NeurIPS 2023 | [](https://vobecant.github.io/POP3D/) |
| `VLM2Scene` | Self-Supervised Image-Text-LiDAR Learning with Foundation Models | AAAI 2024 | [](https://www.google.com/search?q=) |
| `SAL` | Better Call SAL: Towards Learning to Segment Anything in Lidar | ECCV 2024 | [](https://github.com/nv-dvl/segment-anything-lidar) |
| `Affinity3D` | Propagating Instance-Level Semantic Affinity for Zero-Shot Semantic Seg | ACM MM 2024 | [](https://www.google.com/search?q=) |
| `UOV` | 3D Unsupervised Learning by Distilling 2D Open-Vocabulary Segmentation Models | arXiv 2024 | [](https://github.com/sbysbysbys/UOV) |

-----

# 5. Acknowledgements

We thank the authors of the referenced papers for their open-source contributions. 
