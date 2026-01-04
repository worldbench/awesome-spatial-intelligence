<div align=center>

# Forging Spatial Intelligence
### A Roadmap of Multi-Modal Data Pre-Training for Autonomous Systems

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
![Visitors](https://komarev.com/ghpvc/?username=worldbench&repo=awesome-spatial-intelligence&label=Visitors&color=yellow&style=social)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-red.svg?style=flat)](https://github.com/worldbench/awesome-spatial-intelligence/pulls)

| <img width="100%" src="docs/figures/teaser.png" alt="Taxonomy of Spatial Intelligence"> |
|:-:|
| *Figure 1: Taxonomy of Multi-Modal Representation Learning for Spatial Intelligence.* |

</div>

This repository serves as the official resource collection for the paper **"Forging Spatial Intelligence: A Roadmap of Multi-Modal Data Pre-Training for Autonomous Systems"**.

In this work, we establish a systematic taxonomy for the field, unifying terminology, scope, and evaluation benchmarks. We organize existing methodologies into three complementary paradigms based on information flow and abstraction level:

* 📷 **Single-Modality Pre-Training**<br>*The Bedrock of Perception.* Focuses on extracting foundational features from individual sensor streams (Camera or LiDAR) via self-supervised learning techniques, such as Contrastive Learning, Masked Modeling, and Forecasting. This paradigm establishes the fundamental representations for sensor-specific tasks.
* 🔄 **Multi-Modality Pre-Training**<br>*Bridging the Semantic-Geometric Gap.* Leverages cross-modal synergy to fuse heterogeneous sensor data. This category includes **LiDAR-Centric** (distilling visual semantics into geometry), **Camera-Centric** (injecting geometric priors into vision), and **Unified** frameworks that jointly learn modality-agnostic representations.
* 🌍 **Open-World Perception and Planning**<br>*The Frontier of Embodied Autonomy.* Represents the evolution from passive perception to active decision-making. This paradigm encompasses **Generative World Models** (e.g., video/occupancy generation), **Embodied Vision-Language-Action (VLA)** models, and systems capable of **Open-World** reasoning.

📄 **[Paper Link](https://arxiv.org/abs/2512.24385)**


---

### Citation

If you find this work helpful for your research, please kindly consider citing our paper:

```bibtex
@article{wang2026forging,
    title   = {Forging Spatial Intelligence: A Roadmap of Multi-Modal Data Pre-Training for Autonomous Systems},
    author  = {Song Wang and Lingdong Kong and Xiaolu Liu and Hao Shi and Wentong Li and Jianke Zhu and Steven C. H. Hoi},
    journal = {arXiv preprint arXiv:2512.24385},
    year    = {2025}
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
  - [**4. Open-World Perception and Planning**](#4-open-world-perception-and-planning)
      - [Text-Grounded Understanding](#text-grounded-understanding)
      - [Unified World Representation for Action](#unified-world-representation-for-action)
  - [**5. Acknowledgements**](#5-acknowledgements)

-----

# 1. Benchmarks & Datasets

### Vehicle-Based Datasets


| Dataset | Venue | Sensor | Task | Website |
| :-: | :-: | :-: | :-: | :-: |
| `KITTI` | [CVPR'12](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf) | 2 Cam(RGB), 2 Cam(Gray), 1 LiDAR(64) | 3D Det, Stereo, Optical Flow, SLAM | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.cvlibs.net/datasets/kitti/index.php) |
| `ApolloScape` | [TPAMI'19](https://arxiv.org/pdf/1803.06184) | 2 Cam, 2 LiDAR | 3D Det, HD Map | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://apolloscape.auto/) |
| `nuScenes` | [CVPR'20](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf) | 6 Cam(RGB), 1 LiDAR(32), 5 Radar | 3D Det, Seg, Occ, Map | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.nuscenes.org/nuscenes#download) |
| `SemanticKITTI` | [ICCV'19](https://openaccess.thecvf.com/content_ICCV_2019/html/Behley_SemanticKITTI_A_Dataset_for_Semantic_Scene_Understanding_of_LiDAR_Sequences_ICCV_2019_paper.html) | 4 Cam, 1 LiDAR(64) | 3D Det, Occ | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](www.semantic-kitti.org) |
| `Waymo` | [CVPR'20](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf) | 5 Cam(RGB), 5 LiDAR | Perception (Det, Seg, Track), Motion | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://waymo.com/open/licensing/?continue=%2Fopen%2Fdownload%2F) |
| `Argoverse` | [CVPR'19](https://arxiv.org/pdf/1911.02620) | 7 Cam(RGB), 2 LiDAR(32) | 3D Tracking, Forecasting, Map | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.argoverse.org/) |
| `Lyft L5` | [CoRL'20](https://arxiv.org/pdf/2006.14480) | 7 Cam(RGB), 3 LiDAR, 5 Radar | 3D Det, Motion Forecasting/Planning | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://self-driving.lyft.com/level5/data) |
| `A*3D` | [ICRA'20](https://ieeexplore.ieee.org/abstract/document/9197385) | 2 Cam, 1 LiDAR(64) | 3D Det | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/I2RDL2/ASTAR-3D) |
| `KITTI-360` | [TPAMI'22](https://ieeexplore.ieee.org/abstract/document/9786676) | 4 Cam, 1 LiDAR(64) | 3D Det, Occ | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](cvlibs.net/datasets/kitti-360) |
| `A2D2` | [arXiv'20](https://arxiv.org/abs/2004.06320) | 6 Cam, 5 LiDAR(16) | 3D Det | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.a2d2.audi/en/) |
| `PandaSet` | [ITSC'21](https://ieeexplore.ieee.org/abstract/document/9565009) | 6 Cam(RGB), 2 LiDAR(64) | 3D Det, LiDAR Seg | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://scale.com/open-av-datasets/pandaset) |
| `Cirrus` | [ICRA'21](https://ieeexplore.ieee.org/abstract/document/9561267/) | 1 Cam, 2 LiDAR(64) | 3D Det | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://developer.volvocars.com/resources/cirrus/) |
| `ONCE` | [NeurIPS'21](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/67c6a1e7ce56d3d6fa748ab6d9af3fd7-Paper-round1.pdf) | 7 Cam(RGB), 1 LiDAR(40) | 3D Det (Self-supervised/Semi-supervised) | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://once-for-auto-driving.github.io/download.html) |
| `Shifts` | [arXiv'21](https://arxiv.org/abs/2107.07455) | - | 3D Det, HD Map | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://shifts.ai/) |
| `nuPlan` | [arXiv'21](https://arxiv.org/abs/2106.11810) | 8 Cam, 5 LiDAR | 3D Det, HD Map, E2E Plan | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://nuplan.org/) |
| `Argoverse2` | [NeurIPS'21](https://arxiv.org/abs/2301.00493) | 7 Cam, 2 LiDAR(32) | 3D Det, Occ, HD Map, E2E Plan | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.argoverse.org/av2.html) |
| `MONA` | [ITSC'22](https://ieeexplore.ieee.org/abstract/document/9922263) | 3 Cam | 3D Det, HD Map | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://gitlab.lrz.de/tum-cps/mona-dataset) |
| `Dual Radar` | [Sci. Data'25](https://www.nature.com/articles/s41597-025-04698-2) | 1 Cam, 1 LiDAR(80) 2 Radar| 3D Det | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/adept-thu/Dual-Radar) |
| `MAN TruckScenes` | [NeurIPS'24](https://arxiv.org/abs/2407.07462) | 4 Cam, 6 LiDAR(64), 6 RADAR | 3D Det | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://registry.opendata.aws/man-truckscenes/) |
| `OmniHD-Scenes` | [arXiv'24](https://arxiv.org/pdf/2412.10734) | 6 Cam, 1 LiDAR(128), 6 RADAR | 3D Det, Occ, HD Map | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.2077ai.com/OmniHD-Scenes) |
| `AevaScenes` | [2025](https://scenes.aeva.com/) |6 Cam, 6 LiDAR | 3D Det, HD Map | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://scenes.aeva.com) |
| `PhysicalAI-AV` | [2025](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | 7 Cam, 1 LiDAR, 11 RADAR | E2E Plan | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) |

### Drone-Based Datasets

| Dataset | Venue | Sensor | Task | Website |
| :-: | :-: | :-: | :-: | :-: |
| `Campus` | [ECCV'16](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_33) | 1 Cam |Target Forecasting/ Tracking | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](http://cvgl.stanford.edu/projects/uav_data/) |
| `UAV123` | [ECCV'16](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_27) | 1 Cam | UAV Trackong| [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://cemse.kaust.edu.sa/ivul/uav123) |
| `CarFusion` | [CVPR'18](https://openaccess.thecvf.com/content_cvpr_2018/html/Reddy_CarFusion_Combining_Point_CVPR_2018_paper.html) | 22 Cam | 3D Vehicle Reconstruction | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/cvpr2018/index.html) |
| `UAVDT` | [ECCV'18](https://openaccess.thecvf.com/content_ECCV_2018/html/Dawei_Du_The_Unmanned_Aerial_ECCV_2018_paper.html) | 1 Cam | 2D Object Detection/ Tracking | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://datasetninja.com/uavdt) |
| `DOTA` | [CVPR'18](https://openaccess.thecvf.com/content_cvpr_2018/html/Xia_DOTA_A_Large-Scale_CVPR_2018_paper.html) | Multi-Scoure | 2D Object Detection | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://captain-whu.github.io/DOTA/) |
| `VisDrone` | [TPAMI'21](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VISDrone/Du_VisDrone-DET2019_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ICCVW_2019_paper.pdf) | 1 Cam | 2D Object Detection/ Tracking | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/VisDrone/VisDrone-Dataset) |
| `DOTA V2.0` | [TPAMI'21](https://ieeexplore.ieee.org/abstract/document/9560031) | Multi-Scoure | 2D Object Detection| [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://captain-whu.github.io/DOTA/) |
| `MOR-UAV` | [MM'20](https://dl.acm.org/doi/abs/10.1145/3394171.3413934) | 1 Cam | Moving Object Recognation | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/murari023/mor-uav) |
| `AU-AIR` | [ICRA'20](https://ieeexplore.ieee.org/abstract/document/9196845) | 1 Cam | 2D Object Detection | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://bozcani.github.io/auairdataset) |
| `UAVid` | [ISPRS JPRS'20](https://arxiv.org/abs/1810.10438) | 1 Cam | Semantic Segmentation | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://uavid.nl) |
| `MOHR` | [Neuro'21](https://www.sciencedirect.com/science/article/abs/pii/S0925231220314338) | 3 Cam | 2D Object Detection | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.sciencedirect.com/science/article/abs/pii/S0925231220314338) |
| `SensatUrban` | [CVPR'21](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_Towards_Semantic_Segmentation_of_Urban-Scale_3D_Point_Clouds_A_Dataset_CVPR_2021_paper.html) | 1 Cam | 2D Object Detection | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/QingyongHu/SensatUrban) |
| `UAVDark135` | [TMC'22](https://ieeexplore.ieee.org/abstract/document/9744417) | 1 Cam | 2D Object Tracking | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/vision4robotics/ADTrack_v2) |
| `MAVREC` | [CVPR'24](https://openaccess.thecvf.com/content/CVPR2024/html/Dutta_Multiview_Aerial_Visual_RECognition_MAVREC_Can_Multi-view_Improve_Aerial_Visual_CVPR_2024_paper.html) | 1 Cam | 2D Obejct Detection | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://mavrec.github.io/) |
| `BioDrone` | [IJCV'24](https://arxiv.org/abs/2402.04519) | 1 Cam | 2D Object Tracking | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](http://biodrone.aitestunion.com/) |
| `PDT` | [ECCV'24](https://link.springer.com/chapter/10.1007/978-3-031-73116-7_4) | 1 Cam, 1 LiDAR | 2D Object Detection | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/RuiXing123/PDT_CWC_YOLO-DP) |
| `UAV3D` | [NeurIPS'24](https://proceedings.neurips.cc/paper_files/paper/2024/hash/643ea9b3843bd8f525ad8801437a2022-Abstract-Datasets_and_Benchmarks_Track.html) | 5 Cam | 3D Object Detection/ Tracking| [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://huiyegit.github.io/UAV3D_Benchmark/) |
| `IndraEye` | [arXiv'24](https://arxiv.org/abs/2410.20953) | 1 Cam | 2D Object Detection/ Semantic Segmentation | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://sites.google.com/view/indraeye) |
| `UAVScenes` | [ICCV'25](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_UAVScenes_A_Multi-Modal_Dataset_for_UAVs_ICCV_2025_paper.html) | 1 Cam, 1 LiDAR | Semantic Segmentation, Visual Localization | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/sijieaaa/UAVScenes) |

### Other Robotic Platforms

| Dataset | Venue | Platform | Sensors | Website |
|:-:|:-:|:-|:-:|:-:|
| `RailSem19` | [CVPRW'19](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Zendel_RailSem19_A_Dataset_for_Semantic_Rail_Scene_Understanding_CVPRW_2019_paper.pdf) | Railway | 1× Camera | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://www.wilddash.cc/railsem19) |
| `FRSign` | [arXiv'20](https://arxiv.org/abs/2002.05665) | Railway | 2× Camera (Stereo) | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://frsign.irt-systemx.fr/) |
| `RAWPED` | [TVT'20](https://ieeexplore.ieee.org/abstract/document/9050835) | Railway | 1× Camera | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://zenodo.org/records/3741742)|
| `SRLC` | [AutCon'21](https://www.sciencedirect.com/science/article/pii/S0926580521002909) | Railway | 1× LiDAR | |
| `Rail-DB` | [MM'22](https://dl.acm.org/doi/abs/10.1145/3503161.3548050) | Railway | 1× Camera | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/Sampson-Lee/Rail-Detection) |
| `RailSet` | [IPAS'22](https://ieeexplore.ieee.org/abstract/document/10052883) | Railway | 1× Camera |  |
| `OSDaR23` | [ICRAE'23](https://ieeexplore.ieee.org/abstract/document/10458449) | Railway | 9× Camera, 6× LiDAR, 1× Radar | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://osdar23.com/) |
| `Rail3D` | [Infra'24](https://www.researchgate.net/publication/379701734_Multi-Context_Point_Cloud_Dataset_and_Machine_Learning_for_Railway_Semantic_Segmentation) | Railway | 4× Camera, 1× LiDAR | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/akharroubi/Rail3D) |
| `WHU-Railway3D` | [TITS'24](https://ieeexplore.ieee.org/abstract/document/10716569) | Railway | 1× LiDAR | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/WHU-USI3DV/WHU-Railway3D) |
| `FloW` | [ICCV'21](https://openaccess.thecvf.com/content/ICCV2021/html/Cheng_FloW_A_Dataset_and_Benchmark_for_Floating_Waste_Detection_in_ICCV_2021_paper.html) | USV (Water) | 2× Camera, 1× 4D Radar | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/ORCA-Uboat/FloW-Dataset) |
| `DartMouth` | [IROS'21](https://ieeexplore.ieee.org/abstract/document/9636028) | USV (Water) | 3× Camera, 1× LiDAR | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/dartmouthrobotics/asv_detection_dataset.git) |
| `MODS` | [TITS'21](https://arxiv.org/abs/2105.02359) | USV (Water) | 2× Camera, 1× LiDAR | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/bovcon/mods) |
| `SeaSAW` | [CVPRW'22](https://openaccess.thecvf.com/content/CVPR2022W/Precognition/html/Kaur_Sea_Situational_Awareness_SeaSAW_Dataset_CVPRW_2022_paper.html) | USV (Water) | 5× Camera | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://sea-machines.com/introducing-sea-machines-industry-leading-sea-situational-awareness-seasaw-dataset/) |
| `WaterScenes` | [T-ITS'24](https://ieeexplore.ieee.org/document/10571852) | USV (Water) | 1× Camera, 1× 4D Radar | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/WaterScenes/WaterScenes) |
| `MVDD13` | [Appl. Ocean Res.'24](https://www.sciencedirect.com/science/article/pii/S0141118723003760) | USV (Water) | 1× Camera | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/yyuanwang1010/MVDD13) |
| `SeePerSea` | [TFR'25](https://arxiv.org/pdf/2404.18411) | USV (Water) | 1× Camera, 1× LiDAR | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://seepersea.github.io) |
| `WaterVG` | [TITS'25](https://ieeexplore.ieee.org/abstract/document/10847630) | USV (Water) | 1× Camera, 1× 4D Radar | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/WaterVG/WaterVG) |
| `Han et al.` | [NMI'24](https://arxiv.org/abs/2308.15143) | Legged Robot | 1× Depth Camera | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://tencent-roboticsx.github.io/lifelike-agility-and-play/) |
| `Luo et al.` | [CVPR'25](https://openaccess.thecvf.com/content/CVPR2025/html/Luo_Omnidirectional_Multi-Object_Tracking_CVPR_2025_paper.html) | Legged Robot | 1× Panoramic Camera | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/xifen523/OmniTrack) |
| `QuadOcc` | [arXiv'25](https://arxiv.org/abs/2511.03571) | Legged Robot | 1× Panoramic Camera, 1× LiDAR | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://github.com/MasterHow/OneOcc) |
| `M3ED` | [CVPRW'23](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/html/Chaney_M3ED_Multi-Robot_Multi-Sensor_Multi-Environment_Event_Dataset_CVPRW_2023_paper.html) | Multi-Robot | 3× Camera, 2× Event Camera, 1× LiDAR | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://m3ed.io/download/) |
| `Pi3DET` | [ICCV'25](https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Perspective-Invariant_3D_Object_Detection_ICCV_2025_paper.html) | Multi-Robot | 3× Camera, 2× Event Camera, 1× LiDAR | [![Website](https://img.shields.io/badge/Link-yellow?style=flat-square&logo=gitbook)](https://pi3det.github.io/) |


-----

# 2. Single-Modality Pre-Training

### LiDAR-Only

> *Methods utilizing Point Cloud Contrastive Learning, Masked Autoencoders (MAE), or Forecasting.*

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `PointContrast` | [Unsupervised Pre-training for 3D Point Cloud Understanding](https://arxiv.org/abs/2007.10985) | ECCV 2020 | [![GitHub](https://img.shields.io/github/stars/facebookresearch/PointContrast)](https://github.com/facebookresearch/PointContrast) |
| `DepthContrast` | [Self-supervised Pretraining of 3D Features on any Point-Cloud](https://arxiv.org/abs/2101.02691) | ICCV 2021 | [![GitHub](https://img.shields.io/github/stars/facebookresearch/DepthContrast)](https://github.com/facebookresearch/DepthContrast) |
| `GCC-3D` | [Exploring geometry-aware contrast and clustering harmonization for self-supervised 3d object detection](https://openaccess.thecvf.com/content/ICCV2021/html/Liang_Exploring_Geometry-Aware_Contrast_and_Clustering_Harmonization_for_Self-Supervised_3D_Object_ICCV_2021_paper.html) | ICCV 2021 | [](https://www.google.com/search?q=) |
| `ContrastiveSceneContexts` | [Exploring data-efficient 3d scene understanding with contrastive scene contexts](https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Exploring_Data-Efficient_3D_Scene_Understanding_With_Contrastive_Scene_Contexts_CVPR_2021_paper.html) | CVPR 2021 | [![GitHub](https://img.shields.io/github/stars/facebookresearch/ContrastiveSceneContexts)](https://github.com/facebookresearch/ContrastiveSceneContexts) |
| `SegContrast` | [3D Point Cloud Feature Representation Learning through Self-supervised Segment Discrimination](https://ieeexplore.ieee.org/document/9681336)| RA-L 2021 | [![GitHub](https://img.shields.io/github/stars/PRBonn/segcontrast)](https://github.com/PRBonn/segcontrast) |
| `GroupContrast` | [Groupcontrast: Semantic-aware self-supervised representation learning for 3d understanding](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_GroupContrast_Semantic-aware_Self-supervised_Representation_Learning_for_3D_Understanding_CVPR_2024_paper.html) | CVPR 2024 | [![GitHub](https://img.shields.io/github/stars/dvlab-research/GroupContrast)](https://github.com/dvlab-research/GroupContrast) |
| `ProposalContrast` | [Unsupervised Pre-training for LiDAR-Based 3D Object Detection](https://arxiv.org/abs/2207.12654) | ECCV 2022 | [![GitHub](https://img.shields.io/github/stars/yinjunbo/ProposalContrast)](https://github.com/yinjunbo/ProposalContrast) |
| `Occupancy-MAE` | [Self-supervised Pre-training Large-scale LiDAR Point Clouds with Masked Occupancy Autoencoders](https://arxiv.org/abs/2206.09900) | T-IV 2023 | [![GitHub](https://img.shields.io/github/stars/chaytonmin/Occupancy-MAE)](https://github.com/chaytonmin/Occupancy-MAE) |
| `ALSO` |[Automotive LiDAR Self-supervision by Occupancy Estimation](https://arxiv.org/abs/2212.05867) | CVPR 2023 | [![GitHub](https://img.shields.io/github/stars/valeoai/ALSO)](https://github.com/valeoai/ALSO) |
| `GD-MAE` | [Generative Decoder for MAE Pre-training on LiDAR Point Clouds](https://arxiv.org/abs/2212.03010) | CVPR 2023 | [![GitHub](https://img.shields.io/github/stars/Nightmare-n/GD-MAE)](https://github.com/Nightmare-n/GD-MAE) |
| `AD-PT` | [Autonomous Driving Pre-Training with Large-scale Point Cloud Dataset](https://arxiv.org/abs/2306.00612) | NeurIPS 2023 | [![GitHub](https://img.shields.io/github/stars/PJLab-ADG/3DTrans)](https://github.com/PJLab-ADG/3DTrans) |
| `E-SSL` | [Equivariant spatio-temporal self-supervision for lidar object detection](https://link.springer.com/chapter/10.1007/978-3-031-73347-5_27) | ECCV 2024 | [](https://www.google.com/search?q=) |
| `PatchContrast` | [Self-Supervised Pre-training for 3D Object Detection](https://arxiv.org/abs/2308.06985) | CVPRW 2025 | [](https://www.google.com/search?q=) |
| `MV-JAR` | [Mv-jar: Masked voxel jigsaw and reconstruction for lidar-based self-supervised pre-training](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_MV-JAR_Masked_Voxel_Jigsaw_and_Reconstruction_for_LiDAR-Based_Self-Supervised_Pre-Training_CVPR_2023_paper.html) | CVOR 2023 | [![GitHub](https://img.shields.io/github/stars/SmartBot-PJLab/MV-JAR)](https://github.com/SmartBot-PJLab/MV-JAR) |
| `Occupancy-MAE` | [Occupancy-mae: Self-supervised pre-training large-scale lidar point clouds with masked occupancy autoencoders](https://ieeexplore.ieee.org/abstract/document/10273603) | TIV 2023 | [![GitHub](https://img.shields.io/github/stars/chaytonmin/Occupancy-MAE)](https://github.com/chaytonmin/Occupancy-MAE) |
| `Core` | [Core: Cooperative reconstruction for multi-agent perception](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_CORE_Cooperative_Reconstruction_for_Multi-Agent_Perception_ICCV_2023_paper.html) | ICCV 2023 | [![GitHub](https://img.shields.io/github/stars/zllxot/CORE)](https://github.com/zllxot/CORE) |
| `MAELi` | [Masked Autoencoder for Large-Scale LiDAR Point Clouds](https://arxiv.org/abs/2212.07207) | WACV 2024 | [](https://www.google.com/search?q=) |
| `BEV-MAE` | [Bird's Eye View Masked Autoencoders for Point Cloud Pre-training](https://arxiv.org/abs/2212.05758) | AAAI 2024 | [![GitHub](https://img.shields.io/github/stars/VDIGPKU/BEV-MAE)](https://github.com/VDIGPKU/BEV-MAE) |
| `AD-L-JEPA` | [AD-L-JEPA: Self-Supervised Spatial World Models with Joint Embedding Predictive Architecture for Autonomous Driving with LiDAR Data](https://arxiv.org/abs/2501.04969) | AAAI 2026 | [![GitHub](https://img.shields.io/github/stars/HaoranZhuExplorer/adljepa)](https://github.com/HaoranZhuExplorer/adljepa) |
| `UnO` | [Unsupervised Occupancy Fields for Perception and Forecasting](https://arxiv.org/abs/2406.08691) | CVPR 2024 | [](https://www.google.com/search?q=) |
| `BEVContrast` | [Self-Supervision in BEV Space for Automotive Lidar Point Clouds](https://arxiv.org/abs/2310.17281) | 3DV 2024 | [![GitHub](https://img.shields.io/github/stars/valeoai/BEVContrast)](https://github.com/valeoai/BEVContrast) |
| `4DContrast` | [4dcontrast: Contrastive learning with dynamic correspondences for 3d scene understanding](https://link.springer.com/chapter/10.1007/978-3-031-19824-3_32) | ECCV 2022 | [![GitHub](https://img.shields.io/github/stars/TerenceCYJ/4DContrast)](https://github.com/TerenceCYJ/4DContrast) |
| `Copilot4D` | [Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion](https://arxiv.org/abs/2311.01017) | ICLR 2024 | [](https://www.google.com/search?q=) |
| `T-MAE` | [Temporal Masked Autoencoders for Point Cloud Representation Learning](https://arxiv.org/abs/2312.10217) | ECCV 2024 | [![GitHub](https://img.shields.io/github/stars/codename1995/t-mae)](https://github.com/codename1995/t-mae) |
| `PICTURE` | [Point Cloud Reconstruction Is Insufficient to Learn 3D Representations](https://dl.acm.org/doi/10.1145/3664647.3680890) | ACM MM 2024 | [](https://www.google.com/search?q=) |
| `LSV-MAE` | [Rethinking Masked-Autoencoder-Based 3D Point Cloud Pretraining](https://ieeexplore.ieee.org/document/10588770) | IV 2024 | [](https://www.google.com/search?q=) |
| `UNIT` | [Unsupervised Online Instance Segmentation through Time](https://arxiv.org/abs/2409.07887) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/valeoai/UNIT)](https://github.com/valeoai/UNIT) |
| `R-MAE` | [Sense Less, Generate More: Pre-training LiDAR Perception with Masked Autoencoders](https://arxiv.org/abs/2406.07833) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/sinatayebati/R-MAE)](https://github.com/sinatayebati/R-MAE) |
| `TurboTrain` | [TurboTrain: Towards efficient and balanced multi-task learning for multi-agent perception and prediction](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_TurboTrain_Towards_Efficient_and_Balanced_Multi-Task_Learning_for_Multi-Agent_Perception_ICCV_2025_paper.html) | ICCV 2025 | [![GitHub](https://img.shields.io/github/stars/ucla-mobility/TurboTrain)](https://github.com/ucla-mobility/TurboTrain) |
| `NOMAE` | [Multi-Scale Neighborhood Occupancy Masked Autoencoder for Self-Supervised Learning in LiDAR Point Clouds](https://openaccess.thecvf.com/content/CVPR2025/html/Abdelsamad_Multi-Scale_Neighborhood_Occupancy_Masked_Autoencoder_for_Self-Supervised_Learning_in_LiDAR_CVPR_2025_paper.html) | CVPR 2025 | [](https://www.google.com/search?q=) |
| `4D Occ` | [Point cloud forecasting as a proxy for 4d occupancy forecasting](https://openaccess.thecvf.com/content/CVPR2023/html/Khurana_Point_Cloud_Forecasting_as_a_Proxy_for_4D_Occupancy_Forecasting_CVPR_2023_paper.html) | CVPR 2023 | [![GitHub](https://img.shields.io/github/stars/tarashakhurana/4d-occ-forecasting)](https://github.com/tarashakhurana/4d-occ-forecasting) |
| `GPICTURE` | [Mutual information-driven self-supervised point cloud pre-training](https://www.sciencedirect.com/science/article/abs/pii/S0950705124013753) | KBS 2025 | [](https://www.google.com/search?q=) |
| `CooPre` | [CooPre: Cooperative pretraining for v2x cooperative perception](https://arxiv.org/abs/2408.11241) | IROS 2025 | [![GitHub](https://img.shields.io/github/stars/ucla-mobility/CooPre)](https://github.com/ucla-mobility/CooPre) |
| `TREND` | [TREND: Unsupervised 3D Representation Learning via Temporal Forecasting for LiDAR Perception](https://arxiv.org/abs/2412.03054) | arXiv 2024 | [](https://www.google.com/search?q=) |


### Camera-Only

> *Self-supervised learning from image sequences for driving/robotics.*

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `INoD` | [Injected Noise Discriminator for Self-Supervised Representation](https://arxiv.org/abs/2303.18101) | RA-L 2023 | [![GitHub](https://img.shields.io/github/stars/robot-learning-freiburg/INoD)](https://github.com/robot-learning-freiburg/INoD/) |
| `TempO` | [Self-Supervised Representation Learning From Temporal Ordering](https://arxiv.org/abs/2302.09043) | RA-L 2024 | [](https://www.google.com/search?q=) |
| `LetsMap` | [Unsupervised Representation Learning for Label-Efficient Semantic BEV Mapping](https://dl.acm.org/doi/abs/10.1007/978-3-031-73636-0_7) | ECCV 2024 | [](https://www.google.com/search?q=) |
| `NeRF-MAE` | [Masked AutoEncoders for Self-Supervised 3D Representation Learning](https://arxiv.org/abs/2404.01300) | ECCV 2024 | [![GitHub](https://img.shields.io/github/stars/zubair-irshad/NeRF-MAE)](https://github.com/zubair-irshad/NeRF-MAE) |
| `VisionPAD` | [A Vision-Centric Pre-training Paradigm for Autonomous Driving](https://arxiv.org/abs/2411.14716) | arXiv 2024 | [](https://www.google.com/search?q=) |

-----

# 3. Multi-Modality Pre-Training

### LiDAR-Centric Pre-Training

> *Enhancing LiDAR representations using Vision foundation models (Knowledge Distillation).*

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `SLidR` | [Image-to-Lidar Self-Supervised Distillation](https://arxiv.org/abs/2203.16258) | CVPR 2022 | [![GitHub](https://img.shields.io/github/stars/valeoai/SLidR)](https://github.com/valeoai/SLidR) |
| `SimIPU` | [Simipu: Simple 2d image and 3d point cloud unsupervised pre-training for spatial-aware visual representations](https://ojs.aaai.org/index.php/AAAI/article/view/20040) | AAAI 2022 | [![GitHub](https://img.shields.io/github/stars/zhyever/SimIPU)](https://github.com/zhyever/SimIPU) |
| `SSPC-Im` | [Self-supervised pre-training of 3d point cloud networks with image data](https://arxiv.org/abs/2211.11801) | CoRL 2022 | [](https://www.google.com/search?q=) |
| `ST-SLidR` | [Self-Supervised Image-to-Point Distillation via Semantically Tolerant Contrastive Loss](https://arxiv.org/abs/2301.05709) | CVPR 2023 | [](https://www.google.com/search?q=) |
| `I2P-MAE` | [Learning 3D Representations from 2D Pre-trained Models via Image-to-Point MAE](https://arxiv.org/abs/2212.06785) | CVPR 2023 | [![GitHub](https://img.shields.io/github/stars/ZrrSkywalker/I2P-MAE)](https://github.com/ZrrSkywalker/I2P-MAE) |
| `TriCC` | [Unsupervised 3D Point Cloud Representation Learning by Triangle Constrained Contrast](https://ieeexplore.ieee.org/document/10203592) | CVPR 2023 | [](https://www.google.com/search?q=) |
| `Seal` | [Segment Any Point Cloud Sequences by Distilling Vision FMs](https://arxiv.org/abs/2306.09347) | NeurIPS 23 | [![GitHub](https://img.shields.io/github/stars/youquanl/Segment-Any-Point-Cloud)](https://github.com/youquanl/Segment-Any-Point-Cloud) |
| `PRED` | [Pre-training via Semantic Rendering on LiDAR Point Clouds](https://arxiv.org/abs/2311.04501) | NeurIPS 23 | [](https://www.google.com/search?q=) |
| `LiMA` | [Beyond one shot, beyond one perspective: Cross-view and long-horizon distillation for better lidar representations](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Beyond_One_Shot_Beyond_One_Perspective_Cross-View_and_Long-Horizon_Distillation_ICCV_2025_paper.html) | ICCV 2025 | [![GitHub](https://img.shields.io/github/stars/Xiangxu-0103/LiMA)](https://github.com/Xiangxu-0103/LiMA) |
| `ImageTo360` | [360° from a Single Camera: A Few-Shot Approach for LiDAR Segmentation](https://arxiv.org/abs/2309.06197) | ICCVW 2023 | [](https://www.google.com/search?q=) |
| `ScaLR` | [Three Pillars improving Vision Foundation Model Distillation for Lidar](https://arxiv.org/abs/2310.17504) | CVPR 2024 | [](https://github.com/ZrrSkywalker/ScaLR) |
| `CSC` | [Building a Strong Pre-Training Baseline for Universal 3D Large-Scale Perception](https://arxiv.org/abs/2405.07201) | CVPR 2024 | [![GitHub](https://img.shields.io/github/stars/chenhaomingbob/CSC)](https://github.com/chenhaomingbob/CSC) |
| `GPC` | [Pre-Training LiDAR-Based 3D Object Detectors Through Colorization](https://arxiv.org/abs/2310.14592v2) | ICLR 2024 | [![GitHub](https://img.shields.io/github/stars/tydpan/GPC)](https://github.com/tydpan/GPC) |
| `Cross-Modal SSL` | [Cross-Modal Self-Supervised Learning with Effective Contrastive Units](https://arxiv.org/abs/2409.06827) | IROS 2024 | [![GitHub](https://img.shields.io/github/stars/qcraftai/cross-modal-ssl)](https://github.com/qcraftai/cross-modal-ssl) |
| `SuperFlow` | [4D Contrastive Superflows are Dense 3D Representation Learners](https://arxiv.org/abs/2407.06190) | ECCV 2024 | [![GitHub](https://img.shields.io/github/stars/Xiangxu-0103/SuperFlow)](https://github.com/Xiangxu-0103/SuperFlow) |
| `Rel` | [Image-to-Lidar Relational Distillation for Autonomous Driving Data](https://arxiv.org/abs/2409.00845) | ECCV 2024 | [](https://www.google.com/search?q=) |
| `HVDistill` | [Transferring Knowledge from Images to Point Clouds via Unsupervised Hybrid-View Distillation](https://arxiv.org/abs/2409.00845) | IJCV 2024 | [![GitHub](https://img.shields.io/github/stars/zhangsha1024/HVDistill)](https://github.com/zhangsha1024/HVDistill) |
| `RadarContrast` | [Self-Supervised Contrastive Learning for Camera-to-Radar Knowledge Distillation](https://ieeexplore.ieee.org/document/10621525) | DCOSS-IoT 2024 | [](https://www.google.com/search?q=) |
| `CM3D` | [Shelf-Supervised Cross-Modal Pre-Training for 3D Object Detection](https://arxiv.org/abs/2406.10115) | CoRL 2024 | [![GitHub](https://img.shields.io/github/stars/meharkhurana03/cm3d)](https://github.com/meharkhurana03/cm3d) |
| `OLIVINE` | [Fine-grained Image-to-LiDAR Contrastive Distillation with Visual Foundation Models](https://arxiv.org/abs/2405.14271) | NeurIPS 2024 | [![GitHub](https://img.shields.io/github/stars/Eaphan/OLIVINE)](https://github.com/Eaphan/OLIVINE) |
| `EUCA-3DP` | [Exploring the Untouched Sweeps for Conflict-Aware 3D Segmentation Pretraining](https://arxiv.org/abs/2407.07465) | arXiv 2024 | [](https://www.google.com/search?q=) |
| `GASP` | [Gasp: Unifying geometric and semantic self-supervised pre-training for autonomous driving](https://arxiv.org/abs/2503.15672) | arXiv 2025 | [![GitHub](https://img.shields.io/github/stars/LiljaAdam/gasp)](https://github.com/LiljaAdam/gasp) |
| `BALViT` | [Label-Efficient LiDAR Scene Understanding with 2D-3D Vision Transformer Adapters](https://openreview.net/forum?id=w3MTdtHYKY) | ICRAW 2025 | [](https://www.google.com/search?q=) |



### Camera-Centric Pre-Training

> *Learning 3D Geometry from Camera inputs using LiDAR supervision.*

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `DD3D` | [Is Pseudo-Lidar needed for Monocular 3D Object detection?](https://arxiv.org/abs/2108.06417) | ICCV 2021 | [![GitHub](https://img.shields.io/github/stars/TRI-ML/dd3d)](https://github.com/TRI-ML/dd3d) |
| `DEPT` | [Delving into the Pre-training Paradigm of Monocular 3D Object Detection](https://arxiv.org/abs/2206.03657) | arXiv 2022 | [](https://www.google.com/search?q=) |
| `OccNet` | [Scene as Occupancy](https://arxiv.org/abs/2306.02851) | ICCV 2023 | [![GitHub](https://img.shields.io/github/stars/OpenDriveLab/OccNet)](https://github.com/OpenDriveLab/OccNet) |
| `GeoMIM` | [Towards Better 3D Knowledge Transfer via Masked Image Modeling](https://arxiv.org/abs/2303.11325) | ICCV 2023 | [![GitHub](https://img.shields.io/github/stars/Sense-X/GeoMIM)](https://github.com/Sense-X/GeoMIM) |
| `GAPretrain` | [Geometric-aware Pretraining for Vision-centric 3D Object Detection](https://arxiv.org/abs/2304.03105) | arXiv 2023 | [![GitHub](https://img.shields.io/github/stars/OpenDriveLab/Birds-eye-view-Perception)](https://github.com/OpenDriveLab/Birds-eye-view-Perception) |
| `UniScene` |[Multi-Camera Unified Pre-training via 3D Scene Reconstruction](https://arxiv.org/abs/2305.18829v5) | RA-L 2024 | [![GitHub](https://img.shields.io/github/stars/chaytonmin/UniScene)](https://github.com/chaytonmin/UniScene) |
| `SelfOcc` | [Self-Supervised Vision-Based 3D Occupancy Prediction](https://arxiv.org/abs/2311.12754) | CVPR 2024 | [![GitHub](https://img.shields.io/github/stars/huang-yh/SelfOcc)](https://github.com/huang-yh/SelfOcc) |
| `ViDAR` | [Visual Point Cloud Forecasting enables Scalable Autonomous Driving](https://arxiv.org/abs/2312.17655) | CVPR 2024 | [![GitHub](https://img.shields.io/github/stars/OpenDriveLab/ViDAR)](https://github.com/OpenDriveLab/ViDAR) |
| `DriveWorld` | [4D Pre-trained Scene Understanding via World Models](https://arxiv.org/abs/2405.04390) | CVPR 2024 | [](https://www.google.com/search?q=) |
| `OccFeat` | [Self-supervised Occupancy Feature Prediction for Pretraining BEV Segmentation](https://arxiv.org/abs/2404.14027) | CVPRW 2024 | [](https://www.google.com/search?q=) |
| `OccWorld` | [Learning a 3D Occupancy World Model for Autonomous Driving](https://arxiv.org/abs/2311.16038) | ECCV 2024 | [![GitHub](https://img.shields.io/github/stars/wzzheng/OccWorld)](https://github.com/wzzheng/OccWorld) |
| `MVS3D` | [Exploiting the Potential of Multi-Frame Stereo Depth Estimation Pre-training](https://ieeexplore.ieee.org/abstract/document/10650924) | IJCNN 2024 | [](https://www.google.com/search?q=) |
| `OccSora` | [4D Occupancy Generation Models as World Simulators](https://arxiv.org/abs/2405.20337) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/wzzheng/OccSora)](https://github.com/wzzheng/OccSora) |
| `MIM4D` | [Masked Modeling with Multi-View Video for Autonomous Driving](https://arxiv.org/abs/2403.08760) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/hustvl/MIM4D)](https://github.com/hustvl/MIM4D) |
| `GaussianPretrain` | [A Simple Unified 3D Gaussian Representation for Visual Pre-training](https://arxiv.org/abs/2411.12452) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/Public-BOTs/GaussianPretrain)](https://github.com/Public-BOTs/GaussianPretrain) |
| `S3PT` | [S3pt: Scene semantics and structure guided clustering to boost self-supervised pre-training for autonomous driving](https://arxiv.org/abs/2410.23085) | WACV 2025 | []() |
| `UniFuture` | [Seeing the Future, Perceiving the Future: A Unified Driving World Model for Future Generation and Perception](https://github.com/dk-liang/UniFuture) | arXiv 2025 | [![GitHub](https://img.shields.io/github/stars/dk-liang/UniFuture)](https://github.com/dk-liang/UniFuture) |
| `GaussianOcc` | [Gaussianocc: Fully self-supervised and efficient 3d occupancy estimation with gaussian splatting](https://openaccess.thecvf.com/content/ICCV2025/html/Gan_GaussianOcc_Fully_Self-supervised_and_Efficient_3D_Occupancy_Estimation_with_Gaussian_ICCV_2025_paper.html) | ICCV 2025 | [![GitHub](https://img.shields.io/github/stars/GANWANSHUI/GaussianOcc)](https://github.com/GANWANSHUI/GaussianOcc) |
| `GaussianTR` | [Gausstr: Foundation model-aligned gaussian transformer for self-supervised 3d spatial understanding](https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_GaussTR_Foundation_Model-Aligned_Gaussian_Transformer_for_Self-Supervised_3D_Spatial_Understanding_CVPR_2025_paper.html) | CVPR 2025 | [![GitHub](https://img.shields.io/github/stars/hustvl/GaussTR)](https://github.com/hustvl/GaussTR) |
| `DistillNeRF` | [Distillnerf: Perceiving 3d scenes from single-glance images by distilling neural fields and foundation model features](https://proceedings.neurips.cc/paper_files/paper/2024/file/720991812855c99df50bc8b36966cd81-Paper-Conference.pdf) | NeurIPS 2024 | [![GitHub](https://img.shields.io/github/stars/NVlabs/distillnerf)](https://github.com/NVlabs/distillnerf) |

### Unified Pre-Training

> *Joint optimization of multi-modal encoders for unified representations.*

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `PonderV2` | [Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm](https://arxiv.org/abs/2310.08586) | arXiv 2023 | [![GitHub](https://img.shields.io/github/stars/OpenGVLab/PonderV2)](https://github.com/OpenGVLab/PonderV2) |
| `UniPAD` | [A Universal Pre-training Paradigm for Autonomous Driving](https://arxiv.org/abs/2310.08370) | CVPR 2024 | [![GitHub](https://img.shields.io/github/stars/Nightmare-n/UniPAD)](https://github.com/Nightmare-n/UniPAD) |
| `UniM2AE` | [Multi-Modal Masked Autoencoders with Unified 3D Representation](https://arxiv.org/abs/2308.10421) | ECCV 2024 | [![GitHub](https://img.shields.io/github/stars/hollow-503/UniM2AE)](https://github.com/hollow-503/UniM2AE) |
| `ConDense` | [Consistent 2D/3D Pre-training for Dense and Sparse Features](https://arxiv.org/abs/2408.17027) | ECCV 2024 | [](https://www.google.com/search?q=) |
| `Unified Pretrain` | [Learning Shared RGB-D Fields: Unified Self-supervised Pre-training](https://arxiv.org/abs/2405.17942) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/Xiaohao-Xu/Unified-Pretrain-AD)](https://github.com/Xiaohao-Xu/Unified-Pretrain-AD/) |
| `BEVWorld` | [A Multimodal World Simulator for Autonomous Driving via Unified BEV Latent Space](https://arxiv.org/abs/2407.05679) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/zympsyche/BevWorld)](https://github.com/zympsyche/BevWorld) |
| `NS-MAE` | [Learning Shared RGB-D Fields: Unified Self-supervised Pre-training for Label-efficient LiDAR-Camera 3D Perception](https://arxiv.org/abs/2405.17942) | arXiv 2024 | []() |
| `CLAP` | [CLAP: Unsupervised 3D Representation Learning for Fusion 3D Perception via Curvature Sampling and Prototype Learning](https://arxiv.org/abs/2412.03059) | arXiv 2024 | []() |
| `GS3` | [Point Cloud Unsupervised Pre-training via 3D Gaussian Splatting](https://arxiv.org/abs/2411.18667) | arXiv 2024 | []() |
| `Hermes` | [Hermes: A unified self-driving world model for simultaneous 3d scene understanding and generation](https://arxiv.org/abs/2501.14729) | ICCV 2025 | [![GitHub](https://img.shields.io/github/stars/LMD0311/HERMES)](https://github.com/LMD0311/HERMES) |
| `LRS4Fusion` | [Self-Supervised Sparse Sensor Fusion for Long Range Perception](https://openaccess.thecvf.com/content/ICCV2025/html/Palladin_Self-Supervised_Sparse_Sensor_Fusion_for_Long_Range_Perception_ICCV_2025_paper.html) | ICCV 2025 | [![GitHub](https://img.shields.io/github/stars/princeton-computational-imaging/LRS4Fusion)](https://github.com/princeton-computational-imaging/LRS4Fusion) |
| `Gaussian2Scene` | [Gaussian2Scene: 3D Scene Representation Learning via Self-supervised Learning with 3D Gaussian Splatting](https://arxiv.org/abs/2506.08777) | arXiv 2025 | []() |

### Incorporating Additional Sensors: With Radar

> *Incorporating additional modalities into pre-training frameworks for representation learning.*


| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `RadarContrast` | [Self-Supervised Contrastive Learning for Camera-to-Radar Knowledge Distillation](https://ieeexplore.ieee.org/abstract/document/10621525) | DCOSS-IoT 2024 | []() |
| `AssociationNet` | [Radar camera fusion via representation learning in autonomous driving](https://openaccess.thecvf.com/content/CVPR2021W/MULA/html/Dong_Radar_Camera_Fusion_via_Representation_Learning_in_Autonomous_Driving_CVPRW_2021_paper.html?trk=public_post_comment-text) | CVPRW 2021 | []() |
| `MVRAE` | [Multi-View Radar Autoencoder for Self-Supervised Automotive Radar Representation Learning](https://ieeexplore.ieee.org/abstract/document/10588463) | IV 2024 | []() |
| `SSRLD` | [Self-supervised representation learning for the object detection of marine radar](https://dl.acm.org/doi/abs/10.1145/3532213.3532328) | ICCAI 2022 | []() |
| `U-MLPNet` | [Learning Omni-Dimensional Spatio-Temporal Dependencies for Millimeter-Wave Radar Perception](https://www.mdpi.com/2072-4292/16/22/4256) | Remote Sens 2024 | []() |
| `4D-ROLLS` | [4D-ROLLS: 4D Radar Occupancy Learning via LiDAR Supervision](https://arxiv.org/abs/2505.13905) | arXiv 2025 | [![GitHub](https://img.shields.io/github/stars/CLASS-Lab/4D-ROLLS)](https://github.com/CLASS-Lab/4D-ROLLS) |
| `SS-RODNet` | [Pre-Training For mmWave Radar Object Detection Through Masked Image Modeling](https://ieeexplore.ieee.org/abstract/document/10424733) | SS-RODNet | []() |
| `Radical` | [Bootstrapping autonomous driving radars with self-supervised learning](https://openaccess.thecvf.com/content/CVPR2024/html/Hao_Bootstrapping_Autonomous_Driving_Radars_with_Self-Supervised_Learning_CVPR_2024_paper.html) | CVPR 2024 | [![GitHub](https://img.shields.io/github/stars/yiduohao/Radical)](https://github.com/yiduohao/Radical) |
| `RiCL` | [Leveraging Self-Supervised Instance Contrastive Learning for Radar Object Detection](https://arxiv.org/abs/2402.08427) | arXiv 2024 | []() |
| `RSLM` | [Radar spectra-language model for automotive scene parsing](https://ieeexplore.ieee.org/abstract/document/10993898) | RADAR 2024 | []() |

### Incorporating Additional Sensors: With Event Camera

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `ECDP` | [Event Camera Data Pre-training](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Event_Camera_Data_Pre-training_ICCV_2023_paper.html) | ICCV 2023 | [![GitHub](https://img.shields.io/github/stars/Yan98/Event-Camera-Data-Pre-training)](https://github.com/Yan98/Event-Camera-Data-Pre-training) |
| `MEM` | [Masked Event Modeling: Self-Supervised Pretraining for Event Cameras](https://openaccess.thecvf.com/content/WACV2024/html/Klenk_Masked_Event_Modeling_Self-Supervised_Pretraining_for_Event_Cameras_WACV_2024_paper.html) | WACV 2024 | [![GitHub](https://img.shields.io/github/stars/tum-vision/mem)](https://github.com/tum-vision/mem) |
| `DMM` | [Data-efficient event camera pre-training via disentangled masked modeling](https://arxiv.org/abs/2403.00416) | arXiv 2024 | []() |
| `STP` | [Enhancing Event Camera Data Pretraining via Prompt-Tuning with Visual Models](https://openreview.net/pdf?id=XTBdPLhiRL) | - | []() |
| `ECDDP` | [Event Camera Data Dense Pre-training](https://link.springer.com/chapter/10.1007/978-3-031-72775-7_17) | ECCV2024 | [![GitHub](https://img.shields.io/github/stars/Yan98/Event-Camera-Data-Dense-Pre-training)](https://github.com/Yan98/Event-Camera-Data-Dense-Pre-training/) |
| `EventBind` | [Eventbind: Learning a unified representation to bind them all for event-based open-world understanding](https://link.springer.com/chapter/10.1007/978-3-031-72897-6_27) | ECCV2024 | [![GitHub](https://img.shields.io/github/stars/jiazhou-garland/EventBind)](https://github.com/jiazhou-garland/EventBind) |
| `EventFly` | [EventFly: Event Camera Perception from Ground to the Sky](https://openaccess.thecvf.com/content/CVPR2025/html/Kong_EventFly_Event_Camera_Perception_from_Ground_to_the_Sky_CVPR_2025_paper.html) | CVPR 2025 | []() |

-----

# 4. Open-World Perception and Planning

### Text-Grounded Understanding

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `CLIP2Scene` | [Towards Label-efficient 3D Scene Understanding by CLIP](https://arxiv.org/abs/2301.04926) | CVPR 2023 | [![GitHub](https://img.shields.io/github/stars/runnanchen/CLIP2Scene)](https://github.com/runnanchen/CLIP2Scene) |
| `OpenScene` | [3D Scene Understanding with Open Vocabularies](https://arxiv.org/abs/2211.15654) | CVPR 2023 | [![GitHub](https://img.shields.io/github/stars/pengsongyou/openscene)](https://github.com/pengsongyou/openscene) |
| `CLIP-ZSPCS` | [Transferring CLIP's Knowledge into Zero-Shot Point Cloud Semantic Segmentation](https://arxiv.org/pdf/2312.07221) | MM 2023 | []() |
| `CLIP-FO3D` | [Learning Free Open-world 3D Scene Representations from 2D Dense CLIP](https://arxiv.org/abs/2303.04748) | ICCVW 2023 | [](https://www.google.com/search?q=) |
| `POP-3D` | [Open-Vocabulary 3D Occupancy Prediction from Images](https://arxiv.org/abs/2401.09413) | NeurIPS 2023 | [![GitHub](https://img.shields.io/github/stars/vobecant/POP3D)](https://github.com/vobecant/POP3D) |
| `VLM2Scene` | [Self-Supervised Image-Text-LiDAR Learning with Foundation Models](https://ojs.aaai.org/index.php/AAAI/article/view/28121) | AAAI 2024 | [![GitHub](https://img.shields.io/github/stars/gbliao/VLM2Scene)](https://github.com/gbliao/VLM2Scene) |
| `IntraCorr3D` | [Hierarchical intra-modal correlation learning for label-free 3d semantic segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Kang_Hierarchical_Intra-modal_Correlation_Learning_for_Label-free_3D_Semantic_Segmentation_CVPR_2024_paper.html) | CVPR 2024 | []() |
| `SAL` | [Better Call SAL: Towards Learning to Segment Anything in Lidar](https://arxiv.org/abs/2403.13129) | ECCV 2024 | [![GitHub](https://img.shields.io/github/stars/nv-dvl/segment-anything-lidar)](https://github.com/nv-dvl/segment-anything-lidar) |
| `Affinity3D` | [Propagating Instance-Level Semantic Affinity for Zero-Shot Semantic Seg](https://dl.acm.org/doi/10.1145/3664647.3680651) | ACM MM 2024 | [](https://www.google.com/search?q=) |
| `UOV` | [3D Unsupervised Learning by Distilling 2D Open-Vocabulary Segmentation Models for Autonomous Driving](https://arxiv.org/html/2405.15286v1) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/sbysbysbys/UOV)](https://github.com/sbysbysbys/UOV) |
| `OVO` | [OVO: Open-Vocabulary Occupancy](https://arxiv.org/abs/2305.16133) | arXiv 2023 | [![GitHub](https://img.shields.io/github/stars/dzcgaara/OVO-Open-Vocabulary-Occupancy)](https://github.com/dzcgaara/OVO-Open-Vocabulary-Occupancy) |
| `LangOcc` | [Langocc: Self-supervised open vocabulary occupancy estimation via volume rendering](https://arxiv.org/abs/2407.17310) | 3DV 2025 | [![GitHub](https://img.shields.io/github/stars/boschresearch/LangOcc)](https://github.com/boschresearch/LangOcc) |
| `VEON` | [VEON: Vocabulary-Enhanced Occupancy Prediction](https://link.springer.com/chapter/10.1007/978-3-031-72949-2_6) | ECCV 2024 | []() |
| `LOcc` | [Language Driven Occupancy Prediction](https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Language_Driven_Occupancy_Prediction_ICCV_2025_paper.html) | ICCV 2025 | [![GitHub](https://img.shields.io/github/stars/pkqbajng/locc)](https://github.com/pkqbajng/locc) |
| `UP-VL` | [Unsupervised 3D Perception with 2D Vision-Language Distillation for Autonomous Driving](https://openaccess.thecvf.com/content/ICCV2023/html/Najibi_Unsupervised_3D_Perception_with_2D_Vision-Language_Distillation_for_Autonomous_Driving_ICCV_2023_paper.html) | ICCV 2023 | []() |
| `ZPCS-MM` | [See more and know more: Zero-shot point cloud segmentation via multi-modal visual data](https://openaccess.thecvf.com/content/ICCV2023/html/Lu_See_More_and_Know_More_Zero-shot_Point_Cloud_Segmentation_via_ICCV_2023_paper.html) | ICCV 2023 | []() |
| `CNS` | [Towards label-free scene understanding by vision foundation models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ef6c94e9cf4d169298479ee2e230ee13-Abstract-Conference.html) | NeurIPS | [![GitHub](https://img.shields.io/github/stars/runnanchen/Label-Free-Scene-Understanding)](https://github.com/runnanchen/Label-Free-Scene-Understanding) |
| `3DOV-VLD` | [3D Open-Vocabulary Panoptic Segmentation with 2D-3D Vision-Language Distillation](https://link.springer.com/chapter/10.1007/978-3-031-73661-2_2) | ECCV 2024 | []() |
| `CLIP^2` | [CLIP2: Contrastive Language-Image-Point Pretraining from Real-World Point Cloud Data](https://arxiv.org/abs/2303.12417) | CVPR 2023 | []() |
| `AdaCo` | [Adaco: Overcoming visual foundation model noise in 3d semantic segmentation via adaptive label correction]() | AAAI 2025 | []() |
| `TT-Occ` | [TT-Occ: Test-Time Compute for Self-Supervised Occupancy via Spatio-Temporal Gaussian Splatting](https://arxiv.org/abs/2503.08485) | arXiv 2025 | []() |
| `AutoOcc` | [AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting](https://arxiv.org/abs/2502.04981) | ICCV 2025 | []() |



### Unified World Representation for Action

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `OccWorld` | [OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](https://link.springer.com/chapter/10.1007/978-3-031-72624-8_4) | ECCV 2024 | [![GitHub](https://img.shields.io/github/stars/wzzheng/OccWorld)](https://github.com/wzzheng/OccWorld) |
| `GenAD` | [Generalized Predictive Model for Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Generalized_Predictive_Model_for_Autonomous_Driving_CVPR_2024_paper.html) | CVPR 2024 | [![GitHub](https://img.shields.io/github/stars/OpenDriveLab/DriveAGI)](https://github.com/OpenDriveLab/DriveAGI) |
| `OccSora` | [OccSora: 4D Occupancy Generation Models as World Simulators for Autonomous Driving](https://arxiv.org/abs/2405.20337) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/wzzheng/OccSora)](https://github.com/wzzheng/OccSora) |
| `OccLLaMA` | [OccLLaMA: An Occupancy-Language-Action Generative World Model for Autonomous Driving](https://arxiv.org/abs/2409.03272) | arXiv 2024 | []() |
| `OccVAR` | [OccVAR: Scalable 4D Occupancy Prediction via Next-Scale Prediction](https://openreview.net/forum?id=X2HnTFsFm8) | - | []() |
| `RenderWorld` | [Renderworld: World Model with Self-Supervised 3D Label](https://ieeexplore.ieee.org/abstract/document/11127609) | ICRA 2025 | []() |
| `Drive-OccWorld` | [Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving](https://ojs.aaai.org/index.php/AAAI/article/view/33010) | AAAI 2025 | [![GitHub](https://img.shields.io/github/stars/yuyang-cloud/Drive-OccWorld)](https://github.com/yuyang-cloud/Drive-OccWorld) |
| `LAW` | [Enhancing End-to-End Autonomous Driving with Latent World Model](https://arxiv.org/abs/2406.08481) | ICLR 2025 | [![GitHub](https://img.shields.io/github/stars/BraveGroup/LAW)](https://github.com/BraveGroup/LAW) |
| `FSF-Net` | [FSF-Net: Enhance 4D occupancy forecasting with coarse BEV scene flow for autonomous driving](https://www.sciencedirect.com/science/article/abs/pii/S0031320325010337) | PR 2025 | []() |
| `DriveX` | [DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving](https://arxiv.org/abs/2505.19239) | arXiv 2025 | []() |
| `SPOT` | [SPOT: Scalable 3D Pre-training via Occupancy Prediction for Autonomous Driving](https://arxiv.org/abs/2309.10527) | TPAMI 2025 | [![GitHub](https://img.shields.io/github/stars/PJLab-ADG/3DTrans)](https://github.com/PJLab-ADG/3DTrans) |
| `WoTE` | [End-to-End Driving with Online Trajectory Evaluation via BEV World Model](https://arxiv.org/pdf/2504.01941) | ICCV 2025 | [![GitHub](https://img.shields.io/github/stars/liyingyanUCAS/WoTE)](https://github.com/liyingyanUCAS/WoTE) |
| `FASTopoWM` | [FASTopoWM: Fast-Slow Lane Segment Topology Reasoning with Latent World Models](https://arxiv.org/abs/2507.23325) | arXiv 2025 | [![GitHub](https://img.shields.io/github/stars/YimingYang23/FASTopoWM)](https://github.com/YimingYang23/FASTopoWM) |
| `OccTens` | [OccTENS: 3D Occupancy World Model via Temporal Next-Scale Prediction](https://arxiv.org/abs/2509.03887) | arXiv 2025 | []() |
| `OccVLA` | [Occvla: Vision-language-action model with implicit 3d occupancy supervision](https://arxiv.org/abs/2509.05578) | arXIv 2025 | []() |
| `World4Drive` | [World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_World4Drive_End-to-End_Autonomous_Driving_via_Intention-aware_Physical_Latent_World_Model_ICCV_2025_paper.html) | ICCV 2025 | [![GitHub](https://img.shields.io/github/stars/ucaszyp/World4Drive)](https://github.com/ucaszyp/World4Drive) |

-----

# 5. Acknowledgements

We thank the authors of the referenced papers for their open-source contributions. 
