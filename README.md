<div align=center>

# Forging Spatial Intelligence
### A Survey on Multi-Modal Pre-Training for Autonomous Systems

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
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
* 🌍 **Open-World Perception and Planning** *The Frontier of Embodied Autonomy.* Represents the evolution from passive perception to active decision-making. This paradigm encompasses **Generative World Models** (e.g., video/occupancy generation), **Embodied Vision-Language-Action (VLA)** models, and systems capable of **Open-World** reasoning.

📄 **[Paper Link]()**


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
  - [**4. Open-World Perception and Planning**](#4-open-world-perception-and-planning)
      - [Text-Grounded Understanding](#text-grounded-understanding)
      - [Unified World Representation for Action](#unified-world-representation-for-action)
  - [**5. Acknowledgements**](#5-acknowledgements)

-----

# 1. Benchmarks & Datasets

### Vehicle-Based Datasets


| Dataset | Venue | Sensor | Task | Download |
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

| Dataset | Venue | Sensor | Task | Download |
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
| `SegContrast` | [3D Point Cloud Feature Representation Learning through Self-supervised Segment Discrimination](https://ieeexplore.ieee.org/document/9681336)| RA-L 2021 | [![GitHub](https://img.shields.io/github/stars/PRBonn/segcontrast)](https://github.com/PRBonn/segcontrast) |
| `ProposalContrast` | [Unsupervised Pre-training for LiDAR-Based 3D Object Detection](https://arxiv.org/abs/2207.12654) | ECCV 2022 | [![GitHub](https://img.shields.io/github/stars/yinjunbo/ProposalContrast)](https://github.com/yinjunbo/ProposalContrast) |
| `Occupancy-MAE` | [Self-supervised Pre-training Large-scale LiDAR Point Clouds with Masked Occupancy Autoencoders](https://arxiv.org/abs/2206.09900) | T-IV 2023 | [![GitHub](https://img.shields.io/github/stars/chaytonmin/Occupancy-MAE)](https://github.com/chaytonmin/Occupancy-MAE) |
| `ALSO` |[Automotive LiDAR Self-supervision by Occupancy Estimation](https://arxiv.org/abs/2212.05867) | CVPR 2023 | [![GitHub](https://img.shields.io/github/stars/valeoai/ALSO)](https://github.com/valeoai/ALSO) |
| `GD-MAE` | [Generative Decoder for MAE Pre-training on LiDAR Point Clouds](https://arxiv.org/abs/2212.03010) | CVPR 2023 | [![GitHub](https://img.shields.io/github/stars/Nightmare-n/GD-MAE)](https://github.com/Nightmare-n/GD-MAE) |
| `AD-PT` | [Autonomous Driving Pre-Training with Large-scale Point Cloud Dataset](https://arxiv.org/abs/2306.00612) | NeurIPS 2023 | [![GitHub](https://img.shields.io/github/stars/PJLab-ADG/3DTrans)](https://github.com/PJLab-ADG/3DTrans) |
| `PatchContrast` | [Self-Supervised Pre-training for 3D Object Detection](https://arxiv.org/abs/2308.06985) | arXiv 2023 | [](https://www.google.com/search?q=) |
| `MAELi` | [Masked Autoencoder for Large-Scale LiDAR Point Clouds](https://arxiv.org/abs/2212.07207) | WACV 2024 | [](https://www.google.com/search?q=) |
| `BEV-MAE` | [Bird's Eye View Masked Autoencoders for Point Cloud Pre-training](https://arxiv.org/abs/2212.05758) | AAAI 2024 | [![GitHub](https://img.shields.io/github/stars/VDIGPKU/BEV-MAE)](https://github.com/VDIGPKU/BEV-MAE) |
| `UnO` | [Unsupervised Occupancy Fields for Perception and Forecasting](https://arxiv.org/abs/2406.08691) | CVPR 2024 | [](https://www.google.com/search?q=) |
| `BEVContrast` | [Self-Supervision in BEV Space for Automotive Lidar Point Clouds](https://arxiv.org/abs/2310.17281) | 3DV 2024 | [![GitHub](https://img.shields.io/github/stars/valeoai/BEVContrast)](https://github.com/valeoai/BEVContrast) |
| `Copilot4D` | [Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion](https://arxiv.org/abs/2311.01017) | ICLR 2024 | [](https://www.google.com/search?q=) |
| `T-MAE` | [Temporal Masked Autoencoders for Point Cloud Representation Learning](https://arxiv.org/abs/2312.10217) | ECCV 2024 | [![GitHub](https://img.shields.io/github/stars/codename1995/t-mae)](https://github.com/codename1995/t-mae) |
| `PICTURE` | [Point Cloud Reconstruction Is Insufficient to Learn 3D Representations](https://dl.acm.org/doi/10.1145/3664647.3680890) | ACM MM 2024 | [](https://www.google.com/search?q=) |
| `LSV-MAE` | [Rethinking Masked-Autoencoder-Based 3D Point Cloud Pretraining](https://ieeexplore.ieee.org/document/10588770) | IV 2024 | [](https://www.google.com/search?q=) |
| `UNIT` | [Unsupervised Online Instance Segmentation through Time](https://arxiv.org/abs/2409.07887) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/valeoai/UNIT)](https://github.com/valeoai/UNIT) |
| `R-MAE` | [Sense Less, Generate More: Pre-training LiDAR Perception with Masked Autoencoders](https://arxiv.org/abs/2406.07833) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/sinatayebati/R-MAE)](https://github.com/sinatayebati/R-MAE) |

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
| `ST-SLidR` | [Self-Supervised Image-to-Point Distillation via Semantically Tolerant Contrastive Loss](https://arxiv.org/abs/2301.05709) | CVPR 2023 | [](https://www.google.com/search?q=) |
| `I2P-MAE` | [Learning 3D Representations from 2D Pre-trained Models via Image-to-Point MAE](https://arxiv.org/abs/2212.06785) | CVPR 2023 | [![GitHub](https://img.shields.io/github/stars/ZrrSkywalker/I2P-MAE)](https://github.com/ZrrSkywalker/I2P-MAE) |
| `TriCC` | [Unsupervised 3D Point Cloud Representation Learning by Triangle Constrained Contrast](https://ieeexplore.ieee.org/document/10203592) | CVPR 2023 | [](https://www.google.com/search?q=) |
| `Seal` | [Segment Any Point Cloud Sequences by Distilling Vision FMs](https://arxiv.org/abs/2306.09347) | NeurIPS 23 | [![GitHub](https://img.shields.io/github/stars/youquanl/Segment-Any-Point-Cloud)](https://github.com/youquanl/Segment-Any-Point-Cloud) |
| `PRED` | [Pre-training via Semantic Rendering on LiDAR Point Clouds](https://arxiv.org/abs/2311.04501) | NeurIPS 23 | [](https://www.google.com/search?q=) |
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

-----

# 4. Open-World Perception and Planning

### Text-Grounded Understanding

| Model | Paper | Venue | GitHub |
|:-:|:-|:-:|:-:|
| `CLIP2Scene` | [Towards Label-efficient 3D Scene Understanding by CLIP](https://arxiv.org/abs/2301.04926) | CVPR 2023 | [![GitHub](https://img.shields.io/github/stars/runnanchen/CLIP2Scene)](https://github.com/runnanchen/CLIP2Scene) |
| `OpenScene` | [3D Scene Understanding with Open Vocabularies](https://arxiv.org/abs/2211.15654) | CVPR 2023 | [![GitHub](https://img.shields.io/github/stars/pengsongyou/openscene)](https://github.com/pengsongyou/openscene) |
| `CLIP-FO3D` | [Learning Free Open-world 3D Scene Representations from 2D Dense CLIP](https://arxiv.org/abs/2303.04748) | ICCVW 2023 | [](https://www.google.com/search?q=) |
| `POP-3D` | [Open-Vocabulary 3D Occupancy Prediction from Images](https://arxiv.org/abs/2401.09413) | NeurIPS 2023 | [![GitHub](https://img.shields.io/github/stars/vobecant/POP3D)](https://github.com/vobecant/POP3D) |
| `VLM2Scene` | [Self-Supervised Image-Text-LiDAR Learning with Foundation Models](https://ojs.aaai.org/index.php/AAAI/article/view/28121) | AAAI 2024 | [![GitHub](https://img.shields.io/github/stars/gbliao/VLM2Scene)](https://github.com/gbliao/VLM2Scene) |
| `SAL` | [Better Call SAL: Towards Learning to Segment Anything in Lidar](https://arxiv.org/abs/2403.13129) | ECCV 2024 | [![GitHub](https://img.shields.io/github/stars/nv-dvl/segment-anything-lidar)](https://github.com/nv-dvl/segment-anything-lidar) |
| `Affinity3D` | [Propagating Instance-Level Semantic Affinity for Zero-Shot Semantic Seg](https://dl.acm.org/doi/10.1145/3664647.3680651) | ACM MM 2024 | [](https://www.google.com/search?q=) |
| `UOV` | [3D Unsupervised Learning by Distilling 2D Open-Vocabulary Segmentation Models for Autonomous Driving](https://arxiv.org/html/2405.15286v1) | arXiv 2024 | [![GitHub](https://img.shields.io/github/stars/sbysbysbys/UOV)](https://github.com/sbysbysbys/UOV) |


### Unified World Representation for Action

-----

# 5. Acknowledgements

We thank the authors of the referenced papers for their open-source contributions. 
