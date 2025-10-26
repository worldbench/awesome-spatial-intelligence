# Data Representation Learning from Onboard Sensors: A Survey
## Motivation

总结现有无人系统上装载传感器上（包含相机、激光雷达、毫米波雷达等）采集数据的表示学习，包含无人车、无人机、机械狗、轨道交通等不同平台，重点关注不同传感器交互条件下的预训练，以及跨平台、跨域的泛化问题。

[Overleaf](https://www.overleaf.com/8862367956qdzhgwmrgfxr#45fb58)(待整理)

## 时间
- 8.22 制定计划 确定相关任务分工
- 9.15 完成三部分不同平台相关数据集、方法整理，确定整体框架
- 10.15 完成数据集、传感器配置部分撰写，和相关方法收集
- 11.15 完成预训练方法部分撰写和整体初稿
- 12.15 细化完善文献，投稿至相关期刊

## 各部分内容安排
### Introduction
- 说明研究背景和motivation，强调表示学习的重要性和中各种无人系统中的应用，突出多传感器交互预训练的潜力和挑战
- 明确本文的研究目标和scope，确定关注的重点
- 说明本文相较于之前综述的不同和贡献

### Background
侧重于背景介绍
- 说明表示学习的概念和应用
- 简要说明不同无人系统的传感器配置、数据特点和关注任务


### Fundamental Techniques for Pre-training on Sensor Data
侧重于具体技术
- 介绍自监督学习 迁移学习 多模态学习等通用预训练领域的相关技术


### Platform-specific Datasets from Sensors
主要介绍数据集和传感器配置
- 介绍不同平台的传感器、数据集、相关预训练方法
- 分为无人车、无人机、其他平台（如机械狗、轨道交通）三个部分，整理数据集和方法的相关表格
#### Autonomous Vehicles
可参考的现有相关综述：

[SCIENTIA SINICA Informationis][Open-sourced Data Ecosystem in Autonomous Driving: the Present and Future](https://arxiv.org/abs/2312.03408)

[Forging Vision Foundation Models for Autonomous Driving: Challenges, Methodologies, and Opportunities](https://arxiv.org/abs/2401.08045)


#### Drones
可参考的现有相关文章和数据集：

[CVPR 2024][Multiview Aerial Visual Recognition (MAVREC): Can Multi-view Improve Aerial Visual Perception?](https://mavrec.github.io/)

[CVPR 2024 Workshop][DDOS: The Drone Depth and Obstacle Segmentation Dataset](https://openreview.net/forum?id=FZxofmVOwg)

[ACM MM 2023 Workshop][UAVs in Multimedia: Capturing the World from a New Perspective](https://dl.acm.org/doi/proceedings/10.1145/3607834): Cross-view Drone-based Geo-localization/Drone-based Object Detection / Scene Understanding [2024](https://www.zdzheng.xyz/ACMMM2024Workshop-UAV)

[Self-Supervised Pretraining and Controlled Augmentation Improve Rare Wildlife Recognition in UAV Images](https://openaccess.thecvf.com/content/ICCV2021W/LUAI/papers/Zheng_Self-Supervised_Pretraining_and_Controlled_Augmentation_Improve_Rare_Wildlife_Recognition_in_ICCVW_2021_paper.pdf)

[Drone Dataset Summary](https://github.com/agentmorris/agentmorrispublic/blob/main/drone-datasets.md)

[VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)

[SensatUrban](https://github.com/QingyongHu/SensatUrban)

[Stanford Drone Dataset](https://techfinder.stanford.edu/technology/stanford-drone-dataset-multi-scale-multi-target-social-navigation)

[Anti-UAV](https://github.com/ucas-vg/Anti-UAV)

Related Survey:

[Deep learning-based object detection in low-altitude UAV datasets: A survey](https://www.sciencedirect.com/science/article/pii/S0262885620301785)

[Vehicle detection from UAV imagery with deep learning: A review](https://ieeexplore.ieee.org/abstract/document/9439930/?casa_token=4nWKkY6NUF8AAAAA:LSbBV_XgcmERez4ebltmUCZsSbseuBp4fa0D4Un2OnI5DcFpMHtqWnJnOpovj2Ebxv_V_QQaJnwhgA)

[A survey of deep learning techniques for vehicle detection from UAV images](https://www.sciencedirect.com/science/article/pii/S1383762121001107)



#### Other Platforms
##### Robotic Dogs
可参考的现有相关文章和数据集：

[NMI 2024][Lifelike Agility and Play in Quadrupedal Robots using Reinforcement Learning and Generative Pre-trained Models](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play)

##### Rail Transportation
可参考的现有相关文章和数据集：

[ACM MM 2022][Rail Detection: An Efficient Row-based Network and a New Benchmark](https://github.com/Sampson-Lee/Rail-Detection)

##### Unmanned Surface Vehicle
可参考的现有相关文章和数据集：

[NanoMVG: USV-Centric Low-Power Multi-Task Visual Grounding based on Prompt-Guided Camera and 4D mmWave Radar](https://arxiv.org/pdf/2408.17207)

### Multi-Modality Interaction for Pre-Training
主要介绍方法
- 介绍传感器交互、融合相关的预训练
- 介绍多任务学习在传感器数据集上的应用
- 介绍跨平台和跨域的预训练：如何使用通用场景的基础模型泛化到不同平台

根据涉及模态分类：
- 单一模态的预训练
  - 纯图像
  - 纯点云
- 涉及不同模态交互的预训练
  - 点云辅助图像预训练：增强图像感知几何的能力
  - 图像辅助点云预训练：增强点云感知语义的能力
  - 点云图像联合的预训练
  - 更多模态的预训练技术：引入文本信息，获得开放世界的感知能力

根据代理任务分类（分类暂时参考了[Forge_VFM4AD](https://github.com/zhanghm1995/Forge_VFM4AD)：
- 基于对比学习（including with text，构建正负特征样本对）
- 基于知识蒸馏（使用现有的Image基础模型蒸馏给LiDAR为主）
- 基于重建（Mask Image Modeling/类MAE）
- 基于渲染（输出渲染回2D相机平面including RGB、Depth、Semantics）
- 基于世界模型建模（时序Forecasting，使用现有帧预测未来帧）五类

### Challenges and Future Directions
- 数据规模和多样性
- 实时性和计算资源
- 安全性和隐私
- 相关潜在的研究方向

### Conclusion
总结全文的发现和结论


## 相关综述
需要在文章中说明和他们的不同

[Forging Vision Foundation Models for Autonomous Driving: Challenges, Methodologies, and Opportunities](https://arxiv.org/abs/2401.08045)

[Self-Supervised Multimodal Learning: A Survey](https://arxiv.org/abs/2304.01008)

[Unsupervised Representation Learning for Point Clouds with Deep Neural Networks: A Survey](https://arxiv.org/abs/2202.13589)

[Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond](https://arxiv.org/abs/2405.03520)

