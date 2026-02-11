# 读过论文的一些总结

记录日常读过的论文，按照方向与年份整理，方便快速翻阅与检索。
当前笔记涵盖语音关键词检测、语音识别、自监督与无监督域自适应等方向的 300+ 篇文章，方便团队成员快速定位灵感或复习经典。

## 目录
- [语音与关键词检测](#语音与关键词检测)
- [无监督域自适应与相关工作](#无监督域自适应与相关工作)

## 语音与关键词检测

> 主要是读的关键词检测文章，其中有少部分 ASR 文章。

<details open>
<summary>按年份整理的语音/关键词论文（2014-2022）</summary>

- **2014**
  - 2014 Small-footprint keyword spotting using deep neural networks - 最早的神经网络做小参数量关键词，softmax
- **2015**
  - 2015 A time delay neural network architecture for efficient modeling of long temporal contexts - dilation, TDNN，利用上下文信息
  - 2015 LAS-Listen, Attend and Spell - 端到端基于 attention 的方法
- **2016**
  - 2016 An End-to-End Architecture for Keyword Spotting and Voice Activity Detection - CRNN 和 CTC 做的关键词
  - 2016 Multi-task learning and Weighted Cross-entropy for DNN-based Keyword Spotting - 大词汇量迁移以及权重交叉熵
- **2017**
  - 2017 Attention Is All You Need - transformer 的论文
  - 2017 Compressed time delay neural network for small-footprint keyword spotting - TDNN 做关键词
  - 2017 End-to-end Keywords Spotting Based on Connectionist Temporal Classification for Mandarin - ASR+CTC 做关键词
  - 2017 Hello Edge: Keyword Spotting on Microcontrollers - 网络参数的计算、内存以及性能
  - 2017 JOINT CTC-ATTENTION BASED END-TO-END SPEECH RECOGNITION USING MULTI-TASK LEARNING - CTC 和 attention 联合训练
  - 2017 MAX-POOLING LOSS TRAINING OF LONG SHORT-TERM MEMORY NETWORKS FOR SMALL-FOOTPRINT KEYWORD SPOTTING - max pooling 做关键词
- **2018**
  - 2018 Attention-based End-to-End Models for Small-Footprint Keyword Spotting - 基于 attention 的关键词
  - 2018 Compact Feedforward Sequential Memory Networks for Small-footprint Keyword Spotting - 多帧预测
  - 2018 CosFace: Large Margin Cosine Loss for Deep Face Recognition - softmax 的几种变形
  - 2018 DEEP RESIDUAL LEARNING FOR SMALL-FOOTPRINT KEYWORD SPOTTING - resnet 和扩张卷积做关键词
  - 2018 DONUT: CTC-based Query-by-Example Keyword Spotting - CTC 做自定义关键词，先识别再匹配
  - 2018 Efficient keyword spotting using time delay neural networks - TDNN 并加入跳连接
  - 2018 Robust Classification with Convolutional Prototype Learning - 原型网络
  - 2018 Stochastic Adaptive Neural Architecture Search for Keyword Spotting - shortcut 决定网络结构
  - 2018 TCN--An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling - TCN
- **2019**
  - 2019 A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling - 五种声音事件检测的池化损失
  - 2019 A time delay neural network with shared weight self-attention for small-footprint keyword spotting - 单矩阵自注意力
  - 2019 Convolutional Neural Networks for Small-footprint Keyword Spotting - CNN 做关键词，讨论参数压缩
  - 2019 EFFICIENT KEYWORD SPOTTING USING DILATED CONVOLUTIONS AND GATING - WaveNet 风格的扩张卷积
  - 2019 FOCAL LOSS AND DOUBLE-EDGE-TRIGGERED DETECTOR FOR ROBUST SMALL-FOOTPRINT KEYWORD SPOTTING - focal loss 以及重复触发
  - 2019 Selective Kernel Networks - 多核特征加权
  - 2019 SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition - 类似图像随机 mask
  - 2019 Temporal Convolution for Real-time Keyword Spotting on Mobile Devices (TC-ResNet) - 一维卷积同时考虑低高频
  - 2019 THE SPEECHTRANSFORMER FOR LARGE-SCALE MANDARIN CHINESE SPEECH RECOGNITION - 帧率、采样、focal loss 的优化
  - 2019 Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context - 感受野变长
- **2020**
  - 2020 A depthwise separable convolutional neural network for keyword spotting on an embedded system - 深度可分离卷积
  - 2020 ADAPTATION OF RNN TRANSDUCER WITH TEXT-TO-SPEECH TECHNOLOGY FOR KEYWORD SPOTTING - text2speech 增加样本
  - 2020 AdderNet: Do We Really Need Multiplications in Deep Learning? - 卷积乘法替换为加法
  - 2020 CRNN-CTC BASED MANDARIN KEYWORDS SPOTTING - CRNN 与 CTC 在中文上的应用
  - 2020 Depthwise separable convolutional ResNet with squeeze-and-excitation blocks for small-footprint keyword spotting - DS-ResNet 结合 SE
  - 2020 Domain Aware Training for Far-field Small-footprint Keyword Spotting - 远场域自适应
  - 2020 Federated Self-Supervised Learning of Multi-Sensor Representations for Embedded Intelligence - 对比损失的二分类
  - 2020 Keyword retrieving in continuous speech using connectionist temporal classification - 使用 CTC 的关键词检索
  - 2020 MINING EFFECTIVE NEGATIVE TRAINING SAMPLES FOR KEYWORD SPOTTING - 负样本选择缓解不平衡
  - 2020 Multi-scale Convolution for Robust Keyword Spotting - 深度可分离卷积先降维再升维
  - 2020 QUARTZNET: DEEP AUTOMATIC SPEECH RECOGNITION WITH 1D TIME-CHANNEL SEPARABLE CONVOLUTIONS - 深度可分离卷积做语音识别
  - 2020 Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets - 模块顺序调整
  - 2020 Re-weighted Interval Loss for Handling Data Imbalance Problem of End-to-End Keyword Spotting - 类权重与样本权重
  - 2020 SMALL-FOOTPRINT KEYWORD SPOTTING ON RAW AUDIO DATA WITH SINC-CONVOLUTIONS - 使用 sinc 卷积抽特征
  - 2020 Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition - WeNet 初始版本
- **2021**
  - 2021 A Novel Re-weighted CTC Loss for Data Imbalance in Speech Keyword Spotting - 基于 FAR、FRR 的 CTC 权重
  - 2021 A Streaming End-to-End Framework For Spoken Language Understanding - 语义理解框架
  - 2021 ATTENTION IS ALL YOU NEED IN SPEECH SEPARATION - transformer 做语音分离
  - 2021 AUC OPTIMIZATION FOR ROBUST SMALL-FOOTPRINT KEYWORD SPOTTING WITH LIMITED TRAINING DATA - 多维 AUC
  - 2021 Broadcasted residual learning for efficient keyword spotting - 频率维度压缩再扩展
  - 2021 deformable TDNN with adaptive receptive fields for speech recognition - 自适应感受野 TDNN
  - 2021 End-to-End Transformer-Based Open-Vocabulary Keyword Spotting with Location-Guided Local Attention - attention 估计关键词起始位置
  - 2021 Energy-friendly keyword spotting system using add-based convolution - 卷积乘法替换为加法
  - 2021 Few-Shot Keyword Spotting in Any Language Mark - 大模型微调解决 few-shot
  - 2021 Keyword transformer: A self-attention model for keyword spotting - transformer 做关键词
  - 2021 Learning Efficient Representations for Keyword Spotting with Triplet Loss - triplet 损失
  - 2021 LIGHTWEIGHT DYNAMIC FILTER FOR KEYWORD SPOTTING - 动态卷积
  - 2021 Metric Learning for Keyword Spotting - 类中心
  - 2021 Noisy student-teacher training for robust keyword spotting - 老师学生网络
  - 2021 RECENT DEVELOPMENTS ON ESPNET TOOLKIT BOOSTED BY CONFORMER - ESPnet 更新
  - 2021 Text Anchor Based Metric Learning for Small-footprint Keyword Spotting - triplet 中的中心替换
  - 2021 U2++: Unified Two-pass Bidirectional End-to-end Model for Speech Recognition - 双向注意力 mask
  - 2021 visual Keyword Spotting with Attention - 多模态视频与关键词
  - 2021 TRANSFORMER-BASED END-TO-END SPEECH RECOGNITION WITH LOCAL DENSE SYNTHESIZER ATTENTION - 局部注意力
  - 2021 LAC-Efficient conformer-based speech recognition with linear attention - 线性 attention
  - 2021 rotary- Conformer-based End-to-end Speech Recognition With Rotary Position Embedding - 旋转位置编码
- **2022**
  - 2022 CONVMIXER: FEATURE INTERACTIVE CONVOLUTION WITH CURRICULUM LEARNING FOR SMALL FOOTPRINT AND NOISY FAR-FIELD KEYWORD SPOTTING - 渐进训练
  - 2022 END-TO-END LOWRESOURCE KEYWORD SPOTTING THROUGH CHARACTER RECOGNITION AND BEAM-SEARCH RE-SCORING - LibriSpeech 预训练辅助 GSC
  - 2022 Few-Shot Keyword Spotting With Prototypical Networks - 类中心
  - 2022 IMPROVING FEATURE GENERALIZABILITY WITH MULTITASK LEARNING IN CLASS INCREMENTAL LEARNING - 多任务提升泛化
  - 2022 Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection - 自监督帮助视频伪造检测
  - 2022 PROGRESSIVE CONTINUAL LEARNING FOR SPOKEN KEYWORD SPOTTING - 连续学习
  - 2022 Two-stage streaming keyword detection and localization with multi-scale depthwise temporal convolution (MDTC) - 多尺度深度可分离卷积
  - 2022 Understanding Audio Features via Trainable Basis Functions Kwan - 时频谱可学习基函数
  - 2022 UNIFIED SPECULATION, DETECTION, AND VERIFICATION KEYWORD SPOTTING - 利用 VAD 的延时可控 max pooling
</details>

## 无监督域自适应与相关工作

> 这个总结按年份与核心思想进行划分，涵盖综述、伪标签、特征对齐、对抗学习、分类器设计、前沿方向、传统方法、任务场景以及相关但非 UDA 的工作。  
> 大致分为：综述、伪标签、基于特征的、对抗的、分类器的、较为前沿的、传统的、任务，以及一些非 UDA 的文章。

### 综述
- 2006 MMD--A Kernel Method for the Two-Sample-Problem Arthur--MMD
- 2009 A survey on transfer learing--- ----经典综述，阐述了与自适应的定义、分类以及方法
- 2010 Relative Clustering Validity Criteria: A Comparative Overview---关于聚类的一篇综述
- 2019 A review of domain adaptation without target label
- 2019 Transfer Adaptation Learning: A Decade Survey
- 2020 A Decade Survey of Transfer Learning (2010–2020)
- 2020 A comprehensive survey on transfer learning ---1328
- 2020 A survey of Unsupervised deep domain adaptation
- 2020 A survey of unsupervised deep domain adaptation--TIST--308
- 2020 Deep visual domain adaptation: a survey---深度域自适应的重要综述，2018 到现在引用量超千
- 2020 Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey ----视觉自监督综述
- 2020 a survey of transformers---关于 transformer 的综述
- 2021 Deep Spoken Keyword Spotting: An Overview---深度关键词综述

### 伪标签
- 2016 RTN---unsupervised domain adaptation with residual transfer network--两个分类器，相差一个扰动函数
- 2018 Unsupervised domain adaptation for semantic segmentation via class-balanced self-train--ECCV-728---生成伪标签类不平衡--代码：https://github.com/yzou2/CBST?utm_source=catalyzex.com
- 2019 progressive feature alignment for unsupervised domain adaptation--easy 2 hard
- 2021 Domain Adaptation with Auxiliary Target Domain-Oriented Classifier---特殊目标域分类器帮助伪标签 (https://github.com/02Bigboy/read_paper/assets/74250589/d010204a-ba72-40b9-b92f-00d2c6ce9b48)

### 基于特征的
- 2006 Neural Structural Correspondence Learning for Domain Adaptation--自动编码器
- 2013 Connecting the Dots with Landmarks: Discriminatively Learning Domain-Invariant Features for Unsupervised Domain Adaptation---landmarks--相似的源域样本加到目标域
- 2013 Dlid: Deep learning for domain adaptation by interpolating between domains---插值桥梁做域自适应
- 2013 Subspace Interpolation via Dictionary Learning for Unsupervised Domain Adaptation---子空间插值
- 2014 How transferable are features in deep neural networks?--7647--帮助理解网络的迁移与 finetune
- 2015 CVPR Landmarks-based Kernelized Subspace Alignment for Unsupervised Domain Adaptation--landmarks
- 2015 DAN---Learning Transferable Features with Deep Adaptation Networks--ICML-3557--将多核 MMD 引入深度的先驱
- 2016 Deep CORAL: Correlation Alignment for Deep domain adaptation---CORAL 的深度化
- 2016 NIPS RTN--unsupervised domain adaptation with residual transfer networks---兼顾特征与分类器的残差
- 2016 Unsupervised Visual Representation Learning by Graph-based Consistent Constraints--图结构、对比学习与循环一致性
- 2016 nips-Domain Separation Networks---特征拆分为域特有与域共享再对齐
- 2016 progressive neural network--网络结构可重用历史知识
- 2017 CMD--CENTRAL MOMENT DISCREPANCY (CMD) FOR DOMAIN-INVARIANT REPRESENTATION LEARNING--新的度量准则
- 2017 Deep Unsupervised Convolutional Domain Adaptation--在卷积层引入 CORAL
- 2017 JAN--Deep transfer learning with joint adaptation networks--ICML--1636--JDA 的深度扩展
- 2017 Mind the class weight bias: weighted maximum mean discrepancy for unsupervised domain adaptation---类不平衡的加权 MMD
- 2017 Representation Learning by Learning to Count--计数任务的自监督
- 2018 A DIRT-T APPROACH TO UNSUPERVISED DOMAIN ADAPTATION---聚类假设提升辨别性
- 2018 Beyond Sharing Weights for Deep Domain Adaptation--PAMI--369--源域与目标域各自网络
- 2018 Boosting Self-Supervised Learning via Knowledge Transfer--拼图混合自监督
- 2018 DICD--Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adaptation---李爽老师的 DICD
- 2018 Learning Image Representations by Completing Damaged Jigsaw Puzzles--拼图自监督
- 2018 TNNSL--Feature analysis of marginalized stacked denoising autoencoder for unsupervised domain adaptation--自动编码器
- 2019 CADA--Attending to Discriminative Certainty for Domain Adaptation---attention 融入域自适应
- 2019 Contrastive Adaptation Network for Unsupervised Domain Adaptation--CAN--对比聚类
- 2019 Cross-Domain Visual Representations via Unsupervised Graph Alignment---源域与目标域不同类距离约束
- 2019 Domain Generalization by Solving Jigsaw Puzzles--CVPR--443---拼图自监督做域泛化
- 2019 MULTIPLE SUBSPACE ALIGNMENT IMPROVES DOMAIN ADAPTATION--多子空间对齐
- 2019 Self-Supervised Domain Adaptation for Computer Vision Tasks--87--旋转自监督
- 2019 Self-Supervised Representation Learning by Rotation Feature Decoupling--旋转相关性拆分
- 2019 TIP Locality Preserving Joint Transfer for Domain Adaptation---映射保持局部结构
- 2019 Transfer Independently Together: A Generalized Framework for Domain Adaptation---异构域自适应
- 2019 Transferrable Prototypical Networks for Unsupervised Domain Adaptation---原型网络
- 2019 UNSUPERVISED DOMAIN ADAPTATION THROUGH SELF-SUPERVISION--arxiv--150--多自监督任务辅助--代码：https://github.com/yueatsprograms/uda_release?utm_source=catalyzex.com
- 2019 Virtual Mixup Training for Unsupervised Domain Adaptation ---mixup
- 2020 DSAN--Deep Subdomain Adaptation Network for Image Classification--LMMD--将 STL 扩展到深度，提出 local MMD
- 2020 Discriminative Transfer Feature and Label Consistency for Cross-Domain Image Classification---李爽老师的 DTLC
- 2020 Domain Adaptation via Image Style Transfer---风格迁移
- 2020 Domain Conditioned Adaptation Network---通道注意力显式建模域差异
- 2020 Fisher deep domain adaptation--Fisher 判别
- 2020 IMPROVE UNSUPERVISED DOMAIN ADAPTATION WITH MIXUP TRAINING---mixup 做域自适应
- 2020 PAMI Unsupervised Domain Adaptation via Discriminative Manifold Propagation---黎曼流形
- 2020 Rethinking Distributional Matching Based Domain Adaptation---判别式替代分布匹配
- 2020 Spherical Space Domain Adaptation with Robust Pseudo-label Loss---球特征空间
- 2020 Unsupervised Domain Adaptation via Discriminative Manifold Embedding and Alignment---余弦相似度度量
- 2020 Unsupervised Transfer Learning with Self-Supervised Remedy--arxiv--3--自监督域自适应
- 2020 toward discriminability and diversity: batch nuclear-norm maximization under label insufficient situation---CVPR--提升辨别性与多样性
- 2021 Aligning Correlation Information for Domain Adaptation in Action Recognition---对齐 Gram 矩阵
- 2021 CDCL--cross domain contrastive learning for unsupervised domain adaptation ---对比损失
- 2021 Contrastive Domain Adaptation---对比学习
- 2021 DANICE：domain adaptation without forgetting in neural image compression---挑选参数进行微调
- 2021 Dynamic Weighted Learning for Unsupervised Domain Adaptation--辨别性与对齐的动态权重
- 2021 FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation---mixup
- 2021 Gradually Vanishing Bridge for Adversarial Domain Adaptation---搭桥分离域特征
- 2021 PAMI Aggregating Randomized Clustering-Promoting Invariant Projections for Domain Adaptation---聚类保持
- 2021 PAMI Generalized Domain Conditioned Adaptation Network---通道级别注意力
- 2021 PAMI Self-Supervised Learning Across Domains ----旋转与拼图联合
- 2021 PAMI Where and How to Transfer: Knowledge Aggregation-Induced Transferability Perception for Unsupervised Domain Adaptation--聚合感知迁移性
- 2021 Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation---原型 + 自监督
- 2021 Rethinking Maximum Mean Discrepancy for Visual Domain Adaptation---重新思考 MMD 并解决不足
- 2021 Self-Supervised Domain Adaptation with Consistency Training-6-旋转和拼图自监督--代码效果一般
- 2021 Transferrable Contrastive Learning for Visual Domain Adaptation--对比学习
- 2022 Unsupervised domain adaptation via distilled discriminative clustering---聚类对齐

### 对抗的
- 2014 DDC--Deep Domain Confusion: Maximizing for Domain Invariance--深度域自适应的鼻祖
- 2015 Unsupervised domain adaptation by backpropagation---DANN-ICML--3897--对抗先驱--代码：https://github.com/fungtion/DANN?utm_source=catalyzex.com
- 2017 ADDA--Adversarial Discriminative Domain Adaptation---早期对抗域自适应
- 2018 Conditional Adversarial Domain Adaptation---联合分类与特征
- 2018 MADA---multi-adversarial domain adaptation--多对抗，每类一个 GAN
- 2018 NIPS Conditional Adversarial Domain Adaptation---CADA，对抗与分类特征非线性融合
- 2019 DTA--Drop to Adapt: Learning Discriminative Features for Unsupervised Domain Adaptation---对抗融合类信息，使用 dropout
- 2019 Joint Adversarial Domain Adaptation---加入类辨别器
- 2019 Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation--类级对抗
- 2019 Transfer Learning with Dynamic Adversarial Adaptation Network----104--MEDA 扩展至神经网络
- 2019 Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation---约束较小奇异值
- 2020 Implicit Class-Conditioned Domain Alignment for Unsupervised Domain Adaptation---分析域对抗捷径并通过采样解决
- 2020 Maximum Density Divergence for Domain Adaptation--MDD-PAMI--121--新的度量准则融入对抗--代码：李晶晶主页
- 2020 Transformer Based Multi-Source Domain Adaptation---多源域对抗
- 2021 Bi-Classifier Determinacy Maximization for Unsupervised Domain Adaptation--双分类器
- 2021 Challenging tough samples in unsupervised domain adaptation---熵划分样本难度
- 2021 Cross-Domain Gradient Discrepancy Minimization for Unsupervised Domain Adaptation---引入梯度相似性
- 2021 DC-FUDA: Improving Deep Clustering via Fully Unsupervised Domain Adaptation ---聚类结合 GAN
- 2021 PAMI Divergence-agnostic Unsupervised Domain Adaptation by Adversarial Attacks---源域加扰动
- 2022 Adversarial Mixup Ratio Confusion for Unsupervised Domain Adaptation---mixup 多样化

### 分类器的方法
- 2017 Asymmetric Tri-training for Unsupervised Domain Adaptation--三个分类器
- 2018 Maximum Classifier Discrepancy for Unsupervised Domain Adaptation--MCD-CVPR--1153--差异分类器对抗--代码：https://github.com/mil-tokyo/MCD_DA?utm_source=catalyzex.com
- 2019 EASY TRANSFER LEARNING BY EXPLOITING INTRA-DOMAIN STRUCTURES ---无参分类器，偏传统
- 2020 Collaborative Unsupervised Domain Adaptation for Medical Image Diagnosis---双分类器判断目标域迁移性
- 2020 DADA--Discriminative Adversarial Domain Adaptation---融合域判别器与分类器
- 2021 Reducing bias to source samples for unsupervised domain adaptation---缓解分类器偏向源域

### 较前沿的域自适应
- 2014 Asymmetric and Category Invariant Feature Transformations for Domain Adaptation--源域与目标域维度不同
- 2016 CVPR Learning Cross-Domain Landmarks for Heterogeneous Domain Adaptation---异构域 landmarks
- 2016 Domain Adaptation in the Absence of Source Domain Data---源域数据不可用
- 2017 Distant domain transfer learning---远距域迁移
- 2018 zero-shot domain adaptation----zero-shot 域自适应
- 2019 On Learning Invariant Representation for Domain Adaptation--标签分布未对齐的危害
- 2020 Cross-domain Self-supervised Learning for Domain Adaptation with Few Source Labels--arxiv--24--源域标签很少--暂无代码
- 2020 Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation--ICML--258--无需源域数据，代码：https://github.com/tim-learn/SHOT?utm_source=catalyzex.com
- 2020 Domain Aggregation Networks for Multi-Source Domain Adaptation--多源域
- 2020 Heterogeneous Graph Attention Network for Unsupervised Multiple-Target Domain Adaptation---图神经网络
- 2020 Label-Noise Robust Domain Adaptation---源域标签有噪声
- 2020 PAMI Deep Residual Correction Network for Partial Domain Adaptation---部分域自适应
- 2020 Test-Time Training with Self-Supervision for Generalization under Distribution Shifts---159--测试时自监督--有代码
- 2020 Transformer Based Multi-Source Domain Adaptation---transformer 多源域
- 2020 Universal Domain Adaptation through Self-Supervision--NIPS-100--自监督聚类解决统一域自适应--代码：https://github.com/VisionLearningGroup/DANCE?utm_source=catalyzex.com
- 2021 Faster Domain Adaptation Networks---域自适应加速
- 2021 PDA--SOURCE CLASS SELECTION WITH LABEL PROPAGATION FOR PARTIAL DOMAIN ADAPTATION---部分域自适应
- 2021 Transformer-Based Source-Free Domain Adaptation---源域 free

### 传统的
- 2006 Analysis of Representations for Domain Adaptation--A-distance 等
- 2006 SCL--Domain Adaptation with Structural Correspondence Learning---结构相关学习，文本
- 2007 Analysis of Representations for Domain Adaptation--域自适应理论
- 2008 Transfer Learning via Dimensionality Reduction--AAAI-717--降维
- 2010 A theory of learning from different domains---基础理论
- 2010 Boosting for transfer learning with multiple sources--441--权重方法
- 2010 TCA-transfer component analysis--NN-3236--MMD 先驱
- 2012 GKF--Geodesic Flow Kernel for Unsupervised Domain Adaptation--CVPR--2160--流形映射
- 2013 ARTL--Adaptation Regularization: A General Framework for Transfer Learning---流形一致正则化
- 2013 JDA--Transfer Feature Learning with Joint Distribution Adaptation
- 2013 SA--Unsupervised Visual Domain Adaptation Using Subspace Alignment--1199--子空间对齐先驱
- 2014 TJM--Transfer Joint Matching for Unsupervised Domain Adaptation--CVPR-603
- 2014 TKL-Domain Invariant Transfer Kernel Learning--TKDE-176--度量准则
- 2015 SDA--Subspace Distribution Alignment for Unsupervised Domain Adaptation--138--子空间对齐
- 2015 Sample selection for visual domain adaptation via sparse coding---稀疏表示
- 2016 Correlation Alignment for Unsupervised Domain Adaptation--CORAL--CVPR--201--二阶统计量对齐
- 2016 Prediction Reweighting for Domain Adaptation---最近邻约束
- 2017 BDA--Balanced Distribution Adaptation for Transfer Learning
- 2017 SCA--Scatter Component Analysis: A Unified Framework for Domain Adaptation and Domain Generalization--散度与方差
- 2017 When Unsupervised Domain Adaptation Meets Tensor Representations--张量塔克分解
- 2018 Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adaptation--DICD
- 2018 Graph Adaptive Knowledge Transfer for Unsupervised Domain Adaptation---软标签
- 2018 Learning domain-shared group-sparse representation for unsupervised domain adaptation---稀疏编码
- 2018 MEDA--Visual Domain Adaptation with Manifold Embedded Distribution Alignment--流形、权重因子、动态对齐
- 2018 STL--Stratified Transfer Learning for Cross-domain Activity Recognition--特征映射，多子空间
- 2019 A Graph Embedding Framework for Maximum Mean Discrepancy-Based Domain Adaptation Algorithms---MMD 与图嵌入
- 2019 Bridging Theory and Algorithm for Domain Adaptation---理论
- 2019 Frustratingly Easy Domain Adaptation---特征扩展三个版本
- 2020 Domain Adaptation by Class Centroid Matching and Local Manifold Self-Learning（CMMS）---聚类整体赋标签
- 2020 Domain Adaptation on Graphs by Learning Aligned Graph Bases---图拉普拉斯
- 2020 Unsupervised Domain Adaptation via Structured Prediction Based Selective Pseudo-Labeling-SPL-AAAI-90--双伪标签策略加局部保留映射
- 2021 Unsupervised domain adaptation based on cluster matching and Fisher criterion for image classification--聚类匹配与 Fisher

### 任务
- 2017 Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection---图像分割
- 2017 Learning Features by Watching Objects Move---视频动态
- 2017 Show, Adapt and Tell: Adversarial Training of Cross-domain Image Captioner
- 2018 CyCADA: Cycle-Consistent Adversarial Domain Adaptation----循环一致
- 2018 FCTN--A full convolutional tri-branch network for domain adaptation---Tri-training
- 2018 Unsupervised domain adaptation for semantic segmentation via class-balanced self-train
- 2019 Unsupervised Person Re-Identification with Iterative Self-Supervised Domain Adaptation--CVF--2019
- 2020 MLSL--multi-level self-supervised learning for domain adaptation
- 2020 Domain Adaptation of Transformers for English Word Segmentation---词分割
- 2020 Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision---熵划分目标域难度
- 2021 CROSS-DOMAIN ACTIVITY RECOGNITION VIA SUB-STRUCTURAL OPTIMAL TRANSPORT
- 2021 Domain Adaptation in Multi-Channel Autoencoder based Features for Robust Face Anti-Spoofing---防欺诈
- 2021 IMPROVED DATA SELECTION FOR DOMAIN ADAPTATION IN ASR---语音识别
- 2021 Maximizing Cosine Similarity Between Spatial Features for Unsupervised Domain Adaptation in Semantic Segmentation---余弦相似度
- 2021 Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation--方差估计不确定性
- 2021 SELF-SUPERVISED LEARNING BASED DOMAIN ADAPTATION FOR ROBUST SPEAKER VERIFICATION---自监督做说话人
- 2021 Self-Supervised Learning for Domain Adaptation on Point Clouds
- 2021 UNSUPERVISED DOMAIN ADAPTATION FOR SPEECH RECOGNITION VIA UNCERTAINTY DRIVEN SELF-TRAINING---语音识别

### 非域自适应的论文
- 1987 Methodology review of clustering methods---864---聚类总结
- 2000 A new LDA-based face recognition system which can solve the small sample size problem---解决 LDA 小样本
- 2002 Learning from Labeled and Unlabeled Data with Label Propagation--标签传播
- 2002 On Spectral Clustering: Analysis and an algorithm---谱聚类
- 2003 LPP---Locality Preserving Projections ---降维
- 2004 Learning with Local and Global Consistency---半监督
- 2006 Graph Embedding and Extensions: A General Framework for Dimensionality Reduction---降维、图
- 2008 SSDA--Semi-Supervised Discriminant Analysis using Robust Path-Based Similarity---LDA 扩展
- 2015 Distilling the Knowledge in a Neural Network---蒸馏，softmax 加 temperature
- 2015 Supervised transfer kernel sparse coding for image classification---稀疏编码分类器
- 2015 TRAINING CONVOLUTIONAL NETWORKS WITH NOISY LABELS---腐败矩阵解决噪声标签
- 2016 Dimensionality Reduction by Learning an Invariant Mapping--弹簧类比
- 2016 Fast Patch-based Style Transfer of Arbitrary Style---图像风格迁移
- 2016 Image Style Transfer Using Convolutional Neural Networks--更细致的风格迁移
- 2016 Joint Unsupervised Learning of Deep Representations and Image Clusters--深度聚类
- 2016 Label Distribution Learning--多标签映射--有代码
- 2016 Unsupervised Deep Embedding for Clustering Analysis --深度聚类
- 2017 Focal Loss for Dense Object Detection---focal loss
- 2017 Learning Discrete Representations via Information Maximizing Self-Augmented Training--聚类方法
- 2017 Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach---噪声标签
- 2017 Multi-task Self-Supervised Visual Learning---四个自监督任务联合
- 2017 Self-supervised learning of visual features through embedding images into text topic spaces--CVPR--多模态
- 2017 cycleGan
- 2018 Cross-Domain Self-supervised Multi-task Feature Learning using Synthetic Imagery--合成图像自监督，使用 GAN
- 2018 Deep Clustering for Unsupervised Learning of Visual Features---深度聚类
- 2018 Harmonic Mean Linear Discriminant Analysis ---LDA 改进
- 2018 Squeeze-and-Excitation Networks---通道赋权
- 2018 Supervised Deep Sparse Coding Networks---稀疏编码融入网络
- 2018 nips--Using Trusted Data to Train Deep Networks on Labels Corrupted by Severe Noise---信任集与腐败矩阵
- 2019 Confidence Regularized Self-Training---伪标签过度自信
- 2019 Distant Supervised Centroid Shift: A Simple and Efficient Approach to Visual Domain Adaptation---缩小类内方差
- 2019 MixMatch: A Holistic Approach to Semi-Supervised Learning---半监督混合匹配
- 2019 NIPS Transferable normalization: toward improving transferable of deep neural network---BN 改进，可用于语音
- 2019 ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring---mixup 改进
- 2019 Revisiting Self-Supervised Visual Representation Learning---比较 CNN 宽度
- 2019 Self-Supervised Convolutional Subspace Clustering Network---聚类反馈网络
- 2019 Self-Supervised Semi-Supervised Learning---自监督 + 半监督
- 2019 WAV2VEC: UNSUPERVISED PRE-TRAINING FOR SPEECH RECOGNITION---自监督语音识别
- 2019 When Does Label Smoothing Help?---何时标签平滑有用
- 2019 When Does Label Smoothing Help?--885-NIPS---防止过拟合--代码：https://github.com/seominseok0429/label-smoothing-visualization-pytorch?utm_source=catalyzex.com
- 2019 nips--Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty---自监督提升鲁棒性
- 2020 A convolutional neural network with sparse representation---稀疏编码结合小波与剪切变换
- 2020 CVPR Self-Supervised Learning of Pretext-Invariant Representations---变换后的样本与原样本特征一致
- 2020 Online Deep Clustering for Unsupervised Representation Learning---聚类
- 2020 Pre-Trained Image Processing Transformer ---图像 transformer
- 2020 Random Erasing Data Augmentation---随机擦除数据增强
- 2020 Self‑supervised autoencoders for clustering and classification--自动编码器降维聚类
- 2020 self-supervised visual feature learning with deep neural networks A survey--PAMI-833--深度视觉自监督
- 2021 CPC--Representation Learning with Contrastive Predictive Coding---对比预测编码
- 2021 Emerging Properties in Self-Supervised Vision Transformers---自蒸馏自监督
- 2021 Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation---多图拼接，可借鉴语音
- 2021 TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech---绘图值得学习
