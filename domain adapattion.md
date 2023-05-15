这个总结我将按年份，思想内容   
**综述：**   
2006 MMD--A Kernel Method for the Two-Sample-Problem Arthur--MMD   
2009 A survey on transfer learing--- ----经典综述，阐述了与自适应的定义，分类，以及方法   
2010 Relative Clustering Validity Criteria: A Comparative Overview---关于聚类的一篇综述   
2019 A review of domain adaptation without target label   
2019 Transfer Adaptation Learning: A Decade Survey   
2020 A Decade Survey of Transfer Learning (2010–2020)   
2020 A comprehensive  survey on transfer learning ---1328   
2020 A survey of Unsupervised deep domain adaptation   
2020 A survey of unsupervised deep domain adaptation--TIST--308   
2020 Deep visual domain adaptation:  a survey---一篇深度域自适应的论文，2018到现在有一千多引用量   
2020 Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey ----视觉方面的自监督的综述   
2020 a survey of transformers---关于transformer的综述   
2021 Deep Spoken Keyword Spotting: An Overview---深度关键词的综述   
**伪标签的：**   
2016 RTN---unsupervised domain adaptation with residual transfer network--两个分类器，相差一个扰动函数   
2018 Unsupervised domain adaptation for semantic segmentation via class-balanceed self-train--ECCV-728---生成伪标签类不平衡--代码：https://github.com/yzou2/CBST?utm_source=catalyzex.com   
2019 progressive feature alignment for unsupervised domain adaptation--easy 2 hard   
2021 Domain Adaptation with Auxiliary Target Domain-Oriented Classifier---设计了一个特殊的目标域的分类器帮助产生伪标签 (https://github.com/02Bigboy/read_paper/assets/74250589/d010204a-ba72-40b9-b92f-00d2c6ce9b48)   
**基于特征的：**   
2006 Neural Structural Correspondence Learning for Domain Adaptation--自动编码器   
2013 Connecting the Dots with Landmarks: Discriminatively Learning Domain-Invariant Features for Unsupervised Domain Adaptation---landmarks--相似的源域样本加到目标域   
2013 Dlid: Deep learning for domain adaptation by interpolating between domains---插值桥梁做域自适应   
2013 Subspace Interpolation via Dictionary Learning for Unsupervised Domain Adaptation---子空间插值   
2014 How transferable are features in deep neural networks?--7647--帮助理解网络是怎么迁移的，以及finetune   
2015 CVPR Landmarks-based Kernelized Subspace Alignment for Unsupervised Domain Adaptation--landmarks   
2015 DAN---Learning Transferable Features with Deep Adaptation Networks--ICML-3557-将多核MMD引入深度的先驱   
2016 Deep CORAL: Correlation Alignment for Deep domain  adaptation---coral的深度化   
2016 NIPS RTN--unsupervised domain adaptation with residual transfer networks---利用残差，即适应的特征，又适应了分类器   
2016 Unsupervised Visual Representation Learning by Graph-based Consistent Constraints--图，对比学习，循环一致性   
2016 nips-Domain Separation Networks---这篇文章把从样本学到的表示分为域special部分和域共享部分，然后用共享的部分来对齐，那个special部分来帮助学得共享   
2016 progressive neural network--新的网络结构，可以重用之前的知识   
2017 CMD--CENTRAL MOMENT DISCREPANCY (CMD) FOR DOMAIN-INVARIANT REPRESENTATION LEARNING--这篇提出来一个新的度量准则--中心距差异   
2017 Deep Unsupervised Convolutional Domain Adaptation--deepcoral的扩展，在卷积层的特征也用coral--一般是在全连接层哈！--这个思想简单，以后可以试一试   
2017 JAN--Deep transfer learning with joint adaptation networks--ICML--1636--JDA的深度扩展吧   
2017 Mind the class weight bias: weighted maximum mean discrepancy for unsupervised domain adaptation---类不平衡，加权重。权重MMD   
2017 Representation Learning by Learning to Count--数数的自监督   
2018 A DIRT-T APPROACH TO UNSUPERVISED DOMAIN ADAPTATION---聚类假设，提高辨别性   
2018 Beyond Sharing Weights for Deep Domain Adaptation--PAMI--369--源域和目标域各自用一个网络   
2018 Boosting Self-Supervised Learning via Knowledge Transfer--两张不同的拼图混起来   
2018 DICD--Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adaptation---李爽老师的DICD   
2018 Learning Image Representations by Completing Damaged Jigsaw Puzzles--拼图消失自监督   
2018 TNNSL--Feature analysis of marginalized stacked denoising autoenconder for unsupervised domain adaptation--NNLS-28-自动编码器   
2019 CADA--Attending to Discriminative Certainty for Domain Adaptation---基于attention的域自适应   
2019 Contrastive Adaptation Network for Unsupervised Domain Adaptation--CAN--聚类--类间距离，类内距离   
2019 Cross-Domain Visual Representations via Unsupervised Graph Alignment---源域和目标域不同类的距离等于一个L   
2019 Domain Generalization by Solving Jigsaw Puzzles--CVPR--443---拼图的自监督域自适应   
2019 MULTIPLE SUBSPACE ALIGNMENT IMPROVES DOMAIN ADAPTATION--多子空间对齐   
2019 Self-Supervised Domain Adaptation for Computer Vision Tasks--87--旋转自监督，学到更好的视觉表示   
2019 Self-Supervised Representation Learning by Rotation Feature Decoupling--有些照片与旋转没关系，有些有关系   
2019 TIP Locality Preserving Joint Transfer for Domain Adaptation---映射   
2019 Transfer Independently Together: A Generalized Framework for Domain Adaptation---异构域自适应   
2019 Transferrable Prototypical Networks for Unsupervised Domain Adaptation---原型网络   
2019 UNSUPERVISED DOMAIN ADAPTATION THROUGH SELF-SUPERVISION--arxiv--150-用多个自监督任务去帮助对齐-旋转和位置--代码：https://github.com/yueatsprograms/uda_release?utm_source=catalyzex.com   
2019 Virtual Mixup Training for Unsupervised Domain Adaptation ---mixup   
2020 DSAN--Deep Subdomain Adaptation Network for Image Classification--LMMD-将STL扩展到深度，提出了localMMD   
2020 Discriminative Transfer Feature and Label Consistency for Cross-Domain Image Classification---李爽老师的DTLC   
2020 Domain Adaptation via Image Style Transfer---图片风格转化   
2020 Domain Conditioned Adaptation Network---这篇用到了通道注意力以及显示的学习源域和目标域的差异   
2020 Fisher deep domain adaptation--Fisher   
2020 IMPROVE UNSUPERVISED DOMAIN ADAPTATION WITH MIXUP TRAINING---mixup做域自适应   
2020 PAMI Unsupervised Domain Adaptation via Discriminative Manifold Propagation---黎曼流形来做的   
2020 Rethinking Distributional Matching Based Domain Adaptation---用一些判别模式来代替分布匹配   
2020 Spherical Space Domain Adaptation with Robust Pseudo-label Loss---球特征空间，就是将特征进行L2泛化后的特征   
2020 Unsupervised Domain Adaptation via Discriminative Manifold Embedding and Alignment---余弦相似度来做的类间和类内距离   
2020 Unsupervised Transfer Learning with Self-Supervised Remedy--arxiv--3--自监督域自适应   
2020 toward discriminability and diversity:batch nuclear-norm maximization under label insufficient situation---CVPR--提高辨别性和密度   
2021 Aligning Correlation Information for Domain Adaptation in Action Recognition---对齐源域和目标域的gram矩阵来对齐   
2021 CDCL--cross domain contrastive learning for unsupervised domain adaptation ---对比损失做域自适应   
2021 Contrastive Domain Adaptation---用对比损失做的   
2021 DANICE：domain adaptation without forgetting in neural image compression---模型迁移到小数据集上的策略---挑选一些参数进行finetune   
2021 Dynamic Weighted Learning for Unsupervised Domain Adaptation--辨别性和对齐动态权重调整   
2021 FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation---mixup   
2021 Gradually Vanishing Bridge for Adversarial Domain Adaptation---搭桥，总的特征减去域special的特征，就是域不变的特征   
2021 PAMI Aggregating Randomized Clustering-Promoting Invariant Projections for Domain Adaptation---聚类同类的弄近   
2021 PAMI Generalized Domain Conditioned Adaptation Network---通道级别的注意力   
2021 PAMI Self-Supervised Learning Across Domains ----旋转拼图一起做的   
2021 PAMI Where and How to Transfer: Knowledge  Aggregation-Induced Transferability Perception for Unsupervised Domain Adaptation--where and how   
2021 Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation---通过原型或者叫类中心做域自适应   
2021 Rethinking Maximum Mean Discrepancy for Visual Domain Adaptation---重新思考MMD的不足，通过理论推导发现的，并解决问题   
2021 Self-Supervised Domain Adaptation with Consistency Training-6-旋转和拼图自监督--有代码，效果感觉一般   
2021 Transferrable Contrastive Learning for Visual Domain Adaptation--对比学习   
2022 Unsupervised domain adaptation via distilled discriminative clustering---聚类对齐   
**对抗的：**   
2014 DDC--Deep Domain Confusion: Maximizing for Domain Invariance--深度域自适应的鼻祖了，比DAN还前面   
2015 Unsupervised domain adaptstion by backpropagation---DANN-ICML--3897---用对抗做域自适应先驱--代码：https://github.com/fungtion/DANN?utm_source=catalyzex.com   
2017 ADDA--Adversarial Discriminative Domain Adaptation ---也是比较早的对抗域自适应   
2018 Conditional Adversarial Domain Adaptation---分类器特征与特征提取的特征联合起来   
2018 MADA---multi-adversarial domain adptation--多对抗，即k类就k个GAN   
2018 NIPS Conditional Adversarial Domain Adaptation---CADA，对抗与分类的特征非线性融合   
2019 DTA--Drop to Adapt:Learning Discriminative Features for Unsupervised Domain Adaptation---对抗网络加入了类的信息，用的是drop out实现的   
2019 Joint Adversarial Domain Adaptation---加入了类辨别器   
2019 Taking A Closer Look at Domain Shift:Category-level Adversaries for Semantics Consistent Domain Adaptation--类级别的对抗   
2019 Transfer Learning with Dynamic Adversarial Adaptation Network----104--MEDA扩展到神经网络   
2019 Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation---考虑了较小奇异值的部分   
2020 Implicit Class-Conditioned Domain Alignment for Unsupervised Domain Adaptation---这篇分析了域对抗训练的一个捷径，并通过采样策略解决了这个问题   
2020 Maximum Density Divergence for Domain Adaptation--MDD-PAMI--121--提出来batch层面的新的度量准则，并融入对抗框架--代码：李晶晶主页   
2020 Transformer Based Multi-Source Domain Adaptation---多源域域自适应   
2021 Bi-Classifier Determinacy Maximization for Unsupervised Domain Adaptation--bi-class   
2021 Challenging tough samples in unsupervised domain adaptation---通过熵把样本分成简单和困难，然后再将困难的转化为容易的   
2021 Cross-Domain Gradient Discrepancy Minimization for Unsupervised Domain Adaptation---对抗是类不可知，引入梯度相似性   
2021 DC-FUDA: Improving Deep Clustering via Fully Unsupervised Domain Adaptation ---聚类结合着gans   
2021 PAMI  Divergence-agnostic Unsupervised Domain Adaptation by Adversarial Attacks---源域加扰动   
2022 Adversarial Mixup Ratio Confusion for Unsupervised Domain Adaptation---mixup,并不是只是非黑即白了   
**分类器的方法：**   
2017 Asymmetric Tri-training for Unsupervised Domain Adaptation--三个分类器   
2018 Maximum Classifier Discrepancy for Unsupervised Domain Adaptation--MCD-CVPR--1153--用两个差异的的分类器与特征提取层做类似对抗训练--代码：https://github.com/mil-tokyo/MCD_DA?utm_source=catalyzex.com   
2019 EASY TRANSFER LEARNING BY EXPLOITING INTRA-DOMAIN STRUCTURES ---无参的分类器--但好像是传统的   
2020 Collaborative Unsupervised Domain Adaptation for Medical Image Diagnosis---两个分类器来判断目标域的迁移性   
2020 DADA--Discriminative Adversarial Domain Adaptation---将域辨别器很分类器融合到一起设计的，效果是比较不错的。   
2021 Reducing bias to source samples for unsupervised domain adaptation---分类器偏向源域的解决办法   
较前沿的域自适应：   
2014 Asymmetric and Category Invariant Feature Transformations for Domain Adaptation--源域和目标域维度不一样   
2016 CVPR Learning Cross-Domain Landmarks for Heterogeneous Domain Adaptation---异构域自适应，landmarks   
2016 Domain Adaptation in the Absence of Source Domain Data---源域数据不可用   
2017 Distant domain transfer learning---远距离的域自适应   
2018 zero-shot domain adaptation----zero-shot域自适应   
2019 On Learning Invariant Representation for Domain Adaptation--标签分布没有对齐的危害   
2020 Cross-domain Self-supervised Learning for Domain Adaptation with Few Source Labels--arxiv--24-源域标签很少--可惜没有代码   
2020 Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation--ICML--258-不需要用源域数据，有代码！！https://github.com/tim-learn/SHOT?utm_source=catalyzex.com   
2020 Domain Aggregation Networks for Multi-Source Domain Adaptation--多源域单目标域   
2020 Heterogeneous Graph Attention Network for Unsupervised Multiple-Target Domain Adaptation---图神经网络，做异构域自适应   
2020 Label-Noise Robust Domain Adaptation---源域标签有噪声   
2020 PAMI Deep Residual Correction Network for Partial Domain Adaptation---部分域自适应   
2020 Test-Time Training with Self-Supervision for Generalization under Distribution Shifts---159--测试时用自监督--有点意思--有代码   
2020 Transformer Based Multi-Source Domain Adaptation---transformer做多源域自适应   
2020 Universal Domain Adaptation through Self-Supervision--NIPS-100-自监督聚类解决全面域自适应--代码：https://github.com/VisionLearningGroup/DANCE?utm_source=catalyzex.com   
2021 Faster Domain Adaptation Networks---域自适应的加速   
2021 PDA--SOURCE CLASS SELECTION WITH LABEL PROPAGATION FOR PARTIAL DOMAIN ADAPTATION---部分域自适应   
2021 Transformer-Based Source-Free Domain Adaptation---源域free   
**传统的：**   
2006 Analysis of Representations for Domain Adaptation--A-distance等   
2006 SCL--Domain Adaptation with Structural Correspondence Learning---结构相关学习，做文本的   
2007 Analysis of Representations for Domain Adaptation--域自适应的理论   
2008 Transfer Learning via Dimensionality Reduction--AAAI-717-降维做域自适应   
2010 A theory of learning from different domains---基础理论文章   
2010 Boosting for transfer learning with multiple sources--441--基于权重的方法   
2010 TCA-transfer component analysis--NN-3236--迁移学习MMD的先驱   
2012 GKF--Geodesic Flow Kernel for Unsupervised Domain Adaptation--CVPR--2160--流形学习做域自适应，但这个流形我还没太懂呀，这个映射有什么特殊之处？   
2013 ARTL--Adaptation Regularization: A General Framework for Transfer Learning---504--流形一致正则化   
2013 JDA--Transfer Feature Learning with Joint Distribution Adaptation   
2013 SA--Unsupervised Visual Domain Adaptation Using Subspace Alignment--1199--子空间对齐先驱   
2014 TJM--Transfer Joint Matching for Unsupervised Domain Adaptation--CVPR-603   
2014 TKL-Domain Invariant Transfer Kernel Learning--TKDE-176-介绍了一些度量准则   
2015 SDA--Subspace Distribution Alignment for Unsupervised Domain Adaptation--138--子空间对齐的方法   
2015 Sample selection for visual domain adaptation via sparse coding---稀疏表示   
2016 Correlation Alignment for Unsupervised Domain Adaptation--coral--CVPR--201---提出来二阶统计量协方差对齐   
2016 Prediction Reweighting for Domain Adaptation---最近邻约束   
2017 BDA--Balanced Distribution Adaptation for Transfer Learning   
2017 SCA--Scatter Component Analysis: A Unified Framework for Domain Adaptation and Domain Generalization--散度和方差   
2017 When Unsupervised Domain Adaptation Meets Tensor Representations--张量的塔克分解做的域自适应   
2018 Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adaptation--DICD   
2018 Graph Adaptive Knowledge Transfer for Unsupervised Domain Adaptation---软标签做域自适应   
2018 Learning domain-shared group-sparse representation for unsupervised domain adaptation---稀疏编码的域自适应   
2018 MEDA--Visual Domain Adaptation with Manifold Embedded Distribution Alignment--流形，权重因子，动态对齐   
2018 STL--Stratified Transfer Learning for Cross-domain Activity Recognition--特征映射，有多少个类，就有多少个子空间   
2019 A Graph Embedding Framework for Maximum Mean Discrepancy-Based Domain Adaptation Algorithms---mmd与图嵌入结合   
2019 Bridging Theory and Algorithm for Domain Adaptation---偏理论的一篇文章   
2019 Frustratingly Easy Domain Adaptation---每个特征分成了三个版本   
2020 Domain Adaptation by Class Centroid Matching and Local Manifold Self-Learning（CMMS）---利用聚类，目标域整个类一起给标签   
2020 Domain Adaptation on Graphs by Learning Aligned Graph Bases---图拉普拉斯   
2020 Unsupervised Domain Adaptation via Structured Prediction Based Selective Pseudo-Labeling-SPL-AAAI-90--用两个伪标签成成策略提升伪标签精度，并用局部保留映射做   
2021 Unsupervised domain adaptation based on cluster matching and Fisher criterion for image classification--聚类匹配和Fisher   
**任务：**   
2017 Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection---图像分割的   
2017 Learning Features by Watching Objects Move---看视频动不动   
2017 Show, Adapt and Tell: Adversarial Training of Cross-domain Image Captioner   
2018  CyCADA: Cycle-Consistent Adversarial Domain Adaptation----循环训练，映射过去，映射回来不变   
2018 FCTN--A full convolutional tri-branch network for domain adaptation---tri-train   
2018 Unsupervised domain adaptation for semantic segmentation via class-balanceed self-train   
2019 Unsupervised Person Re-Identification with Iterative Self-Supervised Domain Adaptation--CVF--2019   
2020  MLSL--multi-level self-supervised learning for domain adaptation   
2020 Domain Adaptation of Transformers for English Word Segmentation---词分割   
2020 Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision---用熵将目标域分成易分和难分！   
2021 CROSS-DOMAIN ACTIVITY RECOGNITION VIA SUB- STRUCTURAL OPTIMAL TRANSPORT   
2021 Domain Adaptation in Multi-Channel Autoencoder based Features for Robust Face Anti-Spoofing---域自适应在防欺诈时的应用   
2021 IMPROVED DATA SELECTION FOR DOMAIN ADAPTATION IN ASR---语音识别   
2021 Maximizing Cosine Similarity Between Spatial Features for Unsupervised Domain Adaptation in Semantic Segmentation---余弦相似度做的   
2021 Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation--用方差去估计伪标签不确定性   
2021 SELF-SUPERVISED LEARNING BASED DOMAIN ADAPTATION FOR ROBUST SPEAKER VERIFICATION---利用自监督的域自适应来做说话人识别   
2021 Self-Supervised Learning for Domain Adaptation on Point Clouds   
2021 UNSUPERVISED DOMAIN ADAPTATION FOR SPEECH RECOGNITION VIA UNCERTAINTY DRIVEN SELF-TRAINING---语音识别   
**非域自适应的论文：**   
1987 Methodology review of clustering methods---864---聚类方法的总结   
2000 A new LDA-based face recognition system which can solve the small sample size problem---解决了LDA小样本的问题   
2002 Learning from Labeled and Unlabeled Data with Label Propagation--标签传播   
2002 On Spectral Clustering: Analysis and an algorithm---谱聚类   
2003 LPP---Locality Preserving Projections ---降维的LPP   
2004 Learning with Local and Global Consistency---半监督   
2006 Graph Embedding and Extensions: A General Framework for Dimensionality Reduction---降维，图   
2008 SSDA--Semi-Supervised Discriminant Analysis using Robust Path-Based Similarity---LDA的扩展   
2015 Distilling the Knowledge in a Neural Network---只是蒸馏，softmax加了个temperature   
2015 Supervised transfer kernel sparse coding for image classification---稀疏编码分类器   
2015 TRAINING CONVOLUTIONAL NETWORKS WITH NOISY LABELS---含噪标签通过腐败矩阵解决   
2016 Dimensionality Reduction by Learning an Invariant Mapping--一个降维方法，用弹簧解释的   
2016 Fast Patch-based Style Transfer of Arbitrary Style---图片风格转化   
2016 Image Style Transfer Using Convolutional Neural Networks--图片风格转化，更细致   
2016 Joint Unsupervised Learning of Deep Representations and Image Clusters--一个深度聚类方法   
2016 Label Distribution Learning--多标签的问题，一个实例mapping到多个标签开心沮丧等--有代码   
2016 Unsupervised Deep Embedding for Clustering Analysis --一个深度聚类方法   
2017 Focal Loss for Dense Object Detection---focal loss 类不平衡，降低分类效果好的损失权重   
2017 Learning Discrete Representations via Information Maximizing Self-Augmented Training--一个聚类方法用到了嘻哈学习   
2017 Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach---标签含噪   
2017 Multi-task Self-Supervised Visual Learning---四个自监督任务一起训练一个网络   
2017 Self-supervised learning of visual features through embedding images into text topic spaces--CVPR-文字和图片的多模态！！   
2017 cycleGan   
2018 Cross-Domain Self-supervised Multi-task Feature Learning using Synthetic Imagery--合成图像的自监督，运用了GAN   
2018 Deep Clustering for Unsupervised Learning of Visual Features---深度聚类的方法   
2018 Harmonic Mean Linear Discriminant Analysis ---LDA的改进   
2018 Squeeze-and-Excitation Networks---每个通道赋予不同权重，龙哥和李爽老师都用到过   
2018 Supervised Deep Sparse Coding Networks---稀疏编码融入到网络里面   
2018 nips--Using Trusted Data to Train Deep Networks on Labels Corrupted by Severe Noise---利用信任集，和腐败矩阵，纠正腐败标签   
2019 Confidence Regularized Self-Training---伪标签自信过头   
2019 Distant Supervised Centroid Shift: A Simple and Efficient Approach to Visual Domain Adaptation---缩小类内方差，增加总体方差   
2019 MixMatch: A Holistic Approach to Semi-Supervised Learning---半监督方法，混合匹配   
2019 NIPS Transferable normalization:toward improving transferable of deep neural network.----BN的改进，可以考虑可不可以用在语音上   
2019 ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring.---对mixup的改进   
2019 Revisiting Self-Supervised Visual Representation Learning---不同的CNN对自监督任务的效果--通道数越多效果越好   
2019 Self-Supervised Convolutional Subspace Clustering Network---做聚类的，聚类的结果，又反馈到网络   
2019 Self-Supervised Semi-Supervised Learning---自监督半监督   
2019 WAV2VEC: UNSUPERVISED PRE-TRAINING FOR SPEECH RECOGNITION---自监督的语音识别   
2019 When Does Label Smoothing Help?---什么时候用标签平滑有帮助   
2019 When Does Label Smoothing Help?--885-NIPS---防止过拟合的技术，代码：https://github.com/seominseok0429/label-smoothing-visualization-pytorch?utm_source=catalyzex.com   
2019 nips--Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty---自监督能提高模型地鲁棒性，不应该仅仅从精度判别自监督有没有用   
2020 A convolutional neural network with sparse representation---稀疏编码的小波变换和剪切变换，融入到深度神经网络   
2020 CVPR Self-Supervised Learning of Pretext-Invariant Representations---变换后的样本与原样本学到的特征应该尽可能相似   
2020 Online Deep Clustering for Unsupervised Representation Learning---聚类   
2020 Pre-Trained Image Processing Transformer ---图片上的transformer   
2020 Random Erasing Data Augmentation---随机消除的数据增强--语音的specaug不会就是借鉴的这个吧   
2020 Self‑supervised autoencoders for clustering and classification--evolving systems--8---用自动编码器降维，然后做聚类   
2020 self-supervised visual feature learning with deep neural networks  A survey--PAMI-833---深度的视觉自监督处理方法   
2021 CPC--Representation Learning with Contrastive Predictive Coding---对比预测编码   
2021 Emerging Properties in Self-Supervised Vision Transformers---自蒸馏的自监督   
2021 Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation---这篇自监督任务有意思，几张图拼在一起，语音也可以考虑考虑   
2021 TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech---画图可以学一下   
