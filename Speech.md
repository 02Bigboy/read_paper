主要是读的关键词检测文章，其中有少部分ASR文章  
2014 Small-footprint keyword spotting using deep neural networks---最早的神经网络做小参数量关键词，softmax   
2015 A time delay neural network architecture for efficient modeling of long temporal contexts---dilation, TDNN，利用上下文信息   
2015 LAS-Listen, Attend and Spell ---端到端基于attention的方法   
2016 An End-to-End Architecture for Keyword Spotting and Voice Activity Detection---CRNN和CTC做的关键词   
2016 Multi-task learning and Weighted Cross-entropy for DNN-based Keyword Spotting---大词汇量迁移以及权重交叉熵   
2017 Attention Is All You Need---transformer的论文   
2017 Compressed time delay neural network for small-footprint keyword spotting---TDNN做关键词   
2017 End-to-end Keywords Spotting Based on Connectionist Temporal Classification for Mandarin---ASR+CTC做关键词   
2017 Hello Edge: Keyword Spotting on Microcontrollers---网络参数的计算，内存，以及性能   
2017 JOINT CTC-ATTENTION BASED END-TO-END SPEECH RECOGNITION USING MULTI-TASK LEARNING---CTC和attention联合训练   
2017 MAX-POOLING LOSS TRAINING OF LONG SHORT-TERM MEMORY NETWORKS FOR SMALL-FOOTPRINT KEYWORD SPOTTING---maxpooling做关键词   
2018 Attention-based End-to-End Models for Small-Footprint Keyword Spotting---基于attention的关键词   
2018 Compact Feedforward Sequential Memory Networks for Small-footprint Keyword Spotting---里面用到了多帧预测   
2018 CosFace: Large Margin Cosine Loss for Deep Face Recognition---softmax的几种变形   
2018 DEEP RESIDUAL LEARNING FOR SMALL-FOOTPRINT KEYWORD SPOTTING---resnet和扩张卷积做关键词   
2018 DONUT: CTC-based Query-by-Example Keyword Spotting---ctc做自定义关键词，先识别出是什么，然后测试再匹配   
2018 Efficient keyword spotting using time delay neural networks---TDNN做的，然后加了个跳连接   
2018 Robust Classification with Convolutional Prototype Learning---原型网络   
2018 Stochastic Adaptive Neural Architecture Search for Keyword Spotting---利用shortcut可以根据不同的样本进行网络的选择   
2018 TCN--An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling---TCN   
2019  A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling---五种声音事件检测的池化损失   
2019 A time delay neural network with shared weight self-attention for small-footprint keyword spotting---用一个矩阵代替自注意力的三个矩阵   
2019 Convolutional Neural Networks for Small-footprint Keyword Spotting---CNN做的关键词，其中涉及到一些减小参数那些   
2019 EFFICIENT KEYWORD SPOTTING USING DILATED CONVOLUTIONS AND GATING---wavenet,即因果的dilation卷积，做关键词   
2019 FOCAL LOSS AND DOUBLE-EDGE-TRIGGERED DETECTOR FOR ROBUST SMALL-FOOTPRINT KEYWORD SPOTTING---focal loss以及关键词重复两遍   
2019 Selective Kernel Networks---多个核就多个特征，然后加权（用核函数，就不用多个网络去学多个特征了）   
2019 SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition----这个就和图像上随机mask掉一块是差不多的   
2019 Temporal Convolution for Real-time Keyword Spotting on Mobile Devices ----TC-resnet---较早的一维卷积做关键词，可以同时考虑低高频信息   
2019 THE SPEECHTRANSFORMER FOR LARGE-SCALE MANDARIN CHINESE SPEECH RECOGNITION----transfomer的三个优化策略，帧率，采样，以及focal loss   
2019 Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context---感受野变长了   
2020 A depthwise separable convolutional neural network for keyword spotting on an embedded system---深度分离卷积做关键词   
2020 ADAPTATION OF RNN TRANSDUCER WITH TEXT-TO-SPEECH TECHNOLOGY FOR KEYWORD SPOTTING----用text2speech去生成关键词，就相当于多了样本   
2020 AdderNet: Do We Really Need Multiplications in Deep Learning?---addnet，卷积的乘法操作变成了加法操作   
2020 CRNN-CTC BASED MANDARIN KEYWORDS SPOTTING---CRNN和CTC在中文上做的   
2020 Depthwise separable convolutional ResNet with squeeze-and-excitation blocks for small-footprint keyword spotting---梦龙的dsresnet结合压缩激励（每个通道学一个权重）   
2020 Domain Aware Training for Far-field Small-footprint Keyword Spotting---远场域自适应利用域自适应转化为近场问题   
2020 Federated Self-Supervised Learning of Multi-Sensor Representations for Embedded Intelligence---二分类用对比损失做，挺有让人思考的感觉   
2020 Keyword retrieving in continuous speech using connectionist temporal classification---使用CTC的关键词   
2020 MINING EFFECTIVE NEGATIVE TRAINING SAMPLES FOR KEYWORD SPOTTING---负样本选择，来缓解正负样本不平衡   
2020 Multi-scale Convolution for Robust Keyword Spotting---做深度分离卷积时先降维再升维，可以减小参数量，然后在每层做损失   
2020 QUARTZNET: DEEP AUTOMATIC SPEECH RECOGNITION WITH 1D TIME-CHANNEL SEPARABLE CONVOLUTIONS ----深度分离卷积做语音识别   
2020 Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets---深度分离卷积两个模块的顺序交换一下   
2020 Re-weighted Interval Loss for Handling Data Imbalance Problem of End-to-End Keyword Spotting---解决类不平衡，类权重，和样本权重两个方法   
2020 SMALL-FOOTPRINT KEYWORD SPOTTING ON RAWAUDIO DATA WITH SINC-CONVOLUTIONS---用sinc卷积来抽特征，而不是用MFCC 那些   
2020 Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition---wenet最开始   
2021 A Novel Re-weighted CTC Loss for Data Imbalance in Speech Keyword Spotting---用FAS，FRR来给ctc权重   
2021 A Streaming End-to-End Framework For Spoken Language Understanding---语言理解，好像这个任务比较复杂   
2021 ATTENTION IS ALL YOU NEED IN SPEECH SEPARATION---transformer做语音分离   
2021 AUC OPTIMIZATION FOR ROBUST SMALL-FOOTPRINT KEYWORD SPOTTING WITH LIMITED TRAINING DATA---梦龙的多维AUC   
2021 Broadcasted residual learning for efficient keyword spotting---频率维度压缩再扩展，为的是减小参数量   
2021 deformable TDNN with adaptive receptive fields for speech reconition---感受野可变的TDNN，用的是插值实现的   
2021 End-to-End Transformer-Based Open-Vocabulary Keyword Spotting with Location-Guided Local Attention---attention估计出关键词的起始位置，然后忽略两端   
2021 Energy-friendly keyword spotting system using add-based convolution---卷积的乘法用加法替代   
2021 Few-Shot Keyword Spotting in Any Language Mark---大模型finetune来解决fewshot关键词   
2021 Keyword transformer: A self-attention model for keyword spotting---transformer做关键词   
2021 Learning Efficient Representations for Keyword Spotting with Triplet Loss---triplet做关键词   
2021 LIGHTWEIGHT DYNAMIC FILTER FOR KEYWORD SPOTTING---卷积参数通过样本算出来，动态卷积   
2021 Metric Learning for Keyword Spotting--用类中心来做的   
2021 Noisy student-teacher training for robust keyword spotting---老师学生网络   
2021 RECENT DEVELOPMENTS ON ESPNET TOOLKIT BOOSTED BY CONFORMER---espnet   
2021 Text Anchor Based Metric Learning for Small-footprint Keyword Spotting---将triplet loss 的那个中心换成了另一个特征   
2021 U2++: Unified Two-pass Bidirectional End-to-end Model for Speech Recognition---左右attention mask   
2021 visual Keyword Spotting with Attention---视频与关键词，多模态   
2021 TRANSFORMER-BASED END-TO-END SPEECH RECOGNITION WITH LOCAL DENSE SYNTHESIZER ATTENTION---梦龙的attention只考虑local   
2021 LAC-Efficient conformer-based speech recognition with linear attention---强哥对网络的改进   
2021 rotary- Conformer-based End-to-end Speech Recognition With Rotary Position Embedding ---位置编码用矩阵的旋转变换   
2022 CONVMIXER: FEATURE INTERACTIVE CONVOLUTION WITH CURRICULUM LEARNING FOR SMALL FOOTPRINT AND NOISY FAR-FIELD KEYWORD SPOTTING---progressive 训练   
2022 END-TO-END LOWRESOURCE KEYWORD SPOTTING THROUGH CHARACTER RECOGNITION AND BEAM-SEARCH RE-SCORING---librspeech预训练模型去帮助GSC   
2022 Few-Shot Keyword Spotting With Prototypical Networks---类中心   
2022 IMPROVING FEATURE GENERALIZABILITY WITH MULTITASK LEARNING IN CLASS INCREMENTAL LEARNING---多分类分解成多个子任务   
2022 Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection---自监督帮助视频造假的判断   
2022 PROGRESSIVE CONTINUAL LEARNING FOR SPOKEN KEYWORD SPOTTING---连续学习   
2022 Two-stage streaming keyword detection and localization with multi-scale depthwise temporal convolution---MDTC   
2022 Understanding Audio Features via Trainable Basis Functions Kwan---时频谱变成可学习的   
2022 UNIFIED SPECULATION, DETECTION, AND VERIFICATION KEYWORD SPOTTING---maxpooling延时可控，利用了VAD   
