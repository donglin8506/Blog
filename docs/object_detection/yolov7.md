
论文名称: YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

readpaper地址: https://readpaper.com/pdf-annotate/note?pdfId=4666972415243337729&noteId=729490235758215168


关键词: 

- 梯度分析 gradient analysis

- transition layers 


## Abstract 

YOLOv7 surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS and has the highest accuracy **56.8% AP** among all known real-time object detectors with **30 FPS or higher on GPU V100**. YOLOv7-E6 object detector (56 FPS V100, 55.9% AP) outperforms both transformer-based detector SWINL Cascade-Mask R-CNN (9.2 FPS A100, 53.9% AP) by 509% in speed and 2% in accuracy, and convolutionalbased detector ConvNeXt-XL Cascade-Mask R-CNN (8.6 FPS A100, 55.2% AP) by 551% in speed and 0.7% AP in accuracy, as well as YOLOv7 outperforms: YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5, DETR, Deformable DETR, DINO-5scale-R50, ViT-Adapter-B and many other object detectors in speed and accuracy.Moreover, we train YOLOv7 only on MS COCO dataset from scratch without using any other datasets or pre-trained weights. Source code is released in https:// github.com/ WongKinYiu/ yolov7.


## 1 Introduction

Real-time object detection is a very important topic in computer vision, as it is often a necessary component in computer vision systems. For example, multi-object tracking[94,93], autonomous driving [40,18], robotics [35,58], medical image analysis [34,46], etc. The computing devices that execute real-time object detection is usually some mobile CPU or GPU, as well as various neural processing units (NPU) developed by major manufacturers. For example, the Apple neural engine (Apple), the neural compute stick (Intel), Jetson AI edge devices (Nvidia), the edge TPU (Google), the neural processing engine (Qualcomm), the AI processing unit (MediaTek), and the AI SoCs (Kneron), are all NPUs.Some of the above mentioned edge devices focus on speeding up different operations such as vanilla convolution, depth-wise convolution, or MLP operations.In this paper, the real-time object detector we proposed mainly hopes that it can support both mobile GPU and GPU devices from the edge to the cloud.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [94] FairMOT: On the Fainess of Detection and Re-identification in Multiple Object Tracking | FairMOT | 2020-04-04
| [93] ByteTrack: Multi-Object Tracking by Associating Every Detection Box | ByteTrack | - 
| [40] GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving | GS3D | 2019-03-26
| [18] Deep Multi-Modal Object Detection and Semantic Segmentation for Autonomous Driving: Datasets, Methods, and Challenges | - | 2021-03-01
| [35] Object Detection Approach for Robot Grasp(抓住) Detection | - | 2019-05-20
| [50] Object Detection and Pose Estimation from RGB and Depth Data for Real-Time, Adaptive Robotic Grasping | - | 2021-01-18
| [34] Retina U-Net: Embarrassingly Simple Exploitation(简单到令人尴尬的利用) of Segmentation Supervision for Medical Object Detection | - | 2020-04-30
| [46] CLU-CNNs: Object detection for medical images | CLU-CNNs | 2019-07-20


In recent years, the real-time object detector is still developed for different edge device.For example, the development of MCUNet [49,48] and NanoDet [54] focused on producing low-power single-chip(低功耗单片机) and improving the inference speed on edge CPU. As for methods such as YOLOX [21] and YOLOR [81], they focus on improving the inference speed of various GPUs.More recently, the development of real-time object detector has focused on the design of efficient architecture.As for real-time object detectors that can be used on CPU[54,88,84,83], their design is mostly based on MobileNet [28,66,27], ShuffleNet [92,55], or GhostNet [25].Another mainstream real-time object detectors are developed for GPU [81,21,97], they mostly use ResNet [26], DarkNet [63], or DLA [87], and then use the CSPNet [80] strategy to optimize the architecture.The development direction of the proposed methods in this paper are different from that of the current mainstream real-time object detectors.In addition to architecture optimization, our proposed methods will focus on the optimization of the training process.Our focus will be on some optimized modules and optimization methods which may strengthen the training cost for improving the accuracy of object detection, but without increasing the inference cost. We call the proposed modules and optimization methods trainable bag-of-freebies.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [49] MCUNet: Tiny Deep Learning on IoT Devices | MCUNet | 2020-07-20
| [48] Memory-efficient Patch-based Inference for Tiny Deep Learning | - | 2021-12-06
| [21] YOLOX: Exceeding(超出) YOLO Series in 2021 | YOLOX | 2021-07-18
| [81] You Only Learn One Representaion: Unified Network for Multiple Tasks. | YOLOR | 2021-05-10
| [88] PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices | PP-PicoDet | -
| [84] MobileDets: Searching for Object Detection Architectures for Mobile Accelerators | MobileDets | 2021-06-01
| [83] FBNetV5: Neural Architecture Search for Multiple Tasks in One Run | FBNetV5 | 2021-11-19
| [28] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications | MobileNets | 2017-04-17
| [66] MobileNetV2: Inverted Residuals and Linear Bottlenecks | MobileNetV2 | 2018-01-13
| [27] Searching for MobileNetV3 | MobileNetV3 | 2019-05-06
| [92] ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices | ShuffleNet | 2017-07-04
| [55] ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design | ShuffleNet V2 | 2018-09-08
| [25] GhostNet: More Features from Cheap Operations | GhostNet | 2019-11-27
| [97] Objects as Points | CenterNet | 2019-04-16
| [26] Deep Residual Learning for Image Recognition | ResNet | 2015-12-10
| [63] YOLOv3: An Incremental Improvement | YOLOv3 | 2018-04-08
| [87] Deep Layer Aggregation | DLA | 2017-07-20
| [80] CSPNet: A New Backbone that can Enhance Learning Capability of CNN | CSPNet | 2019-11-27

Recently, model re-parameterization [13,12,29] and dynamic label assignment [20,17,42] have become important topics in network training and object detection.Mainly after the above new concepts are proposed, the training of object detector evolves many new issues. In this paper, we will present some of the new issues we have discovered and devise(设计) effective methods to address them.For model reparameterization, we analyze the model re-parameterization strategies applicable to layers in different networks with the concept of gradient propagation path, and propose planned re-parameterized model.In addition, when we discover that with dynamic label assignment technology, the training of model with multiple output layers will generate new issues. That is: "How to assign dynamic targets for the outputs of different branches?" For this problem, we propose a new label assignment method called coarse-to-fine(从粗到细) lead guided label assignment.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [13] RepVGG: Making VGG-style ConvNets Great Again | RepVGG | 2021-01-11
| [12] Diverse Branch Block: Building a Convolution as an Inception-like Unit | - | 2021-06-01
| [29] Online Convolutional Re-parameterization | - | - 
| [20] OTA: Optimal Transport Assignment for Object Detection | OTA | 2021-03-26
| [17] TOOD: Task-aligned One-stage Object Detection | TOOD | -
| [42] A Dual Weighting Label Assignment Scheme for Object Detection | - 


The contributions of this paper are summarized as follows: (1)we design several trainable bag-of-freebies methods, so that real-time object detection can greatly improve the detection accuracy without increasing the inference cost; (2) for the evolution of object detection methods, we found two new issues, namely how re-parameterized module replaces original module, and how dynamic label assignment strategy deals with assignment to different output layers.In addition, we also propose methods to address the difficulties arising from these issues;(3) we propose "extend" and "compound scaling" methods for the real-time object detector that can effectively utilize parameters and computation; and (4) the method we proposed can effectively reduce about 40% parameters and 50% computation of state-of-the-art real-time object detector, and has faster inference speed and higher detection accuracy.


## 2 Related work

#### 2.1 Real-time object detectors

Currently state-of-the-art real-time object detectors are mainly based on YOLO [61, 62, 63] and FCOS [76, 77], which are [3, 79, 81, 21, 54, 85, 23].Being able to become a state-of-the-art real-time object detector usually requires the following characteristics:(1) a faster and stronger network architecture;(2) a more effective feature integration method [22, 97, 37, 74, 59, 30, 9, 45];(3) a more accurate detection method [76, 77, 69]; (4) a more robust loss function [96, 64, 6, 56, 95, 57]; (5) a more efficient label assignment method [99, 20, 17, 82, 42];and (6) a more efficient training method.In this paper, we do not intend to explore self-supervised learning or knowledge distillation methods that require additional data or large model.Instead, we will design new trainable bag-of-freebies method for the issues derived from the state-of-the-art methods associated with (4), (5), and (6) mentioned above.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [61] You Only Look Once: Unified, Real-Time Object Detection | YOLO | -
| [62] YOLO9000: Better, Faster, Stronger | YOLO9000 | 2017-07-21
| [63] YOLOv3: An Incremental Improvement | YOLOv3 | 2018-04-08
| [76] FCOS: Fully Convolutional One-Stage Object Detection | FCOS | 2019-04-02
| [77] FCOS: A simple and strong anchor-free object detector | FCOS | 2020-06-14
| [3] YOLOv4: Optimal Speed and Accuracy of Object Detection | YOLOv4 | 2020-04-23
| [79] Scaled-YOLOv4: Scaling Cross Stage Partial Network | Scaled-YOLOv4 | 2020-11-16
| [81] You Only Learn One Representaion: Unified Network for Multiple Tasks. | YOLOR | 2021-05-10
| [21] YOLOX: Exceeding(超出) YOLO Series in 2021 | YOLOX | 2021-07-18
| [85] PP-YOLOE: An evolved version of YOLO | PP-YOLOE | -
| [22] NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection | NAS-FPN | 2019-04-16
| [97] Objects as Points | CenterNet | 2019-04-16
| [37] Panoptic Feature Pyramid Networks | - | 2019-01-08
| [74] EfficientDet: Scalable and Efficient Object Detection | EfficientDet | 2019-11-20
| [59] DetectoRS: Detecting Objects with Recursive Feature Pyramid and Swithchable Atrous Convolution | DetectoRS | 2021-06-01
| [30] A2-FPN: Attention Aggregation based Feature Pyramid Network for Instance Segmentation | A2-FPN | 2021-06-01
| [9] Dynamic Head: Unifying Object Detection Heads with Attentions | - | 2021-06-15
| [45] Exploring Plain Vision Transformer Backbones for Object Detection | - | -
| [69] Sparse R-CNN: End-to-End Object Detection with Learnable Proposals | - | 2021-06-01
| [96] IoU Loss for 2D/3D Object Detection | - | -
| [64] Generalized Intersection over Union: A Metric and A Loss fro Bounding Box Regression | - | 2019-02-25
| [6] AP-Loss for Accurate One-Stage Object Detection | - | 2021-11-01
| [56] A Ranking-based, Balanced Loss Function Unifying Classification and Localisation in Object Detection | - | 2020-09-28
| [95] Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression | - | 2019-11-19
| [57] Rank & Sort Loss for Object Detection and Instance Segmentation | - | 2021-07-24
| [99] AutoAssign: Differentiable Label Assignment for Dense Object Detection | AutoAssign | 2020-07-07
| [20] OTA: Optimal Transport Assignment for Object Detection | OTA | 2021-03-26
| [17] TOOD: Task-aligned One-stage Object Detection | TOOD | -
| [42] A Dual(双重的) Weighting Label Assignment Scheme for Object Detection | - 
| [82] End-to-End Object Detection with Fully Convolutional Network | - | 2020-12-07

#### 2.2 Model re-parameterization

Model re-parametrization techniques [71, 31, 75, 19, 33, 11, 4, 24, 13, 12, 10, 29, 14, 78] merge multiple computational modules into one at inference stage.The model re-parameterization technique can be regarded as an ensemble technique, and we can divide it into two categories, i.e., module-level ensemble and model-level ensemble.There are two common practices for model-level reparameterization to obtain the final inference model.One is to train multiple identical models with different training data, and then average the weights of multiple trained models.The other is to perform a weighted average of the weights of models at different iteration number.Modulelevel re-parameterization is a more popular research issue recently. This type of method splits a module into multiple identical or different module branches during training and integrates multiple branched modules into a completely equivalent module during inference.However, not all proposed re-parameterized module can be perfectly applied to different architectures. With this in mind, we have developed new re-parameterization module and designed related application strategies for various architectures.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [71] Rethinking the Inception Architecture for Computer Vision | - | 2015-12-02
| [31] Snapshot Ensembles: Train 1, Get M for Free | - | 2017-04-01
| [75] Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results | - | 2017-01-01
| [19] Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs | - | 2018-02-27
| [33] Averaging Weights Leads to Wider Optima and Better Generalization | - | 2018-03-14
| [11] ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks | ACNet | 2019-08-11
| [4] Ensemble deep learning in bioinformatics | - | 2020-08-17
| [24] ExpandNets: Linear Over-parameterization to Train Compact Convolutional Networks | ExpandNets | 2018-11-26
| [13] RepVGG: Making VGG-style ConvNets Great Again | RepVGG | 2021-01-11
| [12] Diverse Branch Block: Building a Convolution as an Inception-like Unit | - | 2021-06-01
| [10] Re-parameterizing Your Optimizers rather than Architectures | - | 2022-05-30
| [29] Online Convolutional Re-parameterization | - | -
| [14] Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs | - | -
| [78] An Improved One millisecond Mobile Backbone | - | -


#### 2.3 Model scaling

Model scaling [72, 60, 74, 73, 15, 16, 2, 51] is a way to scale up or down an already designed model and make it fit in different computing devices.The model scaling method usually uses different scaling factors, such as resolution (size of input image), depth (number of layer), width (number of channel), and stage (number of feature pyramid), so as to achieve a good trade-off for the amount of network parameters, computation, inference speed, and accuracy.Network architecture search (NAS) is one of the commonly used model scaling methods. NAS can automatically search for suitable scaling factors from search space without defining too complicated rules.The disadvantage of NAS is that it requires very expensive computation to complete the search for model scaling factors. In [15], the researcher analyzes the relationship between scaling factors and the amount of parameters and operations, trying to directly estimate some rules, and thereby obtain the scaling factors required by model scaling. Checking the literature, we found that almost all model scaling methods analyze individual scaling factor independently, and even the methods in the compound scaling category also optimized scaling factor independently.The reason for this is because most popular NAS architectures deal with scaling factors that are not very correlated.We observed that all concatenationbased models, such as DenseNet [32] or VoVNet [39], will change the input width of some layers when the depth of such models is scaled. Since the proposed architecture is concatenation-based, we have to design a new compound scaling method for this model.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [72] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | EfficientNet | 2019-05-24
| [60] Designing Network Design Spaces | - | 2020-03-30
| [74] EfficientDet: Scalable and Efficient Object Detection | EfficientDet | 2019-11-20
| [73] EfficientNetV2: Smaller Models and Faster Training | EfficientNetV2 | 2021-04-01
| [15] Fast and Accurate Model Scaling | - | 2021-03-11
| [16] Simple Training Strategies and Model Scaling for Object Detection | - | 2021-06-30
| [2] Revisiting ResNets: Improved Training and Scaling Strategies | - | 2021-12-06
| [51] Swin Transformer V2: Scaling up Capacity and Resolution | SwinTransformerV2 | 2021-11-18
| [32] Densely Connected Convolutional Networks | DenseNet | 2016-08-25
| [39] An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection | VoVNet | 2019-04-22 


## 3. Architecture

#### 3.1 Extended efficient layer aggregation networks

In most of the literature on designing the efficient architectures, the main considerations are no more than(不外乎是) the number of parameters, the amount of computation, and the computational density.Starting from the characteristics of memory access cost(内存访问成本), Ma et al. [55] also analyzed the influence of the input/output channel ratio, the number of branches of the architecture, and the element-wise operation on the network inference speed.Doll ́ar et al. [15] additionally considered activation when performing model scaling, that is, to put more consideration on the number of elements in the output tensors of convolutional layers.The design of CSPVoVNet [79] in Figure 2 (b) is a variation of VoVNet [39]. In addition to considering the aforementioned(前述的) basic designing concerns, the architecture of CSPVoVNet [79] also analyzes the gradient path, in order to enable the weights of different layers to learn more diverse features.The gradient analysis approach described above makes inferences faster and more accurate.ELAN [1] in Figure 2 (c) considers the following design strategy – "How to design an efficient network?." They came out with a conclusion: By controlling the shortest longest gradient path, a deeper network can learn and converge effectively. In this paper, we propose Extended-ELAN (E-ELAN) based on ELAN and its main architecture is shown in Figure 2 (d).

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [55] ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design | ShuffleNet V2 | 2018-09-08
| [15] Fast and Accurate Model Scaling | - | 2021-03-11
| [79] Scaled-YOLOv4: Scaling Cross Stage Partial Network | Scaled-YOLOv4 | 2020-11-16


Regardless of(无论) the gradient path length and the stacking number of computational blocks in large-scale ELAN, it has reached a stable state.If more computational blocks are stacked unlimitedly, this stable state may be destroyed, and the parameter utilization rate will decrease.The proposed E-ELAN uses expand, shuffle, merge cardinality(基数) to achieve the ability to continuously enhance the learning ability of the network without destroying the original gradient path.In terms of architecture, E-ELAN only changes the architecture in computational block, while the architecture of transition layer is completely unchanged.Our strategy is to use group convolution to expand the channel and cardinality of computational blocks.We will apply the same group parameter and channel multiplier to all the computational blocks of a computational layer.Then, the feature map calculated by each computational block will be shuffled into g groups according to the set group parameter g, and then concatenate them together.


#### 3.2 Model scaling for concatenation-based models

The main purpose of model scaling is to adjust some attributes of the model and generate models of different scales to meet the needs of different inference speeds.For example the scaling model of EfficientNet [72] considers the width, depth, and resolution.As for the scaled-YOLOv4 [79], its scaling model is to adjust the number of stages. In [15], Doll ́ar et al. analyzed the influence of vanilla convolution and group convolution on the amount of parameter and computation when performing width and depth scaling, and used this to design the corresponding model scaling method.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [72] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | EfficientNet | 2019-05-24
| [79] Scaled-YOLOv4: Scaling Cross Stage Partial Network | Scaled-YOLOv4 | 2020-11-16
| [15] Fast and Accurate Model Scaling | - | 2021-03-11


The above methods are mainly used in architectures such as PlainNet or ResNet.When these architectures are in executing scaling up or scaling down, the in-degree and out-degree of each layer will not change, so we can independently analyze the impact of each scaling factor on the amount of parameters and computation.However, if these methods are applied to the concatenation-based architecture, we will find that when scaling up or scaling down is performed on depth, the in-degree of a translation layer which is immediately after a concatenation-based computational block will decrease or increase, as shown in Figure 3 (a) and (b).


It can be inferred from the above phenomenon that(从以上现象不难推理出) we cannot analyze different scaling factors separately for a concatenation-based model but must be considered together.Take scaling-up depth as an example, such an action will cause a ratio change between the input channel and output channel of a transition layer, which may lead to a decrease in the hardware usage of the model.Therefore, we must propose the corresponding compound model scaling method for a concatenation-based model.When we scale the depth factor of a computational block, we must also calculate the change of the output channel of that block.Then, we will perform width factor scaling with the same amount of change on the transition layers, and the result is shown in Figure 3 (c).Our proposed compound scaling method can maintain the properties that the model had at the initial design and maintains the optimal structure.

## 4. Trainable bag-of-freebies

#### 4.1 Planned re-parameterized convolution

Although RepConv [13] has achieved excellent performance on the VGG [68], when we directly apply it to ResNet [26] and DenseNet [32] and other architectures, its accuracy will be significantly reduced(减少).We use gradient flow propagation paths(梯度流传播路径) to analyze how re-parameterized convolution should be combined with different network.We also designed planned re-parameterized convolution accordingly.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [13] RepVGG: Making VGG-style ConvNets Great Again | RepVGG | 2021-01-11
| [68] Very deep convolutional networks for large-scale image recognition | VGG | 2015-01-01
| [26] Deep Residual Learning for Image Recognition | ResNet | 2015-12-10
| [32] Densely Connected Convolutional Networks | DenseNet | 2016-08-25


RepConv actually combines 3 × 3 convolution, 1 × 1convolution, and identity connection in one convolutional layer.RepConv actually combines 3 × 3 convolution, 1 × 1convolution, and identity connection in one convolutional layer. After analyzing the combination and corresponding performance of RepConv and different architectures, we find that the identity connection in RepConv destroys the residual in ResNet and the concatenation in DenseNet, which provides more diversity of gradients for different feature maps. For the above reasons, we use RepConv without identity connection (RepConvN) to design the architecture of planned re-parameterized convolution.In our thinking, when a convolutional layer with residual or concatenation is replaced by re-parameterized convolution, there should be no identity connection.Figure 4 shows an example of our designed "planned re-parameterized convolution" used in PlainNet and ResNet. As for the complete planned re-parameterized convolution experiment in residual-based model and concatenation-based model, it will be presented in the ablation study session.


#### 4.2 Coarse for auxiliary(辅助) and fine for lead loss

Deep supervision [38] is a technique that is often used in training deep networks. Its main concept is to add extra auxiliary head in the middle layers of the network, and the shallow network weights with assistant loss as the guide.Even for architectures such as ResNet [26] and DenseNet [32] which usually converge well, deep supervision [70, 98, 67, 47, 82, 65, 86, 50] can still significantly improve the performance of the model on many tasks.Figure 5 (a) and (b) show, respectively, the object detector architecture "without" and "with" deep supervision.In this paper, we call the head responsible for the final output as the lead head, and the head used to assist（助攻） training is called auxiliary(辅助) head.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [38] Deeply-Supervised Nets | - | 2014-09-18
| [26] Deep Residual Learning for Image Recognition | ResNet | 2015-12-10
| [32] Densely Connected Convolutional Networks | DenseNet | 2016-08-25
| [70] Going Deeper with Convolutions | - | -
| [98] UNet++: A Nested U-Net Architecture for Medical image Segmentation | UNet++ | 2018-07-18
| [67] Object Detection from Scratch with Deep Supervision | DSOD | 2018-09-25
| [47] CBNetV2: A Composite Backbone Network Architecture for Object Detection | CBNetV2 | 2021-07-01
| [82] End-to-End Object Detection with Fully Convolutional Network | POTO | 2020-12-07
| [65] Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity | Sparse DETR | 2021-11-29
| [86] 3D-MAN: 3D Multi-frame Attention Network for Object Detection | 3D-MAN | 2021-06-01
| [50] YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection | YOLOStereo3D | 2021-05-30


Next we want to discuss the issue of label assignment. In the past, in the training of deep network, label assignment usually refers directly to the ground truth and generate hard label according to the given rules. However, in recent years, if we take object detection as an example, researchers often use the quality and distribution of prediction output by the network, and then consider together with the ground truth to use some calculation and optimization methods to generate a reliable soft label [61, 8, 36, 99, 91, 44, 43, 90, 20, 17, 42].For example, YOLO [61] use IoU of prediction of bounding box regression and ground truth as the soft label of objectness.In this paper, we call the mechanism that considers the network prediction results together with the ground truth and then assigns soft labels as "label assigner."

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [61] You Only Look Once: Unified, Real-Time Object Detection | YOLO | -
| [8] Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving | Gaussian YOLOv3 | 2019-04-09
| [36] Probabilistic Anchor Assignment with IoU Prediction for Object Detection | - | 2020-07-16
| [99] AutoAssign: Differentiable Label Assignment for Dense Object Detection | AutoAssign | 2020-07-07
| [91] Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection | - | 2019-12-05
| [44] Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection | - | 2020-06-08
| [43] Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection | - | 2020-11-25
| [90] VarifocalNet: An IoU-aware Dense Object Detector | VarifocalNet | 2020-08-31
| [20] OTA: Optimal Transport Assignment for Object Detection | OTA | 2021-03-26
| [17] TOOD: Task-aligned One-stage Object Detection | TOOD | -
| [42] A Dual(双重的) Weighting Label Assignment Scheme for Object Detection | - 


Deep supervision needs to be trained on the target objectives regardless of the circumstances of auxiliary head or lead head.During the development of soft label assigner related techniques, we accidentally discovered a new derivative issue, i.e., "How to assign soft label to auxiliary head and lead head ?" To the best of our knowledge, the relevant literature has not explored this issue so far. The results of the most popular method at present is as shown in Figure 5 (c), which is to separate auxiliary head and lead head, and then use their own prediction results and the ground truth to execute label assignment.The method proposed in this paper is a new label assignment method that guides both auxiliary head and lead head by the lead head prediction.In other words, we use lead head prediction as guidance to generate coarse-to-fine hierarchical labels, which are used for auxiliary head and lead head learning, respectively. The two proposed deep supervision label assignment strategies are shown in Figure 5 (d) and (e), respectively.


**Lead head guided label assigner** is mainly calculated based on the prediction result of the lead head and the ground truth, and generate soft label through the optimization process.This set of soft labels will be used as the target training model for both auxiliary head and lead head.The reason to do this is because lead head has a relatively strong learning capability, so the soft label generated from it should be more representative of the distribution and correlation between the source data and the target.Furthermore, we can view such learning as a kind of generalized residual learning. By letting the shallower auxiliary head directly learn the information that lead head has learned, lead head will be more able to focus on learning residual information that has not yet been learned.

**Coarse-to-fine lead head guided label assigner** also used the predicted result of the lead head and the ground truth to generate soft label.However, in the process we generate two different sets of soft label, i.e., coarse label and fine label, where fine label is the same as the soft label generated by lead head guided label assigner, and coarse label is generated by allowing more grids to be treated as positive target by relaxing the constraints of the positive sample assignment process. The reason for this is that the learning ability of an auxiliary head is not as strong as that of a lead head, and in order to avoid losing the information that needs to be learned, we will focus on optimizing the recall of auxiliary head in the object detection task. As for the output of lead head, we can filter the high precision results from the high recall results as the final output.However, we must note that if the additional weight of coarse label is close to that of fine label, it may produce bad prior at final prediction.Therefore, in order to make those extra coarse positive grids have less impact, we put restrictions in the decoder, so that the extra coarse positive grids cannot produce soft label perfectly. The mechanism mentioned above allows the importance of fine label and coarse label to be dynamically adjusted during the learning process, and makes the optimizable upper bound of fine label always higher than coarse label.

#### 4.3 Other trainable bag-of-freebies

In this section we will list some trainable bag-offreebies. These freebies are some of the tricks we used in training, but the original concepts were not proposed by us.The training details of these freebies will be elaborated in the Appendix, including (1) Batch normalization in conv-bn-activation topology: This part mainly connects batch normalization layer directly to convolutional layer. The purpose of this is to integrate the mean and variance of batch normalization into the bias and weight of convolutional layer at the inference stage.(2) Implicit knowledge in YOLOR [81] combined with convolution feature map in addition and multiplication manner: Implicit knowledge in YOLOR can be simplified to a vector by pre-computing at the inference stage. This vector can be combined with the bias and weight of the previous or subsequent convolutional layer. (3) EMA model: EMA is a technique used in mean teacher [75], and in our system we use EMA model purely as the final inference model.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [81] You Only Learn One Representaion: Unified Network for Multiple Tasks. | YOLOR | 2021-05-10
| [75] Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results | - | 2017-01-01