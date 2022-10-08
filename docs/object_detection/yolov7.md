
论文名称: YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

readpaper地址: https://readpaper.com/pdf-annotate/note?pdfId=4666972415243337729&noteId=729490235758215168

## Abstract 

YOLOv7 surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS and has the highest accuracy **56.8% AP** among all known real-time object detectors with **30 FPS or higher on GPU V100**. YOLOv7-E6 object detector (56 FPS V100, 55.9% AP) outperforms both transformer-based detector SWINL Cascade-Mask R-CNN (9.2 FPS A100, 53.9% AP) by 509% in speed and 2% in accuracy, and convolutionalbased detector ConvNeXt-XL Cascade-Mask R-CNN (8.6 FPS A100, 55.2% AP) by 551% in speed and 0.7% AP in accuracy, as well as YOLOv7 outperforms: YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5, DETR, Deformable DETR, DINO-5scale-R50, ViT-Adapter-B and many other object detectors in speed and accuracy.Moreover, we train YOLOv7 only on MS COCO dataset from scratch without using any other datasets or pre-trained weights. Source code is released in https:// github.com/ WongKinYiu/ yolov7.


## Introduction

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

