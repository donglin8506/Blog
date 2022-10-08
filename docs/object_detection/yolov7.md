
论文名称: YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

readpaper地址: https://readpaper.com/pdf-annotate/note?pdfId=4666972415243337729&noteId=729490235758215168

## Abstract 

YOLOv7 surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS and has the highest accuracy **56.8% AP** among all known real-time object detectors with **30 FPS or higher on GPU V100**. YOLOv7-E6 object detector (56 FPS V100, 55.9% AP) outperforms both transformer-based detector SWINL Cascade-Mask R-CNN (9.2 FPS A100, 53.9% AP) by 509% in speed and 2% in accuracy, and convolutionalbased detector ConvNeXt-XL Cascade-Mask R-CNN (8.6 FPS A100, 55.2% AP) by 551% in speed and 0.7% AP in accuracy, as well as YOLOv7 outperforms: YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5, DETR, Deformable DETR, DINO-5scale-R50, ViT-Adapter-B and many other object detectors in speed and accuracy.Moreover, we train YOLOv7 only on MS COCO dataset from scratch without using any other datasets or pre-trained weights. Source code is released in https:// github.com/ WongKinYiu/ yolov7.


## Introduction

Real-time object detection is a very important topic in computer vision, as it is often a necessary component in computer vision systems. For example, multi-object tracking[94,93], autonomous driving [40,18], robotics [35,58], medical image analysis [34,46], etc. The computing devices that execute real-time object detection is usually some mobile CPU or GPU, as well as various neural processing units (NPU) developed by major manufacturers.

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
