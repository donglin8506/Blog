
论文名称: An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale

readpaper地址: https://readpaper.com/pdf-annotate/note?pdfId=4666805048735449089&noteId=737652127941750784


## Abstract

While the Transformer architecture has become the de-facto(事实上) standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.


## 1 Introduction

Self-attention-based architectures, in particular Transformers (Vaswani et al., 2017), have become the model of choice in natural language processing (NLP).The dominant approach is to pre-train on a large text corpus and then fine-tune on a smaller task-specific dataset (Devlin et al., 2019). Thanks to Transformers' computational efficiency and scalability, it has become possible to train models of unprecedented(空前的) size, with over 100B parameters (Brown et al., 2020; Lepikhin et al., 2020).With the models and datasets growing, there is still no sign of saturating performance.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| Attention Is All You Need | Transformer | Vaswani et al., 2017
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | BERT | Devlin et al., 2019
| Language Models are Few-Shot Learners | GPT-3 | Brown et al., 2020
| GShard: Scaling Glant Models with Conditional Computation and Automatic Sharding | GShard | Lepikhin et al., 2020

In computer vision, however, convolutional architectures remain dominant (LeCun et al., 1989; Krizhevsky et al., 2012; He et al., 2016). Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019; Wang et al., 2020a). The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns.Therefore, in large-scale image recognition, classic ResNetlike architectures are still state of the art (Mahajan et al., 2018; Xie et al., 2020; Kolesnikov et al., 2020). 

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| Backpropagation applied to handwritten zip code recognition | - | LeCun et al., 1989
| ImageNet Classification with Deep Convolutional Neural Networks | - | Krizhevsky et al., 2012
| Deep Residual Learning for Image Recognition | - | He et al., 2016
| Non-local Neural Networks | - | Wang et al., 2018
| End-to-End Object Detection with Transformers | DETR | Carion et al., 2020
| Stand-Alone Self-Attention in Vision Models | - | Ramachandran et al., 2019
| Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation. | - | Wang et al., 2020a
| Exploring the Limits of Weakly Supervised Pretraining | - | Mahajan et al., 2018
| Self-training with noisy student improves imagenet classification | - | Xie et al., 2020
| Big transfer(BiT): General visual representation learning | BiT | Kolesnikov et al., 2020

Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard Transformer directly to images, with the fewest possible modifications. To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classification in supervised fashion.

When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size.This seemingly discouraging outcome may be expected(这个看起来令人沮丧的结果可能也是在预料之中): Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data.

However, the picture changes if the models are trained on larger datasets (14M-300M images).We find that large scale training trumps(胜过) inductive bias.Our Vision Transformer (ViT) attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints.When pre-trained on the public ImageNet-21k dataset or the in-house JFT-300M dataset, ViT approaches or beats state of the art on multiple image recognition benchmarks. In particular, the best model reaches the accuracy of 88.55% on ImageNet, 90.72% on ImageNet-ReaL, 94.55% on CIFAR-100, and 77.63% on the VTAB suite of 19 tasks.

## 2 Related Work

Transformers were proposed by Vaswani et al. (2017) for machine translation, and have since become the state of the art method in many NLP tasks.Large Transformer-based models are often pre-trained on large corpora and then fine-tuned for the task at hand:BERT (Devlin et al., 2019) uses a denoising self-supervised pre-training task, while the GPT line of work uses language modeling as its pre-training task (Radford et al., 2018; 2019; Brown et al., 2020).

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| Attention Is All You Need | Transformer | Vaswani et al., 2017
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | BERT | Devlin et al., 2019
| Improving language understanding with unsupervised learning | GPT | Radford et al., 2018
| Language models are unsupervised multitask learners | GPT-2 | Radford et al., 2019
| Language models are few-shot learners | GPT-3 | Brown et al., 2020

Naive application of self-attention to images would require that each pixel attends to every other pixel. With quadratic cost in the number of pixels, this does not scale to realistic input sizes. Thus, to apply Transformers in the context of image processing, several approximations(近似方法)have been tried in the past. Parmar et al. (2018) applied the self-attention only in local neighborhoods for each query pixel instead of globally. Such local multi-head dot-product self attention blocks can completely replace convolutions (Hu et al., 2019; Ramachandran et al., 2019; Zhao et al., 2020).In a different line of work, Sparse Transformers (Child et al., 2019) employ scalable approximations to global selfattention in order to be applicable to images.An alternative way to scale attention is to apply it in blocks of varying sizes (Weissenborn et al., 2019), in the extreme case only along individual axes (Ho et al., 2019; Wang et al., 2020a).Many of these specialized attention architectures demonstrate promising results on computer vision tasks, but require complex engineering to be implemented efficiently on hardware accelerators.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| Image transformer | - | Parmar et al. (2018)
| Local relation networks for image recognition | - | Hu et al., 2019
| Stand-alone self-attention in vision models | - | Ramachandran et al., 2019
| Exploring self-attention for image recognition | - | Zhao et al., 2020
| Generating long sequences with sparse transformers | Sparse Transformers | Child et al., 2019
| Scaling autoregressive video models | - | Weissenborn et al., 2019
| Axial attention in multidimensional transformers | - | Ho et al., 2019 
| Axial-deeplab: Stand-alone axial-attention for panoptic segmentation | - | Wang et al., 2020a


Most related to ours is the model of Cordonnier et al. (2020), which extracts patches of size 2 × 2from the input image and applies full self-attention on top.This model is very similar to ViT, but our work goes further to demonstrate that large scale pre-training makes vanilla transformers competitive with (or even better than) state-of-the-art CNNs.Moreover, Cordonnier et al. (2020) use a small patch size of 2 × 2 pixels, which makes the model applicable only to small-resolution images, while we handle medium-resolution images as well.There has also been a lot of interest in combining convolutional neural networks (CNNs) with forms of self-attention, e.g. by augmenting feature maps for image classification (Bello et al., 2019) or by further processing the output of a CNN using self-attention, e.g. for object detection (Hu et al., 2018; Carion et al., 2020), video processing (Wang et al., 2018; Sun et al., 2019), image classification (Wu et al., 2020), unsupervised object discovery (Locatello et al., 2020), or unified text-vision tasks (Chen et al., 2020c; Lu et al., 2019; Li et al., 2019).

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| On the relationship between self-attention and convolutional layers | - | Cordonnier et al. (2020)
| Attention augmented convolutional networks | - | Bello et al., 2019
| Relation Networks for Object Detection | - | Hu et al., 2018
| End-to-End Object Detection with Transformer | DETR | Carion et al., 2020
| Non-local Neural Networks | - | Wang et al., 2018
| VideoBERT: A Joint Model for Video and Language Representation Learning | VideoBERT | Sun et al., 2019
| Visual transformers: Token-based image representation and processing for computer vision | - | Wu et al., 2020
| Object-Centric Learning with Slot Attention | - | Locatello et al., 2020
| UNITER: Universal Image-Text Representation Learning | UNITER | Chen et al., 2020c
| ViLBERT: Pretraining Task-Agnostic Visiollnguistic(视觉语言主义者) Representations for Vision-and-Language Tasks | ViLBERT | Lu et al., 2019
| VisualBERT: A Simple and Performant Baseline for Vision and Language | VisualBERT | Li et al., 2019

Another recent related model is image GPT (iGPT) (Chen et al., 2020a), which applies Transformers to image pixels after reducing image resolution and color space.The model is trained in an unsupervised fashion as a generative model, and the resulting representation can then be fine-tuned or probed linearly for classification performance, achieving a maximal accuracy of 72% on ImageNet.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| Generative pretraining from pixels | iGPT | Chen et al., 2020a


     