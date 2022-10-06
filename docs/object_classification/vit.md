
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

Our work adds to the increasing collection of papers that explore image recognition at larger scales than the standard ImageNet dataset. The use of additional data allows to achieve state-of-the-art results on standard benchmarks (Mahajan et al., 2018; Touvron et al., 2019; Xie et al., 2020). Moreover, Sun et al. (2017) study how CNN performance scales with dataset size, and Kolesnikov et al. (2020); Djolonga et al. (2020) perform an empirical exploration of CNN transfer learning from large scale datasets such as ImageNet-21k and JFT-300M. We focus on these two latter datasets as well, but train Transformers instead of ResNet-based models used in prior works.

     
| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| Exploring the Limits of Weakly Supervised Pretraining | - | Mahajan et al., 2018
| Fixing the train-test resolution discrepancy | - | Touvron et al., 2019
| Self-training with noisy student improves imagenet classification | - | Xie et al., 2020
| Revisiting Unreasonable(不合理的) of Data in Deep Learning Era | - | Sun et al. (2017)
| Big transfer(BiT): General visual representation learning | BiT | Kolesnikov et al. (2020)
| On robustness and transferability of convolutional neural networks  | - | Djolonga et al. (2020)


## 3 Method

In model design we follow the original Transformer (Vaswani et al., 2017) as closely as possible. An advantage of this intentionally simple setup is that scalable NLP Transformer architectures – and their efficient implementations – can be used almost out of the box.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| All is All You Need | Transformer | Vaswani et al., 2017

#### 3.1 Vision Transformer (ViT)

An overview of the model is depicted in Figure 1. The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, we reshape the image $\mathbf{x} \in \Bbb R^{H \times W \times C}$ into a sequence of flattened 2D patches $\mathbf{x}_{p} \in \Bbb {R}^{N \times (P^2 \cdot C)}$, where $(H,W)$ is the resolution of the original image, $C$ is the number of channels, $(P,P)$ is the resolution of each image patch, and $N=HW/P^2$ is the resulting number of patches, which also serves as the effective input sequence length for the Transformer. The Transformer uses constant latent vector size D through all of its layers, so we flatten the patches and map to $D$ dimensions with a trainable linear projection (公式1). We refer to the output of this projection as the patch embeddings.


公式1 : 

$$
\mathbf{z}_0 = [\mathbf{x}_{class};\, \mathbf{x}_{p}^{1} \mathbf{E}; \, \mathbf{x}_{p}^{2} \mathbf{E}; \, \cdots, \, \mathbf{x}_{p}^{N} \mathbf{E}] + \mathbf{E}_{pos} \qquad \qquad \mathbf{E} \in \Bbb{R}^{(P^2 \cdot C) \times D}, \, \mathbf{E}_{pos} \in \Bbb{R}^{(N+1) \times D}
$$


Similar to BERT's [class] token, we prepend（预先准备） a learnable embedding to the sequence of embedded patches ($\mathbf{z_0^0}=\mathbf{x}_{class}$), whose state at the output of the Transformer encoder $(\mathbf{z_L^0})$ serves as the image representation $\mathbf{y}$ (见公式4). Both during pre-training and fine-tuning, a classification head is attached to $\mathbf{y}$. The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.

Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings (Appendix D.4). The resulting sequence of embedding vectors serves as input to the encoder.


The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded selfattention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019).The MLP contains two layers with a GELU non-linearity.


| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| Attention is All You Need | Transformer | Vaswani et al., 2017
| Learning deep transformer models for machine translation | - | Wang et al., 2019
| Adaptive Input Representations for Neural Language Modeling | - | Baevski & Auli, 2019


$$ \begin{align}
\mathbf{z}_0 &= [\mathbf{x}_{class};\, \mathbf{x}_{p}^{1} \mathbf{E}; \, \mathbf{x}_{p}^{2} \mathbf{E}; \, \cdots, \, \mathbf{x}_{p}^{N} \mathbf{E}] + \mathbf{E}_{pos} \qquad \qquad \mathbf{E} \in \Bbb{R}^{(P^2 \cdot C) \times D}, \, \mathbf{E}_{pos} \in \Bbb{R}^{(N+1) \times D} \\

\mathbf{z}_{\ell}^{'}&=MSA(LN(\mathbf{z_{\ell-1}})) + \mathbf{z}_{\ell-1} \qquad \qquad \qquad \qquad \quad \, \ell=1 \cdots L \\

\mathbf{z}_{\ell}&=MLP(LN(\mathbf{z}_{\ell}^{'})) + \mathbf{z}_{\ell}^{'} \qquad \qquad  \qquad \qquad \qquad \quad  \ell=1 \cdots L \\

\mathbf{y}&=LN(\mathbf{z}_{L}^0)

\end{align} $$


**Inductive bias.** We note that Vision Transformer has much less image-specific inductive bias than CNNs.In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model.In ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global.The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at fine-tuning time for adjusting the position embeddings for images of different resolution (as described below). Other than that, the position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch.

**Hybrid Architecture** As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding projection E (Eq. 1) is applied to patches extracted from a CNN feature map. As a special case, the patches can have spatial size 1x1, which means that the input sequence is obtained by simply flattening the spatial dimensions of the feature map and projecting to the Transformer dimension. The classification input embedding and position embeddings are added as described above.

#### 3.2 Fine-Tuning And Higher Resolution

Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks.For this, we remove the pre-trained prediction head and attach a zero-initialized $D \times$ K$ feedforward layer, where $K$ is the number of downstream classes.