
论文名称：Masked Autoencoders Are Scalable Vision Learners

论文地址：https://arxiv.org/abs/2111.06377

readpaper地址：https://readpaper.com/pdf-annotate/note?pdfId=4556957922861522945&noteId=724534621592129536

论文时间：2021.11.11

作者信息：Kaiming He, ..., Ross Girshick 


## 摘要

This paper shows that masked autoencoders(MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the subset of patchesss(without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g. 75%, yields a nontrival and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3x or more) and improve accuracy. Our scalable approach allows for learning high-capacity(大容量) models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy(87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior.

## 引言

Deep learning has witnessed(见证了) an explosion of architectures of continuously growing capability（能做成一件事的能力） and capacity（是否能高效地完成一件事情）[AlexNet, ResNet, Transformer]. Aided by the rapid gains in hardware, models today can easily overfit one million images [ImageNet ] and begin to demand(要求，需要) hundreds of millions of - often publicly inaccessible(公众无法访问的、不易访问的) - labeled images [ViT ]. 


| 论文名称 | 论文标题翻译 | 论文别名 | 论文时间
| :------- | :------- | :------ | :--------
| ImageNet Classification with Deep Convolutional Neural Networks | 使用卷积神经网络做图片分类 | AlexNet | 2012-12-03
| Deep Residual Learning for Image Recognition | 使用残差学习来做图片分类 | ResNet | 2015-12-10
| Attention Is All You Need | 注意力机制是你需要的 | Transformer | - 
| ImageNet: A large-scale hierarchical image database | 一个大规模的层次的图片数据集 | ImageNet | 2009-06-20


This appetite(食欲) of data has been successfully addressed in natural language processing (NLP) by self-supervised pre-training. The solutions, based on autoregressive language modeling in GPT and masked autoencdoing in BERT, are conceptually(概念上) simple: they remove a portion of the data and learn to predict the removed content. These methods now enable training of generalizable（具有泛化性的）NLP models containing over one hundred of billion parameters. 


| 论文名称 | 论文标题翻译 | 论文别名 | 论文时间
| :------- | :------- | :------ | :--------
| Improving Language Understanding by Generative Pre-Training | 使用生成式的预训练方式来提升语言理解 | GPT | - 
| Language Models are Unsupervised Multitask Learners | 语言模型是无监督多任务的学习器 | GPT-2 | - 
| Language Models are Few-Shot Learners | 语言模型是小样本学习器 | GPT-3 | 2020-05-28
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | 深度双向Transformers的预训练用于语言理解 | BERT | - 

The idea of masked autoencoders, a form of more general denoising autoencoders[48], is natural and applicable(适用的) in computer vision as well. Indeed, closely related research in vision[49,39], preceded(先于) BERT. However, despite significant interest in this idea following the success of BERT, progress of autoencoding methods in vision lags behind NLP. We ask: what makes masked autoencoding different between vision and language? We attempt to answer this question from the following perspectives:


(i) Until recently, architectures（架构） were different. In vision, convolutional networks[29] were dominant(占据主导地位的) over the last decade[28]. Convolutions typically operate on regular grids and it is not straightforward to integrate(整合) 'indicators'(指标) such as mask tokens[14] or positional embeddings[47] into convolutional networks.
This architectural gap, however, has been addressed with the introduction of Vision Transformers(ViT)[16] and should no longer present an obstacle(障碍).


(ii) Information density(密度) is different between language and vision. Languages are human-generated signals that are highly semantic and information-dense. When training a model to predict only a few missing words per sentence, this task appears to induce(诱导) sophisticated(复杂的) language understanding. Images, on the contrary, are natural signals with heavy spatial redundancy(冗余) - e.g., a missing patch can be recoverted from neighboring patches with little high-level understanding of parts, objects, and scenes. To overcome this difference and encourage learning useful features, we show that a simple strategy works well in computer vision: masking a very high portion of random patches. This strategy largely reduces redundancy and creates a chanllenging self-supervisory task that requires holistic(整体的) understanding beyond(超过) low-level image statics. To get a qualitative(定性的) sense of our reconstruction task, see Figures 2 - 4.


(iii) The autoencoder's decoder, which maps the latent representation back to the input, plays a different role between reconstructing text and images. In vision, the decoder reconstructs pixels, hence its output is of a lower semantic level than common recognition tasks. This is in contrast to language, where the decoder predicts missing words that contain rich semantic information. While in BERT the decoder can be trival(琐碎的) (an MLP) [14], we found that for images, the decoder design plays a key role in determining the semantic level of the learned latent representations.


Driven by this analysis, we present a simple, effective, and scalable form of a masked autoencoder(MAE) for visual representation learning.
Our MAE masks random patches from the input image and reconstructs the missing patches in the pixel space. It has an asymmetric encoder-decoder design. Our encoder operates only on the visible subset of patches (without mask tokens), and our decoder is lightweight and reconstructs the input from the latent representation along with mask tokens (Figure 1). Shifting the mask tokens to the small decoder in our asymmetric(不对称的) encoder-decoder results in a large reduction in computation. Under this design, a very high masking ratio (e.g., 75%) can achieve a win-win scenario(双赢的设想):it optimizes accuracy while allowing the encoder to process only a small portion(e.g., 25%) of patches. This can reduce overall pre-training time by 3x or more and likewise(同样的) reduce memory consumption(消耗), enabling us to easily scale our MAE to large models.


Our MAE learns very high-capacity that generalize well. With MAE pre-training, we can train data-hungry models like ViT-large/-Huge[16] on ImageNet-1K with improved generalization performance. With a vanilla ViT-Huge model, we achieve 87.8% accuracy when fine-tuned on ImageNet-1K. This outperforms all previous previous results that use only ImageNet-1K data. We also evaluate transfer learning on object detection, instance segmentation, and semantic segmentation. In these tasks, our pre-training achieves better results than its supervised pre-training counterparts, and more importantly, we observe significant gains by scaling up models. These observations are aligned with those witnessed in self-supervised pre-training in NLP[14,40,41,4] and we hope that they will enable our field to explore a similar trajectory(轨迹).


| 论文名称 | 论文标题翻译 | 论文别名 | 论文时间
| :------- | :------- | :------ | :--------
| [48]Extracting and Composing robust features with denoising auto encoders | 使用噪声自编码器的方式提取并组织鲁棒的特征 | - | 2008-07-05
| [49]Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion(标准) | 堆叠噪声自编码器：用一个局部的噪声标准去学习深度网络中有用的表征 | - | 2010-12-01
| [39]Context Encoders: Feature Learning by Inpainting | 上下文编码器：使用修复的方式来做特征学习 | - | 2016-04-25
| [29] Backpropagation applied to handwritten zip code recognition | 反向传播应用到手写体识别 | - | 1989-12-01
| [28] ImageNet Classification with Deep Convolutional Neural Networks | 使用卷积神经网络做图片分类 | AlexNet | 2012-12-03
| [14] | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | 深度双向Transformers的预训练用于语言理解 | BERT | - 
| [47]Attention Is All You Need | 注意力机制是你需要的 | Transformer | - 
| [16]An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 用Transformers做大规模图片识别 | ViT | 2020-10-22
| [40]Improving Language Understanding by Generative Pre-Training | 使用生成式的预训练方式来提升语言理解 | GPT | - 
| [41]Language Models are Unsupervised Multitask Learners | 语言模型是无监督多任务的学习器 | GPT-2 | - 
| [4]Language Models are Few-Shot Learners | 语言模型是小样本学习器 | GPT-3 | 2020-05-28

## 相关工作

**Masked language modeling** and its autoregressive counterparts, e.g., BERT[14] and GPT[40,41,4], are highly successful methods for pre-training in NLP. These methods hold out a portion of the input sequence and train models to predict the missing content. These methods have been shown to scale excellently(非常好地)[4] and a large abundance of evidence indicates that these pre-trained representations generalize well to various downstream tasks.


**Autoencoding** is a classical(经典的) method for learning representations. It has encoder that maps an input to a latent representation and a decoder that reconstructs the input. For example, PCA and k-means are autoencoders[25]. Denoising autoencoders (DAE) [48] are a class of autoencoders that corrupt(腐败、败坏) an input signal and learn to reconstruct the original, uncorrupted signal. A series of methods can be thought of as a generalized DAE under different corruptions, e.g., masking pixels[49,39,6] or removing color channels[59]. Our MAE is a form of denosing autoencoding, but different from the classical DAE in numerous ways.


**Masked image encoding** methods learn representations from images corrupted by masking. The pioneering work of [49] presents masking as a noise type in DAE. Context Encoder[39] inpaints(修补、修复) large missing regions using convolutional networks. Motivated by the success in NLP, related recent methods[6,16,2] are based on Transformers[47]. iGPT[6] operates on sequences of pixels and predicts unknown pixels. The ViT paper[16] studies masked patch prediction for self-supervised learning. Most recently, BEiT[2] proposes to predict discrete(离散的) tokens[37,43].



| 论文名称 | 论文标题翻译 | 论文别名 | 论文时间
| :------- | :------- | :------ | :--------
| [40]Improving Language Understanding by Generative Pre-Training | 使用生成式的预训练方式来提升语言理解 | GPT | - 
| [41]Language Models are Unsupervised Multitask Learners | 语言模型是无监督多任务的学习器 | GPT-2 | - 
| [4]Language Models are Few-Shot Learners | 语言模型是小样本学习器 | GPT-3 | 2020-05-28
| [25]Autoencoders, Minimum Descirption Length and Helmholtz Free Energy | 自动编码器、最小描述长度和Helmholtz自由能 | - | 1993-11-29
| [48]Extracting and Composing robust features with denoising auto encoders | 使用噪声自编码器的方式提取并组织鲁棒的特征 | - | 2008-07-05
| [49]Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion(标准) | 堆叠噪声自编码器：用一个局部的噪声标准去学习深度网络中有用的表征 | - | 2010-12-01
| [39]Context Encoders: Feature Learning by Inpainting | 上下文编码器：使用修复的方式来做特征学习 | - | 2016-04-25
| [6] Generative Pretraining From Pixels | 从像素中做生成式的预训练 | iGPT | 2020-07-12
| [59] Colorful Image Colorization | 彩色图像着色 | - | 2016-03-28
| [16]An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 用Transformers做大规模图片识别 | ViT | 2020-10-22
| [2]BEiT: BERT Pre-Training of Image Transformers | 将bert的预训练方式使用到图片transformer中 | BEiT | 2021-06-15
| [47]Attention Is All You Need | 注意力机制是你需要的 | Transformer | - 
| [37]Neural Discrete Representation Learning. | 神经的离散表征学习 | VQ-VAE | 2017-11-02
| [43]Zero-Shot Text-to-Image Generation | 零样本的文本到图片的生成 | - | 2021-02-24




**Self-supervised learning** approaches have been significant interest in computer vision, often focusing on different pretext tasks for pre-training [15,50,35,59,38,17]. Recently, contrastive learning[3,21] has been popular, e.g., [51,36,22,7], which models image similarity and disimilarity(差异性)(or only similarity[20,8])
between two or more views. Contrastive and related methods strongly depend on data augmentation[7,20,8]. Autoencoding pursues(追求) a conceptually different direction, and it exhibits(展示) different behaviors as we will present.


| 论文名称 | 论文标题翻译 | 论文别名 | 论文时间
| :------- | :------- | :------ | :--------
| [15]Unsupervised visual representation learning by context prediction. | 通过上下文预测来进行无监督的视觉表征学习 | - | 2015
| [50]Unsupervised Learning of Visual Representations using Videos | 使用视频的无监督视频表征学习 | - | 2015-05-04
| [35]Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles | 通过解决拼图 的无监督视觉表征学习 | 2016-03-30
| [59] Colorful Image Colorization | 彩色图像着色 | - | 2016-03-28
| [38]Learning Features by Watching Objects Move | 通过观察物体移动来学习特征 | - | 2016-12-19
| [17]Unsupervised Representation Learning by Predicting Image Rotations | 通过预测图片旋转来进行无监督的表征学习 
| [3]Self-organizing neural network that discovers surfaces in random-dot stereograms | 在随机点立体图中发现表面的自组织神经网络 | - | 2992-01-09
| [21]Dimensionality Reduction by Learning an Invariant Mapping | 通过学一个不变的映射，来是维度减小 | DrLIM | 2006-06-17
| [51]Unsupervised Feature Learning via Non-Parametric Instance Discrimination | 通过无参数的实例判别任务来进行无监督特征学习 | - | -
| [36]Representation Learning with Contrastive Predictive Coding | 基于对比预测编码的表征学习 | - | 2018-07-10
| [22]Momentum Contrast for Unsupervised Visual Representation Learning | 动量对比方法用于无监督视频表征学习 | MoCo | - 
| [7]A Simple Framework for Contrastive Learning of Visual Representations | 一种视觉表征对比学习的简单框架 | SimCLR | 2020-02-13
| [20]Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning | 引导你自己的潜力：一种自监督学习的新方法 | T3C | 2020-06-13
| [8]Exploring Simple Siamese Representation Learning | 探索简单的连体表示学习 | - | 2020-11-20

