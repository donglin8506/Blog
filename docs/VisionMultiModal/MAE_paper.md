
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


(ii) Information density(密度) is different between language and vision. Languages are human-generated signals that are highly semantic and information-dense. When training a model to predict only a few missing words per sentence, this task appears to induce(诱导) sophisticated(复杂的) language understanding. Images, on the contrary, are natural signals with heavy spatial redundancy(冗余) - e.g., a missing patch can be recoverted from neighboring patches with little high-level understanding of parts, objects, and scenes. To overcome this difference and encourage learning useful features, we show that a simple strategy works well in computer vision: masking a very high portion of random patches. This strategy largely reduces redundancy and creates a chanllenging self-supervisory task that requires holistic(整体的) understanding beyond(超过) low-level image statistics. To get a qualitative(定性的) sense of our reconstruction task, see Figures 2 - 4.


(iii) The autoencoder's decoder, which maps the latent representation back to the input, plays a different role between reconstructing text and images. In vision, the decoder reconstructs pixels, hence its output is of a lower semantic level than common recognition tasks. This is in contrast to language, where the decoder predicts missing words that contain rich semantic information. While in BERT the decoder can be trival(琐碎的) (an MLP) [14], we found that for images, the decoder design plays a key role in determining the semantic level of the learned latent representations.


Driven by this analysis, we present a simple, effective, and scalable form of a masked autoencoder(MAE) for visual representation learning. Our MAE masks random patches from the input image and reconstructs the missing patches in the pixel space. It has an asymmetric encoder-decoder design. Our encoder operates only on the visible subset of patches (without mask tokens), and our decoder is lightweight and reconstructs the input from the latent representation along with mask tokens (Figure 1). Shifting the mask tokens to the small decoder in our asymmetric(不对称的) encoder-decoder results in a large reduction in computation. Under this design, a very high masking ratio (e.g., 75%) can achieve a win-win scenario(双赢的设想):it optimizes accuracy while allowing the encoder to process only a small portion(e.g., 25%) of patches. This can reduce overall pre-training time by 3x or more and likewise(同样的) reduce memory consumption(消耗), enabling us to easily scale our MAE to large models.


Our MAE learns very high-capacity that generalize well. With MAE pre-training, we can train data-hungry models like ViT-large/-Huge[16] on ImageNet-1K with improved generalization performance. With a vanilla ViT-Huge model, we achieve 87.8% accuracy when fine-tuned on ImageNet-1K. This outperforms all previous results that use only ImageNet-1K data. We also evaluate transfer learning on object detection, instance segmentation, and semantic segmentation. In these tasks, our pre-training achieves better results than its supervised pre-training counterparts, and more importantly, we observe significant gains by scaling up models. These observations are aligned with those witnessed in self-supervised pre-training in NLP[14,40,41,4] and we hope that they will enable our field to explore a similar trajectory(轨迹).


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


## 方法

Our masked autoencoder(MAE) is a simple autoencoding approach that reconstructs the original signal given its partial observation(基于它部分的观察). Like all autoencoders, our approach has an encoder that maps the observed signal to a latent representation, and a decoder that reconstructs the original signal from the latent representation. Unlike classical autoencoders, we adopt an asymmetric design that allows the encoder to operate only on the partial, observed signal(without mask tokens) and a lightweight decoder that reconstructs the full signal from the latent representation and mask tokens. Figure 1 illustrates the idea, introduced next.


图1

Figure 1. **Our MAE architecture**. During pre-training, a large random subset of image patches(e.g., 75%) is masked out. The encoder is applied to the subset of visible patches. Mask tokens are introduced after the encoder, and the full set of encoded patches and mask tokens is processed by a small decoder that reconstructs the original image in pixels. After pre-training, the decoder is discarded(丢弃) and the encoder is applied to uncorrupted images to produce representations for recognition tasks.


**Masking.** Followiing ViT[16], we divide an image into regular non-overlapping patches. Then we sample a subset of patches and mask (i.e., remove) the remaining ones. Our sampling strategy is straightforward: we sample random patches without replacement, following a uniform distribution. We simply refer to this as "random sampling".


| 论文名称 | 论文标题翻译 | 论文别名 | 论文时间
| :------- | :------- | :------ | :--------
| [16]An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 用Transformers做大规模图片识别 | ViT | 2020-10-22

Random sampling with a high masking ratio (i.e., the ratio of removed patches) largely eliminates(消除) redundancy, thus creating a task that cannot be easily solved by extrapolation(外插、外延) from visible neighboring patches (see Figures 2-4). The uniform distribution prevents a potential center bias (i.e., more masked patches near the image center). Finally, the highly sparse input creates an opportunity for designing an efficient encoder, introduced text.

图2

Figure 2. Example results on ImageNet validation images. For each triplet, we show the masked image (left), our MAE reconstruction (middle), and the ground-truth (right). The masking ratio is 80%, leaving only 39 out of 196 patches. More examples are in the appendix. *As no loss is computed on visible patches. the model output on visible patches is qualitatively worse. One can simply overlay(覆盖) the output with the visible patches to improve visual quality. We intentionally(故意地) opt not do this, so we can more comprehensively demonstrate the method's behavior.*

图3

Figure 3. Example results on COCO validation images, using an MAE trained on ImageNet (the same model weights in Figure 2). Observe the reconstructions on the two right-most examples, which, although different from the ground truth, are semantically plausible(似是而非).

图4

Figure 4. Reconstructions of ImageNet validation images using an MAE pre-trained with a masking ratio of 75% but applied on inputs with higher masking ratios. The predictions differ plausibly from the original images, showing that the method can generalize.


**MAE encoder.** Our encoder is a ViT[16] but applied only on visible, unmasked patches. Just as in a standard ViT, our encoder embeds patches by a linear projection with added positional embeddings, and then processes the resulting set via a series of Transformer blocks. However, our encoder only operates on a small subset (e.g. 25%) of the full set. Masked patches are removed; no mask tokens are used. This allows us to train very large encoders with only a fraction of compute and memory. The full set is handled by a lightweight decoder, described next.


| 论文名称 | 论文标题翻译 | 论文别名 | 论文时间
| :------- | :------- | :------ | :--------
| [16]An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 用Transformers做大规模图片识别 | ViT | 2020-10-22

**MAE decoder.** The input to the MAE decoder is the full set of tokens consisting of (i) encoded visible patches, and (ii) mask tokens. See Figure 1. Each mask token[14] is a shared, learned vector that indicates the presence of (表示..的存在) a missing patch to be predicted. We add positional embeddings to all tokens in this full set; without this, mask tokens would have no information about their location in the image. The decoder has another series of Transformer blocks.

| 论文名称 | 论文标题翻译 | 论文别名 | 论文时间
| :------- | :------- | :------ | :--------
| [14] | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | 深度双向Transformers的预训练用于语言理解 | BERT | - 

The MAE decoder is only used during pre-training to perform the image reconstruction task (only the encoder is used to produce image representations for recognition). Therefore, the decoder architecture can be flexibly designed in a manner that is independent of the encoder design. We experiment with very samll decoders, narrower and shallowe than the encoder. For example, our default decoder has <10% computation per token vs. the encoder. With this asymmetrical design, the full set of tokens are only processed by the lightweight decoder, which significantly reduces pre-training time.

**Reconstruction target.** Our MAE reconstructs the input by predicting the pixel values for each masked patch. Each element in the decoder's output is a vector of pixel values representing a patch. The last layer of the decoder is a linear projection whose number of output channels equals the number of pixel values in a patch. The decoder's output is reshaped to form a reconstructed image. Our loss function computes the mean squared error (MSE) between the reconstructed and original images in the pixel space. We compute the loss only on masked patches, similar to BERT[14].

| 论文名称 | 论文标题翻译 | 论文别名 | 论文时间
| :------- | :------- | :------ | :--------
| [14] | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | 深度双向Transformers的预训练用于语言理解 | BERT | - 

We also study a variant whose reconstruction target is the normalized pixel values of each masked patch. Specifically, we compute the mean and standard deviation of all pixels in a patch and use them to normalize this patch. Using normalized pixels as the reconstruction target improves representation quality in our experiments.

**Simple implementation.** Our MAE pre-training can be implemented efficiently, and importantly, does not require any specialized sparse operations. First we generate a token for every input patch (by linear projection with an added positional embedding). Next we randomly shuffle the list of tokens and remove the last portion of the list, based on the masking ratio. This process produces a small subset of tokens for the encoder and is equivalent to sampling patches withour replacement. After encoding, we append a list of mask tokens to the list of encoded patches, and unshuffle this full list (inverting the random shuffle operation) to align all tokens with their targets. The decoder is applied to this full list (with positional embeddings added). As noted, no sparse operations are needed. This simple implementation introduces negligible(微不足道的) overhead(开销) as the shuffling and unshuffling operations are fast.

## 讨论和结论

Simple algorithms that scale well are the core of deep learning. In NLP, simple self-supervised learning methods (e.g. [40,14,41,4]) enable benefits from exponentially scaling models. In computer vision, practical pre-training paradigms are dominantly supervised (e.g. [28,44,24,16]) despite progress in self-supervised learning. In this study, we observe on ImageNet and in transfer learning that an autoencoder - a simple self-supervised method similar to techniques in NLP - provides scalable benefits. Self-supervised learning in vision may now be embarking on a similar trajectory as in NLP.

| 论文名称 | 论文标题翻译 | 论文别名 | 论文时间
| :------- | :------- | :------ | :--------
| [14]BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | 深度双向Transformers的预训练用于语言理解 | BERT | - 
| [40]Improving Language Understanding by Generative Pre-Training | 使用生成式的预训练方式来提升语言理解 | GPT | - 
| [41]Language Models are Unsupervised Multitask Learners | 语言模型是无监督多任务的学习器 | GPT-2 | - 
| [4]Language Models are Few-Shot Learners | 语言模型是小样本学习器 | GPT-3 | 2020-05-28
| [28]ImageNet Classification with Deep Convolutional Neural Networks | 使用卷积神经网络做图片分类 | AlexNet | 2012-12-03
| [44]Very Deep Convolutional Networks for Large-Scale Image Recognition | - | - | 2014-09-04
| [24]Deep Residual Learning for Image Recognition | 使用残差学习来做图片分类 | ResNet | 2015-12-10
| [16]An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 用Transformers做大规模图片识别 | ViT | 2020-10-22

