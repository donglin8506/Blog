
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


(ii) Information density(密度) is different between language and vision. Languages are human-generated signals that are highly semantic and information-dense. When training a model to predict only a few missing words per sentence, this task appears to induce(诱导) sophisticated(复杂的) language understanding. Images, on the contrary, are natural signals with heavy spatial redundancy(冗余) - e.g., a missing patch can be recoverted from neighboring patches with little high-level understanding of parts, objects, and scenes. 


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
