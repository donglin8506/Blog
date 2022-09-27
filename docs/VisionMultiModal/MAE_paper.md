
论文名称：Masked Autoencoders Are Scalable Vision Learners
论文地址：https://arxiv.org/abs/2111.06377
readpaper地址：https://readpaper.com/pdf-annotate/note?pdfId=4556957922861522945&noteId=724534621592129536
论文时间：2021.11.11
作者信息：Kaiming He, ..., Ross Girshick 


## 摘要

This paper shows that masked autoencoders(MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the subset of patchesss(without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g. 75%, yields a nontrival and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3x or more) and improve accuracy. Our scalable approach allows for learning high-capacity(大容量) models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy(87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior.

## 引言

