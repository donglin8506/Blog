
论文名称: X-CLIP: End-to-End Multi-grained Contrastive Learning for Video-Text Retrieval

readpaper地址: https://readpaper.com/pdf-annotate/note?pdfId=4670865857576960001&noteId=738680763071156224


## Abstract

Video-text retrieval has been a crucial(重要的) and fundamental task in multi-modal research. The development of video-text retrieval has been considerably promoted(已经大大提升) by large-scale multi-modal contrastive pre-training, which primarily focuses on coarse-gained or fine-grained contrast. However, cross-grained contrast, which is the contrast between coarse-grained representations and fine-grained representations, has rarely been explored in prior research.Compared with fine-grained or coarse-grained contrasts, cross-grained contrast calculate the correlation between coarse-grained features and each fine-grained feature,and is able to filter out the unnecessary fine-grained features guided by the coarse-grained feature during similarity calculation, thus improving the accuracy of retrieval.To this end, this paper presents a novel multi-grained contrastive model, namely X-CLIP, for video-text retrieval.However, another challenge lies in the similarity aggregation problem, which aims to aggregate fine-grained and cross-grained similarity matrices to instance-level similarity. To address this challenge, we propose the Attention Over Similarity Matrix (AOSM) module to make the model focus on the contrast between essential frames and words, thus lowering the impact of unnecessary frames and words on retrieval results.With multi-grained contrast and the proposed AOSM module, X-CLIP achieves outstanding performance on five widely-used video-text retrieval datasets, including MSRVTT (49.3 R@1), MSVD (50.4 R@1), LSMDC (26.1 R@1), DiDeMo (47.8 R@1) and ActivityNet (46.2 R@1). It outperforms the previous state-of-the-art by +6.3%, +6.6%, +11.1%, +6.7%, +3.8% relative improvements on these benchmarks, demonstrating the superiority of multi-grained contrast and AOSM. Code is available at https://github.com/xuguohai/X-CLIP.

## 1 Introduction

Video-text retrieval (VTR) is a multi-modal task, which aims to find the most relevant video/text based on the text/video query.With the explosive growth of videos on the Internet, VTR has attracted increasing interests and served as an important role in people's daily life.Recent years have witnessed the rapid development of VTR, which is supported by a series of pre-training multi-modal models [4, 30, 44], innovative(创新的) retrieval methods [3, 5, 13–15, 24, 30, 34, 35, 38 , 41, 54 , 58, 61 , 63, 66 ] and video-text benchmarks [ 2, 6, 7, 45 , 56].

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [4] Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval | - | 2021-01-01
| [30] Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling | CLIPBERT | 2021-06-20
| [44] Learning Transferable Visual Models From Natural Language Supervision | CLIP | 2021-02-26
| [3] Vivit: A video vision transformer | Vivit | 2021
| [5] Is Space-Time Attention All You Need for Video Understanding? | TimeSformer | 2021
| [13] Dual Dense Encoding for Zero-Example Video Retrieval | - | 2019
| [14] MDMMT: Multidomain Multimodal Transformer for Video Retrieval | MDMMT | 2021
| [15] Multi-modal Transformer for Video Retrieval | - | -
| [34] HiT: Hierarchical Transformer with Momentum Contrast for Video-Text Retrieval  | HiT | 2021
| [35] Use what you have: Video retrieval using representations from collaborative experts | - | 2019
| [38] Clip4clip: An empirical study of clip for end to end video clip retrieval | Clip4clip | 2021
| [41] Learning joint embedding with multimodal cues for cross-modal video-text retrieval | - | 2018
| [54] T2VLAD: Global-Local Sequence Alignment for Text-Video Retrieval | T2VLAD | 2021
| [58] Tree-Augmented Cross-Modal Encoding for Complex-Query Video Retrieval | - | 2020
| [61] A joint sequence fusion model for video question answering and retrieval | - | 2018
| [63] Cross-modal and hierarchical modeling of video and text | - | 2018
| [66] Actbert: Learning global-local video-text representations | Actbert | 2020
| [2] Localizing Moments in Video with Natural Language | - | 2017-08-04
| [6] ActivityNet: A large-scale video benchmark for human activity understanding | ActivityNet | 2015-06-07
| [7] Collecting Highly Parallel Data for Paraphrase Evaluation | - | 2011-06-19
| [45] The Long-Short Story of Movie Description | - | 2015-10-07
| [56] MSR-VTT: A Large Video Description Dataset for Bridgin Video and Language | MSR-VTT | 2016-01-01


Recently, with great success in large-scale contrastive languageimage pre-training, VTR has also achieved great progress. Specifically, with 400M image-text pairs for training, CLIP [ 44] can embed the images and sentences into the shared semantic space for similarity calculation.Furthermore, CLIP4Clip [ 38 ] transfers the imagetext knowledge of CLIP to the VTR task, resulting in significant performance improvements on several video-text retrieval datasets. However, CLIP and CLIP4Clip embed the whole sentence and image/video into textual and visual representations, thus lacking the ability to capture fine-grained interactions.To this end, some previous works [29 , 59 ] propose fine-grained contrastive frameworks, which consider the contrast between each word of the sentence and each frame of the video. Moreover, TACo [ 57 ] introduces tokenlevel and sentence-level loss to consider both fine-grained and coarse-grained contrast. Although they have shown promising advances on the VTR task, cross-modality semantic contrast still needs to be systematically explored.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| 