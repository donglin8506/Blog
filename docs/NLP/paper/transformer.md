
论文名称: Attention Is All You Need

readpaper地址: https://readpaper.com/pdf-annotate/note?pdfId=602045685808148480&noteId=727322466937372672


作者介绍

作者顺序是随机的，作者的贡献是等同的。

| 作者 | 公司 | 贡献
| :--- | :--- | :---
| Ashish Vaswani | Google Brain | designed and implemented the first Transformer models and has been crucially(至关重要) involved in every aspect of this work.
| Noam Shazeer | Google Brain | proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail.
| Niki Parmar | Google Research | designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor.
| Jakob Uszkoreit | Google Research | proposed replacing RNNs with self-attention and started the effort to evaluate this idea.
| Llion Jones | Google Research | also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations.
| Aidan N. Gomez | University of Toronto | spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.
| Lukasz Kaiser | Google Brain | 同 Aidan N. Gomez
| Illia Polosukhin | Google Research | 同 Ashish Vaswani


## 摘要

The dominant(主导的) sequence transduction(转录) models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing(丢弃) with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

序列转录模型：输入一个序列，输出一个序列

## 引言

Recurrent neural networks, long shot-term memory[13] and gated recurrent[7] neural networks in particular, have been firmly as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation[35,2,5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures[38,24,15].

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [13] Long short-term memory | LSTM | 1997-11-01
| [7] Empirical evaluation of gated recurrent neural networks on sequence modeling | GRN | 2014-12-11
| [35] Sequence to Sequence Learning with Neural Networks | - | 2014-12-08
| [2] Neural Machine Translation by Jointly Learning to Align and Translate | - | 2014-01-01
| [5] Learning Phrase Representations using RNN Encoder -- Decoder for Statistical Machine Translation | - | 2014-01-01
| [38] Google's Neural Machine Translation System: Bridgingthe the Gap between Human and Machine Translation | - | 2016-09-26
| [24] Effective Approaches to Attention-based Neural Machine Translation | - | 2015-08-17
| [15] Exploring the limits of language modeling | - | 2016-02-07


Recurrent models typically(通常) factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_t$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.(序列很长的时候，为了使当前时刻能依然能获取很早的隐含特征，则需要将那么时刻的隐含特征都存储下来，则比较耗内存) Recent work has achieved significant improvements in computational efficiency through factorization tricks[21] and conditional computation[32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.


| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| Factorization tricks for LSTM networks | - | 2017-02-17

Attention mechanisms have become an integral(不可缺少的) part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences[2,19]. In all but a few cases(除少数情况外)[27], however, such attention mechanisms are used in conjunction with a recurrent network.


| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [2] Neural Machine Translation by Jointly Learning to Align and Translate | - | 2014-01-01
| [19] Structured Attention Networks | - | 2017-02-03


In this work we propose the Transformer, a model architecture eschewing(避免) recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.


## 背景

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU[16], ByteNet[18] and ConvS2S[9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positioins. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant(遥远的) positions[12]. In the Transformer this is reduced to a constant number of operations, albeit(尽管) at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [16] Can active memory replace attention? | - | 2016
| [18] Neural Machine Translation in Linear Time | - | 2016-10-31
| [9] Convolutional Sequence to Sequence Learning | - | 2017-05-08
| [12] Gradient Flow in Recurrent Nets: the Difficulty of Learning Long-Term Dependencies | - | 2001-01-01


Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations[4,27,28,22].

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [4] Long Short-Term Memory-Networks for Machine Reading | - | 2016-01-25
| [28] A Deep Reinforced Model for Abstractive Summarization | - | 2017-05-11
| [22] A Structured Self-Attention Sentence Embedding | - | - 

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks.[34]


| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [34] End-To-End Memory Networks | - | -


To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17,18] and [9].

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [17] Neural GPUs Learn Algorithms | - | 2015-11-25
| [18] Neural Machine Translation in Linear Time | - | 2016-10-31
| [9] Convolutional Sequence to Sequence Learning | - | 2017-05-08


## 模型结构

Most competitive neural sequence transduction models have an encoder-decoder structure[5,2,35]. Here, the encoder maps an input sequence of symbol representations ( $x_1,..,x_n$ ) to a sequence of continuous representations ( $z=(z_1, ..., z_n)$ ). Given $z$, the decoder then generates an output sequence ( $y_1, ..., y_m$ ) of symbols one element at a time. At each step the model is auto-regressive[10], consuming the previously generated symbols as additional input when generating the next.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [5] Learning Phrase Representations using RNN Encoder--Decoder for Statistical Machine Translation | - | 2014-01-01
| [2] Neural Machine Translation by Jointy Learning to Align and Translate | - | 2014-01-01
| [35] Sequence to Sequence Learning with Neural Networks | - | 2014-12-08
| [10] Generating Sequences With Recurrent Neural Networks | - | 2013-08-04


The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

#### **3.1 Encoder and Decoder Stacks**

**Encoder:** The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network(一个简单的MLP层). We employ a residual connection[11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm($x$ + Sublayer($x$)), where Sublayer($x$) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{model}=512$.


| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [11] Deep Residual Learning for Image Recognition | ResNet | 2015-12-10
| [1] Layer Normalization | - | 2016-07-21

什么是LN层？



**Decoder:** The decoder is also composed of a stack of $N=6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequence positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

#### **3.2 Attention**


An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.


###### **3.2.1 Scaled Dot-Product Attention**

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$. We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously(同时), packed together into a matrix $Q$. The keys and values are also packed together into matrices $K$ and $V$. We compute the matrix of outputs as:

$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

The two most commonly used attention functions are additive attention[2], and dot-product (multi-plicative 乘法的) attention. Dot-production attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_k}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| Neural Machine Translation by Jointly Learning Align and Translate | - | 2014-01-01


While for small values of $d_k$ the two mechanisms perform similarily, additive attention outperforms dot product attention without scaling for larger values of $d_k$[3]. We suspect that for large values of $d_K$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [3] Massive Exploration of Neural Machine Translation Architectures | - | 2017-03-11

###### **3.2.2 Multi-Head Attention**

Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to $d_q$, $d_k$, $d_v$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.


Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averagin inhibits this.

$$
MultiHead(Q,K,V)=Concat(head_1, ..., head_h)W^O
$$
$$
head_i = Attention(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})
$$

Where the projections are parameter matrices $W_{i}^{Q} \in R^{d_{model} \times d_k}$, $W_{i}^{K} \in R^{d_{model} \times d_k}$, $W_{i}^{V} \in R^{d_{model} \times d_v}$ and $W^O \in R^{hd_v \times d_{model}}$.

In this work we employ $h=8$ parallel attention layers, or heads. For each of these we use $d_k=d_v=d_{model}/h=64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with fully dimensionality.

###### **3.2.3 Applications of Attention in our Model**

The Transformer uses multi-head attention in three different ways:

- In "encoder-decoder attenion" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics(模仿) the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38,2,9].
- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections. 

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [38] Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation | - | 2016-09-26
| [2] Neural Machine Translation by Jointly Learning to Align and Translate | - | 2014-01-01
| [9] Convolutional sequence to sequence learning. | - | 2017

#### **3.3 Position-wise Feed-Forward Networks**

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

$$
FFN(x) = max(0, xW_1+b_1)W_2+b_2
$$

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is $d_{model}=512$, and the inner-layer has dimensionality $d_{ff}=2048$.


每个MLP单独作用到一个向量上；不同向量上的MLP共享参数；

由于向量在自注意力层中已经融合了所有输入向量的特征，包括其中的时序信息，所以可以进行单独的MLP操作。

层与层之间的MLP是不同的，同一层不同位置的MLP是相同的。

MLP中有一个隐藏层，维度是2048，输入是512维度，输出是512维度。

#### **3.4 Embeddings and Softmax**

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probablities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$.

#### **3.5 Positional Encoding**

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject(注入) some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed[9].

| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [9] Convolutional Sequence to Sequence Learning | - | 2017-05-08

In this work, we use sine and cosine functions of different frequencies:

$$
PE_{(pos, 2i)}=sin(pos/10000^{2i/d_{model}})
$$

$$
PE_{pos, 2i+1}=cos(pos/10000^{2i/d_{model}})
$$

where $pos$ is the position and $i$ is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression(一个几何级数) from $2\pi$ to $1000 * 2\pi$. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed $k$,$PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.

We also experimented with using learned positional embeddings[9] instead, and found that the two versions produced nearly identical results(see Table 3 row E). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.



| 论文名称 | 论文别名 | 论文时间
| :------- | :------ | :--------
| [9] Convolutional Sequence to Sequence Learning | - | 2017-05-08




## 结论

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly(显著地) faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential(使生成不那么时序化) is another research goals of ours.