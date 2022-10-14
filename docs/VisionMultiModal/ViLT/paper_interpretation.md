
标题：ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision


术语: VLP （Vision-and-Language Pre-training的首字母），是一种有效方法





图1 的解释：


本文使用的方法和之前方法相比，多模态融合的时间都相等，但是其他时间，远远小于其他方法。而且我们的方法是第一个 其他时间小于多模态融合时间的方法（0.4ms < 15ms）


- 我们自己方法的其他时间是 0.4ms；多模态融合时间是15ms；总时间是15.4ms

- UNITER-Base 的其他时间是 75 + 810 = 885ms；多模态融合时间是15ms；总时间是900ms

- Pixel-BERT-R50 的其他时间是 45ms；多模态融合时间是15ms；总时间是60ms；


我们的方法之所以比之前的方法所用时间短，是因为我们的方法在图像编码分支没有使用 耗时的CNN-Backbone 或者 更耗时的RPN相关网络；仅仅使用了轻量的 Linear Embedding方法。

