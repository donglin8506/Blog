## 1. Learning Transferable Visual Models From Natural Language Supervision

### 1.1 标题：

从自然语言监督中学习可迁移的视觉模型

### 1.2 摘要：

#### 1.2.1 原文

State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at https://github.com/OpenAI/CLIP.

#### 1.2.2 原文翻译

最先进的计算机视觉系统经过训练，可以预测一组预先确定的物体类别。这种受限制的监督形式限制了它们的一般性和可用性，因为需要额外的标记数据来指定任何其他可视概念。直接从原始文本中学习图像是一个很有前途的选择，它利用了更广泛的监督来源。我们证明了预测哪个标题与哪个图像搭配的简单预训练任务是一种有效且可扩展的方法，可以在从互联网收集的 4 亿（图像、文本）对的数据集上从头开始学习 SOTA 图像表示。在预训练之后，使用自然语言来参考学习的视觉概念（或描述新的概念），使模型能够零样本转移到下游任务。我们通过对 30 多个不同的现有计算机视觉数据集进行基准测试来研究这种方法的性能，这些数据集涵盖 OCR、视频中的动作识别、地理定位和许多类型的细粒度对象分类等任务。该模型非常重要地转移到大多数任务，并且通常与完全监督的基线相比具有竞争力，而无需任何数据集特定的训练。例如，我们在 ImageNet 零样本上匹配原始 ResNet-50 的准确性，而无需使用它所训练的 128 万个训练示例中的任何一个。我们发布我们的代码和预训练的模型权重: https://github.com/OpenAI/CLIP

#### 1.2.3 要点总结

- 使用海量的图片-文本对，来进行模型预训练
- 可以在下游任务上进行零样本推理
- 文本信号作为图片表征的监督信号
  

### 1.3 引言

#### 1.3.1 原文 - 段落1
Pre-training methods which learn directly from raw text have revolutionized NLP over the last few years . (Dai &Le, 2015; Peters er al., 2018; Howard & Ruder, 2018; Radford et at., 2018; Devlin et al., 2018; Raffel et al., 2019). Task-agnostic objectives such as autoregressive and masked language modeling have scaled across many orders of magnitude in compute, model capacity, and data, steadily improving capabilities. The development of "text-to-text" as a standardized input-output interface (McCann et al., 2018; Radford et al., 2019; Raffel et al., 2019) has enabled taskagnostic architectures to zero-shot transfer to downstream datasets removing the need for specialized output heads or dataset specific customization.Flagship systems like GPT-3 (Brown et al., 2020) are now competitive across many tasks with bespoke models while requiring little to no dataset specific training data.

#### 1.3.2 原文翻译 - 段落1
直接从原始文本中学习的预训练方法在过去几年中彻底改变了 NLP。（Dai &Le, 2015; Peters er al., 2018; Howard & Ruder, 2018; Radford et at., 2018; Devlin et al., 2018; Raffel et al., 2019）自回归和掩码语言建模等与任务无关的目标在计算、模型容量和数据方面已经扩展了多个数量级，从而稳步提高了能力。“文本到文本”作为标准化输入输出接口的发展（McCann et al., 2018; Radford er al, 2019; Raffel et al., 2019）使任务无关架构能够零样本传输到下游数据集消除了对专门输出头或数据集特定定制的需要。像 GPT-3（Brown 等人，2020）这样的旗舰系统现在在使用定制模型的许多任务中具有竞争力，同时几乎不需要特定于数据集的训练数据。



#### 1.3.3 引用论文简介 - 段落1

| 论文名称 | 标题翻译 | 论文别名 | 论文时间 | 论文简介 
| :------- | :------- | :------ | :-------- | :--------
| Semi-supervised Sequence Learning                                               | 基于半监督的序列学习 | - | Dai &Le, 2015 | 使用无监督的训练方法改进有监督时序学习性能
| Deep contextualized word representations                                        | 深度上下文的词表达 | - | Peters er al., 2018 | 我们引入了一种新型的深度上下文化词表示，它可以模拟（1）词使用的复杂特征（例如，语法和语义），以及（2）这些使用如何在语言上下文中变化（即，对多义词建模）。我们的词向量是深度双向语言模型 (biLM) 内部状态的学习函数，该模型在大型文本语料库上进行了预训练。我们表明，这些表示可以很容易地添加到现有模型中，并显着改善六个具有挑战性的 NLP 问题的最新技术，包括问答、文本蕴涵和情感分析。我们还提出了一项分析，表明暴露预训练网络的深层内部结构至关重要，允许下游模型混合不同类型的半监督信号。
| Universal Language Model Fine-tuning for Text Classification                    | 文本分类的通用语言模型微调方法 | ULMFiT | Howard & Ruder, 2018 | 归纳迁移学习极大地影响了计算机视觉，但 NLP 中的现有方法仍然需要针对特定​​任务的修改和从头开始训练。我们提出了通用语言模型微调 (ULMFiT)，这是一种有效的迁移学习方法，可应用于 NLP 中的任何任务，并介绍了微调语言模型的关键技术。我们的方法在六个文本分类任务上显着优于最先进的方法，在大多数数据集上将错误降低了 18-24%。此外，只有 100 个标记示例，它与从头开始训练 100 倍的数据的性能相匹配。我们开源我们的预训练模型和代码。
| Improving Language Understanding by Generative Pre-Training                     | 通过生成式的预训练方式改善语言理解能力 | GPT | Radford et at., 2018 | 自然语言理解包括范围广泛的不同任务，例如文本蕴涵、问答、语义相似性评估和文档分类。尽管大量未标记的文本语料库很丰富，但用于学习这些特定任务的标记数据却很少，这使得经过判别训练的模型充分执行具有挑战性。我们证明，通过在各种未标记文本的语料库上对语言模型进行生成式预训练，然后对每个特定任务进行区分性微调，可以实现这些任务的巨大收益。与以前的方法相比，我们在微调期间利用任务感知输入转换来实现有效传输，同时需要对模型架构进行最少的更改。我们证明了我们的方法在自然语言理解的广泛基准上的有效性。我们的通用任务不可知模型优于使用专门为每个任务设计的架构的判别训练模型，在所研究的 12 个任务中的 9 个中显着提高了现有技术。例如，我们在常识推理（Stories Cloze Test）上实现了 8.9% 的绝对改进，在问答（RACE）上实现了 5.7% 的绝对改进，在文本蕴涵（MultiNLI）上实现了 1.5% 的绝对改进。
| BERT: Pre-training of Deep Bidirectional Transformer for Language Understanding   | 用于语言理解的深度双向T预训练方法 | BERT | Devlin et al., 2018 | 我们引入了一种新的语言表示模型，称为 BERT，它代表来自Transformers 的双向编码器表示。与最近的语言表示模型 (Peters et al., 2018a; Rad-ford et al., 2018) 不同，BERT 旨在通过联合调节所有层的左右上下文来预训练来自未标记文本的深度双向表示。因此，预训练的 BERT 模型可以通过一个额外的输出层进行微调，从而为各种任务（例如问答和语言推理）创建最先进的模型，而无需大量特定任务架构修改。BERT 在概念上简单且经验丰富。它在 11 个自然语言处理任务上获得了新的 state-of-the-art 结果，包括将 GLUE 分数推至 80.5%（绝对提高 7.7%），MultiNLI 准确率达到 86.7%（绝对提高 4.6%），SQuAD v1.1。 1 个问答测试 F1 至 93.2（1.5 分绝对提高）和 SQuAD v2.0 测试 F1 至 83.1（5.1 分绝对提高）。
| Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | 使用一个统一的文本到文本的T来探索迁移学习的限制 | - | Raffel et al., 2019 | 迁移学习，其中模型首先在数据丰富的任务上进行预训练，然后在下游任务上进行微调，已成为自然语言处理 (NLP) 中的一项强大技术。迁移学习的有效性带来了方法、方法和实践的多样性。在本文中，我们通过引入一个将所有基于文本的语言问题转换为文本到文本格式的统一框架来探索 NLP 迁移学习技术的前景。我们的系统研究比较了数十种语言理解任务的预训练目标、架构、未标记数据集、迁移方法和其他因素。通过将我们探索的见解与规模和我们新的“巨大的清洁爬网语料库”相结合，我们在许多基准上取得了最先进的结果，包括摘要、问答、文本分类等。为了促进 NLP 迁移学习的未来工作，我们发布了我们的数据集、预训练模型和代码。
| The Natural Language Decathlon: Multitask Learning as Question Answering          | 自然语言十项全能：多任务学习当作问答          | MQAN | McCann et al., 2018 | 深度学习单独提高了许多自然语言处理 (NLP) 任务的性能。然而，一般的 NLP 模型不能出现在专注于单个指标、数据集和任务的特殊性的范式中。我们介绍了自然语言十项全能 (decaNLP)，这是一项跨越十项任务的挑战：问答、机器翻译、摘要、自然语言推理、情感分析、语义角色标签、关系提取、面向目标的对话、语义解析和常识代词解析度。我们将所有任务都视为对上下文的问答。此外，我们提出了一种新的多任务问答网络（MQAN），它可以联合学习 decaNLP 中的所有任务，而无需任何特定于任务的模块或参数。 MQAN 展示了机器翻译和命名实体识别的迁移学习、情感分析和自然语言推理的域适应以及文本分类的零样本能力的改进。我们证明了 MQAN 的多指针生成器解码器是这一成功的关键，并且通过反课程培训策略进一步提高了性能。虽然是为 decaNLP 设计的，但 MQAN 在单任务设置中的 WikiSQL 语义解析任务上也取得了最先进的结果。我们还发布了用于采购和处理数据、训练和评估模型以及重现 decaNLP 的所有实验的代码。
| Language Models are Unsupervised Multitask Learners                               | 语言模型是无监督的多任务学习器              | GPT-2 | Radford er al, 2019 | 自然语言处理任务，例如问答、机器翻译、阅读理解和摘要，通常通过对特定任务数据集的监督学习来处理。我们证明，当在一个名为 WebText 的包含数百万个网页的新数据集上进行训练时，语言模型开始在没有任何明确监督的情况下学习这些任务。当以文档和问题为条件时，语言模型生成的答案在 CoQA 数据集上达到 55 F1 - 在不使用 127,000 多个训练示例的情况下，匹配或超过 4 个基准系统中的 3 个的性能。语言模型的容量对于零样本任务转移的成功至关重要，并且增加它以对数线性方式跨任务提高性能。我们最大的模型 GPT-2 是一个 1.5B 参数的 Transformer，它在 8 个测试语言建模数据集中的 7 个在零样本设置中实现了最先进的结果，但仍然不适合 WebText。模型中的样本反映了这些改进并包含连贯的文本段落。这些发现为构建语言处理系统提供了一条有希望的途径，该系统从自然发生的演示中学习执行任务。
| Language Models are Few-Shot Learners                                             | 语言模型是小样本学习器                     | GPT-3 | Brown et al., 2020 | 最近的工作通过对大型文本语料库进行预训练，然后对特定任务进行微调，证明了许多 NLP 任务和基准测试的显着进步。虽然在架构中通常与任务无关，但这种方法仍然需要数千或数万个示例的特定于任务的微调数据集。相比之下，人类通常可以仅通过几个示例或简单指令执行新的语言任务——目前的 NLP 系统在很大程度上仍然难以做到这一点。在这里，我们展示了扩展语言模型极大地提高了与任务无关的、少样本的性能，有时甚至达到了与先前最先进的微调方法的竞争力。具体来说，我们训练了 GPT-3，这是一种具有 1750 亿个参数的自回归语言模型，比以前的任何非稀疏语言模型多 10 倍，并在少样本设置中测试其性能。对于所有任务，GPT-3 无需任何梯度更新或微调即可应用，任务和小样本演示纯粹通过与模型的文本交互来指定。 GPT-3 在许多 NLP 数据集上实现了强大的性能，包括翻译、问答和完形填空任务，以及一些需要即时推理或领域适应的任务，例如解读单词，在句子，或执行 3 位算术。同时，我们还确定了 GPT-3 的小样本学习仍然困难的一些数据集，以及 GPT-3 面临与大型网络语料库训练相关的方法问题的一些数据集。最后，我们发现 GPT-3 可以生成人类评估者难以将其与人类撰写的文章区分开来的新闻文章样本。我们总体上讨论了这一发现和 GPT-3 的更广泛的社会影响。

#### 1.3.4 原文 - 段落2

These results suggest that the aggregate supervision accessible to modern pre-training methods within web-scale collections of text surpasses that of high-quality crowd-labeled NLP datasets. However, in other fields such as computer vision it is still standard practice to pre-train models on crowd-labeled datasets such as ImageNet (Deng et al., 2009). Could scalable pre-training methods which learn directly from web text result in a similar breakthrough in computer vision? Prior work is encouraging.

#### 1.3.5 原文翻译 - 段落2

这些结果表明，在网络规模的文本集合中，现代预训练方法可访问的聚合监督超过了高质量的人群标记的 NLP 数据集。然而，在计算机视觉等其他领域，在 ImageNet 等人群标记的数据集上预训练模型仍然是标准做法（Deng et al., 2009）。直接从网络文本中学习的可扩展预训练方法能否在计算机视觉领域取得类似的突破？先前的工作令人鼓舞。

#### 1.3.6 引用论文简介 - 段落2

| 论文名称 | 标题翻译 | 论文别名 | 论文时间 | 论文简介 
| :------- | :------- | :------ | :-------- | :--------
| ImageNet: A large-scale hierarchical image database | 大规模分层图片数据库 | ImageNet | Deng et al., 2009 | 互联网上图像数据的爆炸式增长有可能培育出更复杂、更强大的模型和算法来索引、检索、组织图像和多媒体数据并与之交互。但究竟如何利用和组织这些数据仍然是一个关键问题。我们在这里介绍一个名为“ImageNet”的新数据库，这是一个建立在 WordNet 结构主干上的大规模图像本体。 ImageNet 旨在用平均 500-1000 张干净和全分辨率的图像填充 WordNet 的 80,000 个同义词集中的大多数。这将导致由 WordNet 的语义层次组织的数千万张带注释的图像。本文详细分析了 ImageNet 当前状态：12 个子树，5247 个同义词集，总共 320 万张图像。我们表明，ImageNet 在规模和多样性上比当前的图像数据集要大得多，而且准确得多。构建如此大规模的数据库是一项具有挑战性的任务。我们使用 Amazon Mechanical Turk 描述数据收集方案。最后，我们通过对象识别、图像分类和自动对象聚类三个简单的应用来说明 ImageNet 的有用性。我们希望 ImageNet 的规模、准确性、多样性和层次结构能够为计算机视觉社区及其他领域的研究人员提供无与伦比的机会。

#### 1.3.7 原文 - 段落3

Over 20 years ago Mori et al. (1999) explored improving content based image retrieval by training a model to predict the nouns and adjectives in text documents paired with images. Quattoni et al. (2007) demonstrated it was possible to learn more data efficient image representations via manifold learning in the weight space of classifiers trained to predict words in captions associated with images. Srivastava & Salakhutdinov (2012) explored deep representation learning by training multimodal Deep Boltzmann Machines on top of low-level image and text tag features. Joulin et al.(2016) modernized this line of work and demonstrated that CNNs trained to predict words in image captions learn useful image representations. They converted the title, description, and hashtag metadata of images in the YFCC100M dataset (Thomee et al., 2016) into a bag-ofwords multi-label classification task and showed that pretraining AlexNet (Krizhevsky et al., 2012) to predict these labels learned representations which preformed similarly to ImageNet-based pre-training on transfer tasks. Li et al.(2017) then extended this approach to predicting phrase n-grams in addition to individual words and demonstrated the ability of their system to zero-shot transfer to other image classification datasets by scoring target classes based on their dictionary of learned visual n-grams and predicting the one with the highest score. Adopting more recent architecture and pre-training approaches, VirTex(Desai & Johnson,2020), ICMLM(Bulent Sariyildiz et al., 2020), and ConVIRT(Zhang et al., 2020) have recently demonstrated the potential of transformer-based language modeling, masked language modeling, and contrastive objectives to learn image representations from text.


#### 1.3.8 原文翻译 - 段落3
20 多年前 Mori et al. (1999) 通过训练模型来预测与图像配对的文本文档中的名词和形容词，探索了改进基于内容的图像检索。Quattoni et al. (2007)证明可以通过流形学习在训练用于预测与图像相关的字幕中的单词的分类器的权重空间中学习更多数据有效的图像表示。Srivastava & Salakhutdinov (2012) 通过在低级图像和文本标签特征之上训练多模态深度玻尔兹曼机来探索深度表示学习。Joulin et al.(2016)现代化了这一系列工作，并且证明了通过预测图片描述中单词的这种方式可以学习到有用的图片表征。他们将 YFCC100M 数据集 (Thomee et al., 2016) 中图像的标题、描述和主题标签元数据转换为词袋多标签分类任务，并表明预训练 AlexNet (Krizhevsky et al., 2012) 可以预测这些标记学习到的表示，这个实现过程就像使用ImageNet预训练模型用到其他下游任务上。Li et al.(2017) 将上述方法扩展到预测n-grams短语上，不仅限于单个词，这也表明这样的系统具备零样本迁移到其他图片分类数据集上的能力，在这些分类任务中，是通过给所学到的视觉n-grams的词典来给目标类别打分，然后预测得分最大的那个类别。有几个工作，例如VirTex(Desai & Johnson,2020), ICMLM(Bulent Sariyildiz et al., 2020), and ConVIRT(Zhang et al., 2020)，在之前工作的基础上采用最新的网络结构和预训练方法，结果展现了基于transformer的语言模型、掩码语言模型、对比损失等在从文本中学习图片表征的能力。

#### 1.3.9 引用论文简介 - 段落3
| 论文名称 | 标题翻译 | 论文别名 | 论文时间 | 论文简介 
| :------- | :------- | :------ | :-------- | :--------
| Image-to-word transformation based on dividing and vector quantizing images with words | 依据单词将图片分割并向量化，来实现图片到文本的转换 | - | Mori et al. (1999) | 我们提出了一种在图像和文字之间建立关系的方法。我们在方法中采用了两个过程，一个是将每个图像均匀地划分为带有关键字的子图像的过程，另一个是对子图像进行矢量量化的过程。这些过程导致的结果表明，每个子图像可以与一组单词相关联，其中每个单词都是从分配给整个图像的单词中选择的。该方法的原始方面是，（1）分配给整个图像的所有单词都被继承到每个分割的子图像，（2）一组分割图像的每个单词的投票概率由矢量量化的结果估计子图像的特征向量。一些实验表明了所提出方法的有效性。
| Learning Visual Representations using Images with Captions | 学习有字幕图片的视觉表征 | - | Quattoni et al. (2007) | 当前学习视觉类别的方法在有大量标记数据可用时效果很好，但当标记示例的数量很少时可能会遇到严重困难。当标记数据稀缺时，使用未标记数据来学习低维图像表示可能是有益的，但仍然可以捕获区分图像类别所需的信息。本文描述了一种从大量具有相关标题的未标记图像中学习表示的方法；目标是改进未来图像分类问题的学习。实验表明，我们的方法显着优于 (1) 完全监督的基线模型，(2) 忽略字幕并通过仅对未标记图像执行 PCA 来学习视觉表示的模型，以及 (3) 使用输出的模型使用字幕和未标记数据训练的词分类器。我们目前的工作集中于将字幕作为元数据的来源，但更一般地，可以使用其他类型的元数据。
| Multimodal Learning with Deep Boltzmann Machines | 使用玻尔兹曼机做多模态学习 | - | Srivastava & Salakhutdinov (2012) | 描述了一种深度玻尔兹曼机，用于学习由多种输入模式组成的数据生成模型。该模型可用于提取将模态融合在一起的统一表示。我们发现这种表示对于分类和信息检索任务很有用。该模型通过学习多模式输入空间上的概率密度来工作。它使用潜在变量的状态作为输入的表示。即使在某些模态不存在的情况下，该模型也可以通过从它们的条件分布中采样并填充它们来提取这种表示。我们在由图像和文本组成的双模态数据上的实验结果表明，多模态 DBM 可以学习一个良好的生成模型图像和文本输入的联合空间，可用于从单模式和多模式查询中检索信息。我们进一步证明，该模型在判别性任务上明显优于 SVM 和 LDA。最后，我们将我们的模型与其他深度学习方法进行比较，包括自动编码器和深度信念网络，并表明它取得了显着的收益。
| Learning Visual Features from Large Weakly Supervised Data | 从大规模的弱监督数据中学习视觉特征 | - | Joulin et al.(2016) | 在大型监督数据集上训练的卷积网络产生的视觉特征构成了许多计算机视觉问题中最先进技术的基础。这些视觉特征的进一步改进可能需要更大的手动标记数据集，这严重限制了取得进展的速度。在本文中，我们探索了利用大量、弱标记的图像集合来学习良好视觉特征的潜力。我们在包含 1 亿张 Flickr 照片和字幕的数据集上训练卷积网络，并表明这些网络产生的特征在一系列视觉问题中表现良好。我们还表明，网络可以适当地捕获单词相似度，并学习不同语言之间的对应关系。
| YFCC100M: The New Data in Multimedia Research | 用于多媒体研究的新数据集 | YFCC100M | (Thomee et al., 2016) | 我们展示了 Yahoo Flickr 知识共享 1 亿数据集 (YFCC100M)，这是迄今为止发布的最大的公共多媒体集合。该数据集共包含 1 亿个媒体对象，其中照片约 9920 万个，视频约 80 万个，均带有知识共享许可。数据集中的每个媒体对象都由几条元数据表示，例如Flickr 标识符、所有者名称、相机、标题、标签、地理位置、媒体来源。该集合提供了从 Flickr 于 2004 年成立到 2014 年初这些年来照片和视频是如何拍摄、描述和共享的全面快照。在本文中，我们解释了其创建背后的基本原理，以及数据集的含义用于科学、研究、工程和开发。我们进一步提出了多媒体研究中的几个新挑战，现在可以通过我们的数据集进行扩展。
| ImageNet Classification with Deep Convolutional Neural Networks | 使用卷积神经网络来做ImageNet分类 | AlexNet | (Krizhevsky et al., 2012) | 我们训练了一个大型的深度卷积神经网络，将 ImageNet LSVRC-2010 竞赛中的 120 万张高分辨率图像分类为 1000 个不同的类别。在测试数据上，我们实现了 37.5% 和 17.0% 的 top-1 和 top-5 错误率，这比之前的最新技术要好得多。该神经网络有 6000 万个参数和 650,000 个神经元，由五个卷积层组成，其中一些卷积层的后面增加了最大池化层，三个全连接层和最终的 1000 路 softmax。为了加快训练速度，我们使用了非饱和神经元和卷积运算的非常高效的 GPU 实现。为了克服全连接层中的过拟合问题，我们采用了最近开发的称为 dropout 的正则化方法，该方法被证明非常有效。我们还在 ILSVRC-2012 竞赛中使用了该模型的一个变体，并获得了 15.3% 的前 5 名测试错误率，而第二名的测试错误率为 26.2%。
| Learning Visual N-Grams from Web Data | 从网页数据中学习视觉n-grams | - | Li et al.(2017) | 现实世界的图像识别系统需要识别数以万计的类别，这些类别构成了大量的视觉概念。在这种情况下，为训练每类注释数千张图像的传统方法是不可行的，这促使使用网络监督数据。本文探讨了图像识别系统在大量图像和相关用户评论上的训练。特别是，我们开发了可以预测与图像内容相关的任意短语的视觉 n-gram 模型。我们的视觉 n-gram 模型是前馈卷积网络，使用受语言建模中常用的 n-gram 模型启发的新损失函数进行训练。我们展示了我们的模型在短语预测、基于短语的图像检索、关联图像和字幕以及零样本迁移学习方面的优点。
| VirTex: Learning Visual Representations from Textual Annotations | 从文本注释中学习视觉表征 | VirTex | VirTex(Desai & Johnson,2020) | 许多视觉任务基本上都是从预训练的视觉表示开始的，这些视觉表征通常就是在 ImageNet数据集上以有监督的方式训练得到的。最近已经有方法将无监督预训练的方式扩展到大规模无标签图像上。相比之下，我们的目标是从更少的图像中学习高质量的视觉表示。为此，我们重新审视有监督的预训练，并寻找基于分类预训练的更高效利用数据替代方案。我们提出了 VirTex——一种使用语义密集的字幕来学习视觉表示的预训练方法。我们在 COCO Captions 上从头开始训练卷积网络，并将它们转移到下游识别任务，包括图像分类、对象检测和实例分割。在所有任务中，VirTex 产生的特征都匹配或超过在 ImageNet 上学习到的特征——监督或非监督——尽管使用的图像少了十倍。
| Learning Visual Representations with Caption Annotations | 从字幕中学习是视觉表征 | ICMLM | ICMLM(Bulent Sariyildiz et al., 2020) | 预训练通用视觉特征已成为处理许多计算机视觉任务的关键部分。虽然人们可以在大规模有标签的 ImageNet 数据集上学习这些特征，但最近的方法已经研究了如何使用有噪声的、规模更小的甚至没有标签的数据集来进行这种预训练任务。有字幕的图片是比较容易获取的，基于此，我们认为可以充分利用这种被忽视的信息源来监督视觉表征的训练。受到语言模型最近的进展影响，我们提出了ICMLM(image-conditioned masked language modeling)，这是一个在图片-文本对上学习视觉表征的代理任务。ICMLM 在于依靠视觉线索来预测字幕中的掩蔽词。为了解决这个任务，我们提出了带有专用的视觉和文本编码器，我们表明，在解决这个任务过程中所学习到的视觉表征可以较好地迁移到其他下游任务上。我们的实验证实，可以利用图像说明将全局和局部语义信息注入视觉表示。


