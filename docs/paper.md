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
