---
title: 场景文本VQA简单综述（三）
tags: 场景VQA综述
categories: 论文阅读
cover: url(/img/stvqa.png)
abbrlink: 82fc4597
date: 2023-04-07 15:36:19
---

## 相关基本概念补充

### zero-shot

用于分类的任务，一次也不学习，在没有任何训练数据的情况下，通过利用类别的语义信息来完成分类任务。

### 预训练&微调

**预训练**：一般的操作都是在一个大型的数据集上（ImageNet）训练一个模型，然后使用该模型作为类似任务的初始化或者特征提取器。**预先训练的一个模型或者指预先训练模型**的过程为预训练

**微调**：将预训练过的模型作用于自己的数据集，并使参数适应自己数据集的过程。接使用之前保存下来的模型的参数来作为这一任务的初始化参数，然后在训练的过程中，依据结果不断进行一些修改。

### VLP（Vision Language Pre-train）

![](/img/vlp.jpg)

VE = Visual Embedding；TE = Text Embedding；MI = Modality Interaction

可以划分为四种

- 重轻轻：VSE、VSE++、SCAN
- 重重轻：CLIP
- 轻重重：ViLBERT、UNTER、Pixel-BERT
- 轻轻重：ViLT

## Latr

### 摘要

本文提出了一种新的用于场景文本视觉问答（STVQA）的多模态构架，称为LayoutAware Transformer（latr）。本文揭示了当文字布局丰富时语言模块的重要性。**提出一种只需要文本和空间线索的单目标预训练方案**。尽管存在领域上的差距，但在扫描文档上应用这种预训练方案与使用自然图像相比具有一定的优势。扫描的文档易于获取，文本密集，具有多种布局，通过将语言和布局信息联系在一起，帮助模型学习各种空间线索。与现有方法相比，我们的方法执行了无词汇解码，并且如图所示，其泛化能力远远超出了训练词汇。同时，Latr还可以提高OCR错误的鲁棒性（OCR错误时STVQA失败的常见原因）。利用VIT，不需要使用FasterRCNN。效果：TextVQA为+7.6%，ST-VQA为+10.8%，OCR-VQA为+4.0%。

![](/img/stvqa.png)

### 研究问题

1、**仅用文本信息回答问题**；

2、**用文本信息和空间布局信息可以回答**；

3、**用文本、空间布局、视觉特征可以回答**。

### 以前存在问题

上面那篇论文**TAP（text-aware pre-training）**存在的问题（*详情见场景文本VQA简单综述（二）*）：

1、**获取大量带有场景文本的自然图片困难**；

2、**获取到的文本比较稀疏**；

3、**在设计预训练目标函数时没有考虑到空间布局信息和语义特征的融合**。

### 贡献

1、**文本和布局信息**在STVQA问题中时很重要的，提出了**Layout-Aware预训练的方法以及网络架构**；

2、采用**扫描的pdf利于结合文本与布局信息**，在其中进行预训练有利于解决STVQA，即使两者之间的问题领域不同；

3、Latr不需要词汇表（用的T5），在训练词汇以外的情况下也表现也表现良好（之前的很差）；一定程度上可以克服OCR错误；

4、效果牛逼（SOTA）

### 模型结构

Latr主要由3个部分组成，首先，是一个只在文本上预先训练的语言模型（左边），将OCR tokens的边界框（bounding box）的空间嵌入与文档上的进一步布局感知（layout-aware）预训练结合使用。

#### 语言模型

本文的Latr基本完全基于T5。T5结构没有改变。在T5的预训练中使用了大量预训练数据，这其中的数据被称为C4。Common Crawl公开可用的网络档案来获得750GB清理过的英语文本数据的子集，他们称之为Colossal Clean Crawled  Corpus（C4）。C4上的预训练是通过去噪任务完成的，这是掩蔽语言建模（MLM）的变体。我们遵循实施并使用HuggingFace中的权重 。

![](/img/latr.png)

#### 二维空间 Embedding

局部信息的价值对于Transformer来说是很重要的。其关键思想是将文本的2-D位置信息预语言标识相关联合耦合。与文档中的单词不同，自然图像中的场景文本可能以任意形状和角度出现（例如，在手表表面）。因此，我们包括文本的高度和宽度，以标识阅读顺序。

如下图所示，给定OCR标记Oi的情况下，相关联的单词边界框可以由（xi0，yi0，xi1，yi1，hi，wi）定义，其中（xi0、yi0）对应于边界框的左上角的位置，（xi1、yi1）表示右下角的位置；（hi，wi）表示相对于阅读顺序的高度和宽度。为了嵌入边界框信息，我们使用了一个查找表，该查找表通常用于对one-hot表示进行连续编码（例如，PyTorch中的nn.Embedding）。

![](/img/latr2d.png)



#### Layout-Aware预训练

总结：只训练具有文本和空间线索的语言模态，以联合建模文本和布局信息之间的交互，进行大量的预训练。

由于T5仅在文本数据上进行训练，我们需要进一步的预训练，以有效地对齐布局信息（2-d空间嵌入）和语义表示。（本文直接在文档中预训练，而不需要img）。它们是各种复杂布局中的富文本环境的来源。我们执行了布局软件去噪预训练任务，其中包括二维空间嵌入，如图所示。这使得能够在预训练阶段使用没有答案注释的弱数据。与正常的去噪任务一样，我们的布局感知去噪任务掩盖了一系列标记，并迫使模型预测掩盖的跨度。与正常的去噪任务不同，我们还允许模型访问屏蔽令牌的粗略位置，这鼓励模型在完成此任务时充分利用布局信息 



## BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

(注：这个是传统vqa的，应该是现在最先进的模型，自带llm)

参考：https://zhuanlan.zhihu.com/p/606364639

BLIP-2：用冻结的图像编码器和大型语言模型进行语言-图像预训练的自举模型

![](/img/blip-2.png)

### 贡献

1、开放性的多模态内容理解与生成，为未来与llm结合打下基础。

2、以新的视角去看待vqa，引入llm，cv是传感器，llm是处理器

3、相对友好的计算资源。。。指16*A100

4、性能nb

### 相关工作

#### ViLBERT

20年 ViLBERT和Uniter采用了Object-Text对来提升模型对图片的理解能力。Object的引入，不可避免的需要一个笨重的检测器，去检测各种框，使得图像模态显得比较笨重。而且检测器模型不可避免的会存在漏检的问题，可以参考后来Open-Vocabulary一些工作，比如ViLD。这一阶段，显然对图像的理解是多模态的重头戏，文本更多是辅助图像任务的理解。

#### ViLT

ViLT，ALBEF，VLMo，BLIP 等等都抛弃了检测器，彻底摆脱了CNN网络的舒服，全面拥抱Transformer，当然这也得益于本身ViT模型在CV领域的大放光彩，让两个模态的有机融合成为了可能。在这一阶段，文本模态感觉已经可以和图像模态平起平坐了。从在各项具体下游任务（VQA、VG、ITR）的实际表现上来说，已经比较令人满意了。但总感觉差点味道，就是复杂推理。比如VQA上的问题，大多数是简单的逻辑计算或识别，感觉还不够智能。

#### BLIP2

图像输入图像编码器（Image Encoder），得到文本（Text）在Q-Former（BERT初始化）里进行融合，最后输入llm模型。

![](/img/blip-2-2.png)

第一阶段从冻结的图像编码器中引导视觉语言表示学习。第二阶段从冻结的LLM中引导视觉到语言的生成学习，它支持零镜头指示的图像到文本生成。

与之前的模型不同，大多关注vit和encoder有多牛逼，但是忽视result处理器的重要性。只有LLM模型，才能实现这一角色，统一起各个模态的信号，从一个宏观的角度去看待这个问题。

```
Powered by LLMs (e.g. OPT (Zhang et al., 2022), FlanT5 (Chung et al., 2022)), BLIP-2 can be prompted to perform zero-shot image-to-text generation that follows natural language instructions, which enables emerging capabilities such as visual knowledge reasoning, visual conversation, etc.
```

### 如何统一多模态表征（参考ALBEF）

模型与ALBEF相似，不同的是**learned Query**的引入（左下方那个），可以看到这些Query通过Cross-Attention与图像的特征交互，通过Self-Attention与文本的特征交互。这样做的好处有两个：（1）这些Query是基于两种模态信息得到的；（2）无论多大的视觉Backbone，最后都是Query长度的特征输出，大大降低了计算量。比如在实际实验中，ViT-L/14的模型的输出的特征是257x1024的大小，最后也是32x768的Query特征。

![](/img/blip-2-qformer.png)

针对Q-Former的三个训练任务：Image-Text Contrastive Learning (ITC)，Image-grounded Text Generation (ITG)，Image-Text Matching (ITM)。（*补充一下：latr：掩码语言见面（MLM）*）其中 ITC 和 ITM 任务，与ALBEF中的实现类似，只不过图像特征改为了Query的特征，具体可以参考代码实现（ITC和ITM）。**这里比较特别的是ITG任务，与ALBEF中的MLM不同，这里改成了生成整句Text的任务，类似Captioning，具体代码实现ITG**。实际上，这几个任务都是以Query特征和文本特征作为输入得到的，只不过有不同的Mask组合，具体可以参考上图中的右图。

第一阶段，模型训练由上述三个任务组成，主要完成**对于特征的提取和融合**，马上传入llm。



第二阶段，将Query变成llm认识的样子。

![](/img/blip-2-llm.png)

(上)单纯基于解码器的（冻结的）LLM（OPT）；基于编码器-解码器的（冻结的）LLM（FlanT5），全连接层从Q-former的输出适应所选LLM输入的大小。

1、比如前者Decoder（OPT）,采用Query作为输入，文本作为目标；

2、后者Encoder-Decoder（FlanT5）：以Query和一句话的前半段作为输入，后半段作为目标。

### 训练

1. 训练数据方面：包含常见的 COCO，VG，SBU，CC3M，CC12M 以及 115M的LAION400M中的图片。采用了BLIP中的CapFilt方法来Bootstrapping训练数据。
2. CV模型：选择了CLIP的ViT-L/14和ViT-G/14，特别的是，作者采用倒数第二层的特征作为输出。
3. LLM模型：选择了OPT和FlanT5的一些不同规模的模型。
4. 训练时，CV模型和LLM都是冻结的状态，并且参数都转为了FP16。这使得模型的计算量大幅度降低。主要训练的基于BERT-base初始化的Q-Former只有188M的参数量。
5. 最大的模型，ViT-G/14和FlanT5-XXL，只需要16卡A100 40G，训练6+3天就可以完成。
6. 所有的图片都被缩放到224x224的大小。

### 结果

作者用图片配合文字 prompt “a photo of”作为模型的输入。训练过程中冻结LLM，训练Q-Former和CV模型。可以看到，在域内数据集（COCO）上，其表现并没有非常亮眼，但在域外数据集NoCaps上，BLIP2显示出了强大的泛化能力，相较之前的模型有明显的提升。

下面这张图是**VQA微调模型**，其中LLM接收Q-Former的输出和问题作为输入，然后预测答案。我们还提供该问题作为Q-Former的条件，使得提取的图像特征与该问题更相关。训练的参数和IC任务一致，主要是Q-Former和ViT。不同的是，Q-Former和LLM都有Question作为文本输入。Q-Former的文本输入，保证了Query提取到的特征更加的精炼。

![](/img/blip-2-vqa.png)



图中有什么（ViT）+ 问的是什么（Q-Former，LLM）+ 找答案 （LLM）。

### Limitation

由于图文数据集大多数是一对一的匹配，所以很难让模型建立上下文的联系；

LLM模型本身局限决定的。

=======================================================================================







接下来的任务：将img2llm/blip2用于textvqa，看看接口怎么写

记得看看前面两篇论文的方法。



未完待续。。。

