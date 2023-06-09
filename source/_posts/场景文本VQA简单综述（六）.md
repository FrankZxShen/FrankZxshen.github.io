---
title: 场景文本VQA简单综述（六）
tags: 场景VQA综述
categories: 论文阅读
cover: url(/img/stvqa.png)
abbrlink: 3f41d26c
date: 2023-04-13 19:31:09
---

## 相关基本概念补充

### model-agnostic模型无关

与其说是一个深度学习模型，不如说是一个框架。



## Grounding Language Models to Images for Multimodal Generation

基于语言模型的图像多模态生成

### 核心思想

可以本文的方法引导冻结的大语言模型llm，处理和输出任意交错的多模态数据。**本文通过一个冻结预训练的llm和一个冻结预训练的视觉编码器（相当于没有预训练过程），用多任务目标训练图像字幕和图像文本检索**。对于前者，本文从视觉编码器中提取视觉embedding；对于后者，本文训练语言模型学习一个表示图像的新【RET】标记，并通过对比学习将标题的【RET】嵌入映射为接近其配对图像的视觉嵌入，部分模型都是冻结的，我们只在训练期间更新线性层的权重和【RET】标记嵌入。因此，我们提出的方法是非常节约计算和内存效率。

模型的能力：保留纯文本llm生成文本能力的同时，获得新的多模态对话和推理能力。本文提出的方法是**模型无关（model-agnostic）**的，可以用于未来更强大的llm。

### 贡献

1、提出用于自回归（*GPT*）生成的多模态数据冻结检索（FROMAGe），通过图像字幕和对比学习的冻结的视觉、llm模型。这种方法可以仅从图像描述对中学习到强大的few-shot多模态能力。

2、自回归的llm可以执行文本到图像的检索，对输入文本具有更高的敏感性。与现有模型相比，在长而复杂的自由格式文本上更加精确。

3、预训练的纯文本llm现有的功能，例如上下文学习、输入敏感性以及对话生成，可以用于基于视觉的任务。本文证明：（1）给定交错的图像和文本序列的上下文图像检索（2）视觉对话的zero-shot的强性能（3）提高了图像检索对话上下文的敏感性。

本文为**预训练的纯文本llm在视觉基础任务上的能力**提供了进一步的见解。

### 本文方法

本文的方法集成了语言模型和可视化模型，同时保持它们的参数不变。将图像投射到文本空间（参数转化为线性层），将文本embedding到视觉空间。保持模型冻结的动机是利用从大规模预训练中学到llm的能力。能够更好地泛化到few-shot和zero-shot设置。

本文方法讲究一个多任务：**图像字幕+图像文本检索**

先贴个图：

![](/img/FROMAGE.png)



本文采用的llm是OPT6.7B、视觉encoder是CLIP的ViT-L/14。



## Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answerin

基于知识的视觉问答中用答案启发式提示大型语言模型 

### 摘要

基于知识的视觉问答（VQA）需要图像之外的外部知识来回答问题。最近的一些工作试图使用llm（GPT-3）作为隐式知识引擎来获取问答所需要的知识。但本文认为，优于提供的输入信息不足，他们并没有完全激活GPT-3的能力。本文提出的Prophet，这是一个概念上的简单框架，旨在为基于知识的VQA提供给GPT-3答案启发。流程：在没有外部知识的情况下，在基于特定知识的VQA数据集上训练一个普通的VQA模型，之后可以从模型中提取两种类型的互补答案启发式：候选答案和感知答案的例子。最后，两种类型的答案启发式被编码到提示中，以使GPT-3能够更好地理解任务，从而提高其能力。Prophet在两个具有挑战性的基于知识的VQA数据集（OK-VQA和A-OKVQA）上显著优于所有现有的最先进方法，在其测试集上的准确率分别为61.1%和55.7%。

![](/img/prophet.png)

使用GPT-3模型的三个基于知识的VQA框架比较。PICa、KAT直接将标题（C）和问题（Q）输入GPT-3作为提示，但本文认为这样给GPT-3提供的信息其实是不足的。**本文的方法在没有外部知识的情况下学习了一个普通的VQA模型来产生答案启发式**，这为GPT-3提供了更丰富、更具体的任务信息来进行答案预测。 

在本文中，基于知识的VQA的一个简单解决方案是从显式知识库中检索知识条目，但是存在两个问题：**所需的知识可能无法从知识库中成功检索；即使检索到所需的知识，也无可避免的引入大量不相关的知识。**阻碍VQA的学习。

其中PICa是一项开创性的工作，采用了冻结的GPT-3，以格式化的提示组为输入来回答问题（给定一个测试问题图像对，PICa首先使用现成的字幕模型**将图像翻译成字幕，将问题、标题和一些上下文总的示例集成到文本提示中**，该文本提示可以诱导GPT-3直接预测答案 *依靠的是GPT-3强大的知识推理能力*）KAT和REVIVE学习KB增强VQA模型，但由于以下限制，它们尚未完全激活GPT-3：

1、生成的字幕无法覆盖图像中的所有必要的信息。（如“一群人在城市广场上行走”对回答“这些树结出什么果实”这个问题毫无帮助。）

2、GPT-3采用了少量的学习范式，需要一些上下文中的例子来适应新的任务。因此，这些示例的选择对于模型性能至关重要。如[46]（PICa）所述，与**使用基本事实答案相似性的预言机策略相比，其所有示例选择策略的性能都要差得多**。

启发式：在提示中以适当的方式给出一些更好的答案。

### 贡献

answer candidates &  answer-aware

1、前者： 测试输入有希望的答案列表。其中每个答案都与置信度分数相关。

2、后者：上下文中的示例列表，其中每个示例对测试输入都有类似的答案。



### 本文框架

本文主体框架是个概念上简单的两阶段框架。

在**答案启发式生成阶段**，学习普通VQA模型来生成两种类型的答案启发式，

在**启发式增强提示阶段**，答案启发式、问题、标题被集中到一个格式化的提示中，以指示GPT-3预测答案。

![](/img/prophet-2.png)

GPT-3上下文：

#### stage-1 答案启发生成阶段

我们介绍了两种类型的答案启发式：候选答案（answer candidates）和感知答案(answer-aware)的例子。给定由图像和问题组成的测试输入，候选答案（answer candidates）是指测试输入的有希望的答案列表，其中每个答案都与置信度分数相关联。感知答案(answer-aware)的例子指的是一个无正文的例子列表，其中每个例子对测试输入都有相似的答案。

有趣的是，这两种类型的答案启发式可以从在基于知识的VQA任务上训练的任何普通VQA模型中同时获得。 **这里用的VQA模型是MCAN**。*这个模型等会研究。*

将VQA数据集（ok-vqa）表示为**D={(vi，qi，ai) }**，vi是图像、qi是问题、ai是对应的答案。

训练集最频繁的答案形成答案词汇**W={wj}**，从VQA数据集D中可以学习到的普通VQA模型M，对答案S执行分类。通常，VQA模型可以分为两个子模型，即主干Mb和分类头Mh。

**主干Mb充当编码器来融合多模态输入v和q，并获得融合特征z**： z=M(v,q)

分类头Mh用于简单采用线性层和sigmoid函数，将融合的特征z投影到词汇表的预测向量y中。y=Mh(z)

其中**y[i]**表示y的第i个元素，表示是**w[i]**的置信度得分

**Answer candidates**：给定测试输入（v，q）,



未完待续。请补全（周五之前）

主要任务：

~~TAP能不能用？~~~~（先用下sam）~~

阅读预训练+微调流程+MCAN论文，重新复现阅读tap方法

阅读代码

目前先尝试一下Tap的方法（希望可