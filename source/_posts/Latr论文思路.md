---
title: Latr论文思路
abbrlink: 49464
date: 2023-03-30 10:43:00
tags: latr
categories: 论文阅读
cover: url(/img/latr.png)
---

2023-04-04

## LaTr: Layout-Aware Transformer for Scene-Text VQA

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

Latr主要由3个部分组成，首先，是一个只在文本上预先训练的语言模型（左边），将OCR tokens的边界框（bounding box）的空间嵌入与文档上的进一步布局感知（layout-aware）预训练结合使用。总结：只训练具有文本和空间线索的语言模态，以联合建模文本和布局信息之间的交互，进行大量的预训练。

![](/img/latr.png)





2023-03-30

分为两个部分，第一个部分是预训练部分，仅训练具有文本和空格的语言模态（用的ocr数据集），这是一个只在文本上预先训练的语言模型。
接下来，将OCR token令牌边界框的空间嵌入与对文档的进一步布局感知预训练结合使用。

## 预训练

只有传统的LM模块，空间模块。

## 关于训练代码

正式训练过程：其实就是调很多包

### 1、introduction

基于场景文本的VQA任务（多模态体系结构），训练数据集（图像）：TextVQA，并记录相应的权重和偏差。

Latr本质：一种用于做场景文本VQA的布局感知器，用于场景文本可视化问答（STVQA）

优化：提高对OCR错误的鲁棒性，消除对外部物体检测器的需求。

### 2、引入一些包

### 4、制作数据集

将输入预处理为给定的格式，然后将输入提供给模型，得到预处理的输入

### encoding

其中
img has a shape:               torch.Size([3, 384, 500])
boxes has a shape:             torch.Size([512, 6])
tokenized_words has a shape:   torch.Size([512])
question has a shape:          torch.Size([512])
answer has a shape:            torch.Size([512])
id has a shape:                torch.Size([])

### decoding

OCR decoding（1 cup 1 cup 1 cup 250 ml 1 cup 250 ml 1 cup 250 ml 1 cup 1 cup） +
Image decoding（被压扁了。。224*224） +
Question decoding（How big is the measuring cup? ）

### 给dataloader加载Collate function

### 5、define the Datamodule

定义数据集参数，比如训练集、测试集、batch_size啥的


### 6、最终训练阶段

1、定义config文件
2、定义布局VQA训练的各种杂项参数：
输入：boxes、img、question、words ；比如学习率、激活函数、训练损失。。。
3、定义训练主函数
主要参数（max_step：50000，seed：42）
开始训练，保存模型位置”./latr/models“

ok，爆显了，3070真tm的垃圾，又爆显又爆内存，我不玩了

什么时候有3090啊。。。