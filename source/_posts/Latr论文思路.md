---
title: Latr论文思路
abbrlink: 49464
date: 2023-03-30 10:43:00
tags: latr
categories: 论文阅读
cover: url(/img/latr.png)
---

2023-04-03

换个电脑写吧。。。

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