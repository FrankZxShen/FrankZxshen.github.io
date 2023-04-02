---
title: 场景文本VQA简单综述（二）
tags: 场景VQA综述
abbrlink: cfe2e6c5
date: 2023-04-01 12:10:22
categories: 论文阅读
cover: url(/img/stvqa.png)
---

## Scene Text Visual Question Answering——ICCV2019

做stvqa的始祖（其实可能是**OCR-VQA: Visual Question Answering by Reading Text in Images**）

#### 提出的问题

​	当前的VQA最主要的问题在于没有考虑图像中文本的语义信息。基于此原因，作者提出了一个新的数据集，ST-VQA，目的在于强调探索图像中出现的高级语义信息的重要性，并将该信息作为文本线索用于视觉问答过程。我们使用这个数据集来定义一系列难度越来越大的任务，对于这些任务，在视觉信息提供的上下文中读取场景文本是推理和生成适当答案所必需的。针对这些任务，本文提出了一种新的评估指标，既考虑了文本识别模块的推理错误，也考虑了文本识别模块的缺陷。此外，我们还提出了一系列baseline，为新发布的数据集提供了进一步的视角，为进一步的研究奠定了基础。

#### 方法

##### 1、random 

随机选一个答案

##### 2、Scene Text Retrieval

**STR**：使用单个CNN+PHOC，首先，对于一张给定的图像（STR retrieval），使用一个任务字典作为query（uses the specific task dictionaries as queries to a given image），之后，对于最明显的场景文本进行提问（STR bbox）

##### 3、Scene Image OCR

检测到的文本置信度排序

##### 4、Standard VQA models

**SAAA：**CNN+LSTM；CNN采用的是resnet-152，提取图像特征至14 * 14 * 2048，同时用LSTM对单词进行嵌入，然后再用attention对两个特征进行映射，最后再连接，送入全连接层进行分类。优化器采用Adam，batch size为128，epoch为30，初始学习率为0.001，每50000次衰减一半；

**SAN：**图像VGG，文字LSTM；用预训练的VGG模型获取图像特征至14 * 14 * 512，提取问题特征用的是LSTM，batch size设置为100，epoch为150，优化器RMSProp，初始学习率是0.0003，衰减率为0.9999



![](/img/STVQA指标.png)

### 贡献

1、推动OCR与VQA领域结合，STVQA任务挑战了**文本检测、文本识别、自然语言处理**等多个领域之间的交叉问题，为这两个领域的交叉研究提供了新的思路和方法。

2、收集了一个大型数据集STVQA，其中包含了**真实场景中的文本图片和与之相关的问题与答案**，为STVQA任务的研究提供了数据基础。

3、提出一些简单的模型：比如**Standard VQA Models + Scene Text Retrieval**，对图像使用CNN，对单词使用LSTM进行词嵌入。不是现在主流的transformer的方法。

4、尝试跨语言STVQA。



## Multi-Modal Graph Neural Network for Joint Reasoning on Vision and Scene Text 

用于视觉和场景文本联合推理的多模态图神经网络

这个其实和我们的方向有点不一样，用的是图神经网络，没有用transformer

### 贡献

1、提出了一种**多模态图神经网络**（MMGNN），能够联合**处理视觉和场景文本信息**。该模型在现有的场景文本VQA数据集上实现了目前最好的结果。

2、引入了场景文本图（STG）来表示场景文本信息。这种图将场景文本中每个单词作为节点，单词之间的空间和语义作边，用于对场景文本信息进行建模。

3、一种新的词嵌入技术，即基于词语序列和视觉区域序列的双向编码器，能够同时编码场景文本和视觉信息。

4、基于多头注意力机制的联合编码器（JEC），能够将多个视觉区域的信息与场景文本的信息进行结合，并生成联合特征表示。

5、一种基于GCN的多模态图神经网络，能够同时对视觉图和场景文本图进行建模，并学习跨模态交互。

6、效果牛逼（开玩笑，latr的40几不秒杀你）

总结：可以考虑，但是我们不用CNN，且效果不行



## TAP: Text-Aware Pre-training for Text-VQA and Text-Caption

文本VQA和文本标题的文本感知预训练

### 提出的方法

**两项任务：**

文本VQA（TextVQA：旨在**回答跟图像中文字相关的问题**，比如“what is the company name?”，此时输出答案是一个词或一个词组）

文本标题任务（Text-Caption：旨在对一张图片生成一句直接覆盖了图像中所有文字信息描述（感觉有点像根据图片进行人为的推理啥的））

### 问题

目前许多预训练任务不能指导模型关注**图像中的文本信息**，以及**图像中的文本信息和图像内容**的关系，在下游任务Text-VQA和Text-Caption上都取得了最好的效果。

### 模型结构

#### 预训练结构

（注：本文没有用VIT，这可能是个改进方向）