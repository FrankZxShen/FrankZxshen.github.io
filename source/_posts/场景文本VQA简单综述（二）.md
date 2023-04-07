---
title: 场景文本VQA简单综述（二）
tags: 场景VQA综述
abbrlink: cfe2e6c5
date: 2023-04-01 12:10:22
categories: 论文阅读
cover: url(/img/stvqa.png)
---

## 主要数据集

### TextVQA

2.8w图，4.5w问题

### STVQA

2.1w图，3.1w问题

其他还是TextCaps（2.8w图，14w问题）做图像描述的

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



## M4C: Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA

多模态transformer对TextVQA进行迭代式答案预测

当前的数据集和方法大多集中在场景中的视觉成分上，但是往往忽略一个关键的模态：图像中的文本，它继承着对场景理解和推断的重要信息（比如洗手间中的“小心地滑”告示牌）。提出TextVQA数据集。

**TextVQA任务：**模型可以看到、阅读和推理三种模态。对于以前的工作，基于单纯词嵌入的图像文本特征表示能力有限，且容易错过重要的线索。比如文本的token的外观和位置layout（*latr解决*）

**本文的模态多拷贝网（M4C）模型：**

![](/img/M4C1.png)

给定一个问题和一张图片作为输入，从三种模态提取特征表示：问题、图片中的视觉特征（用的FasterRCNN）、图片中的文字。这三种模态分别表示为**问题词特征列表**、**来自现成的目标检测器的视觉目标特征列表**、**基于外部OCR系统的OCR token特征列表**。

通过特定领域的嵌入方法将所有实体（问题词、检测到的视觉对象和检测到的OCR token）投射到一个共同的d维语义空间，并在投射的事物列表上应用多个transformer层。基于transformer的输出，我们通过迭代的自回归解码来预测答案，在每一步，我们的模型要么通过动态指针网络选择一个OCR token，要么从其固定的答案单词表中选择一个词。

### 本文的贡献

1、提出一种基于多模态Transformer和指针网络的文本视觉问答（TextVQA）方法，将文本、图像和回答进行联合建模。

2、提出一种迭代答案预测方法，采用指针试多步解码器进行预测。可以保持准确性的同时提高生成答案的多样性。

3、在几个TextVQA数据集上进行了广泛的实验，比较好。（至少下面这篇论文采用了部分方法模型的encoder）



## TAP: Text-Aware Pre-training for Text-VQA and Text-Caption

文本VQA和文本标题的文本感知预训练

### 提出的方法

**两项任务：**

文本VQA（TextVQA：旨在**回答跟图像中文字相关的问题**，比如“what is the company name?”，此时输出答案是一个词或一个词组）

文本标题任务（Text-Caption：旨在对一张图片生成一句直接覆盖了图像中所有文字信息描述（*感觉有点像根据图片进行人为的推理啥的*）

### 问题

目前许多预训练任务不能指导模型关注**图像中的文本信息**，以及**图像中的文本信息和图像内容**的关系，在下游任务Text-VQA和Text-Caption上都取得了最好的效果。

### 模型结构

#### 预训练结构（右边）

（*注：本文没有用VIT，这可能是个改进方向。但是你用T5和T5X这不全秒了？什么是Google的含金量啊！*）

3个embedding + 4层多模态编码器（Fusion Module）,encoder部分和上面论文一致。模型的输入包括**文本特征序列**`w`、**视觉对象序列**`v^{obj}`、**文字视觉特征序列（OCR）**`v^{ocr}`。

其中**文本特征序列**`w=[w^{q},w^{obj},w^{ocr}]`，`w^{q}`表示问题或者图像描述的token序列，`w^{obj}`表示图像中用Faster R-CNN识别出来的对象的类别序列，`w^{ocr}`指的是通过OCR检测模型得到的图像中的文字序列。

`v^{obj}`和是基于Faster-RCNN得到的bounding boxes抽取的**图像视觉特征序列**。

`v^{ocr}​`是基于OCR检测模型得到的**文字视觉特征序列**。

文本特征序列`w`会先经过文本编码器得到token embedding，然后送入Fusion Module中，视觉特征序列 `v^{obj}`和 和`v^{ocr}`则分别经过一层全连接层，然后送入Fusion Module中。此外还有一个特殊的token <begin>（`p_0`）也会同时输入，用于预测图文是否匹配。

##### 预训练任务分为两个任务：

1、**scene-text language pre-training tasks**

图像中文字相关的涉及**视觉和文本**的多模态训练任务。作者用了**ITM**（image-text matching 图像文字匹配）以及**MLM**（masked language modeling 掩码语言模型），*这个东西其实和latr一样，它用的也是MLM。*常见的策略，这个可以不变（15%mask, 其中80%换成<mask>，10%换成一个随机的词，10%不做修改）。但是不同的是，**本文引入`w^{ocr}`作为文本序列的一部分**，这其实有点牵强，相当于增加了图像文字被mask的概率，这会使得模型更关注文字部分的信息。

2、**Scene-text Visual Pre-training tasks**

图像中文字相关但是仅涉及**视觉**的预训练任务。建模**图像中对象和图像中文字的空间关系**往往能帮助理解两者之间的语义关系，例如一本书的封面上方往往是书名而不是出版社。因此，本文提出了一个专门预测空间关系的任务**RPP**（relative position prediction）。在该任务中，随机选取多模态模型输出的一个视觉对象`f^{obj}_i`和文字视觉对象`f^{ocr}_i`，通过全连接层，预测两者之间的相对位置关系。（*这个。。。可以采用？也是用layout的*）相对位置关系类别在该任务的开始阶段采用2分类（即是否包含），之后拓展到12分类，包括on，cover，overlap， eight-way relative oriention和unrelated。

![](/img/TAP1.png)

#### 下游任务（左边）

对于两个下游任务（还是TextVQA与Text-Caption）采用了统一的模型结构M4C，encoder与预训练的一致，decoder采用了一个全连接层和一个指针网络（Pointer Network）（*我感觉可能用不上*），全连接层将特征映射到一个常用词词典上，后者将特征映射到对应图像包含文本的词典上，解决文字图像中的OOV（out-of-vocabulary）问题。两个任务采用一致的模型结构，但是需要注意的是，对于Text-Caption任务来说，输入是没有文本序列中的子序列`w^q`的，因此这部分在finetune和测试的时候都是空白。

### 贡献+总结

效果

![](/img/TAP2.png)

上面这张图是TextVQA的效果

消融实验结果：

OCR的`w^OCR`、RPP位置任务都能带来很好的提升；

模型数量大、模型本体大对带来提升；



针对特定的下游任务，提出了**特定的预训练任务**->好的效果；

本文验证了即使是**小规模的数据**，也可以用同样的预训练任务来提升模型效果。



## 补充：T5：Text-to-Text Transfer Transformer

T5是一种基于Transformer框架的预训练模型，旨在将自然语言文本的各类任务转化为相同的文本到文本转化问题，包括文本分类、文本问答、文本生成、文本摘要翻译等等。

需要考虑下gpt-3？x不行



未完待续。。。

