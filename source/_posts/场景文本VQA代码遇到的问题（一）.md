---
title: 场景文本VQA代码遇到的问题（一）
tags: 场景VQA代码
categories: 代码阅读
cover: url(/img/Pandora.jpg)
abbrlink: 468cc499
date: 2023-04-19 19:34:40
---

场景文本视觉问答是一个研究领域，涉及使用视觉线索回答自然场景中存在的文本问题。该领域的一些创新包括:

**OCR和VQA的融合**：场景文本视觉问答结合了光学字符识别(OCR)和视觉问答(VQA)技术来识别给定场景中的文本并回答与之相关的问题。这种集成有助于提高系统的准确性。

**注意机制**：许多场景文本可视化问答模型在回答问题时，使用注意机制来关注场景的重要部分，如文本区域或图像背景。这有助于减少输入中的噪声，提高系统的精度。

**多模态融合**：为了回答场景中存在的文本问题，场景文本视觉问答模型需要将文本信息和场景的视觉特征结合起来。采用多模态融合技术将这两种模态信息结合起来，提高了系统的精度。

**迁移学习**：许多场景文本可视化问答模型使用迁移学习技术来利用预先训练的模型来完成相关任务，如VQA或OCR。这有助于提高系统的性能，特别是在训练数据有限的情况下。

**数据集创建**：为了促进这一领域的研究，已经创建了几个数据集，为场景文本可视化问答提供大量带注释的数据。这些数据集有助于开发和评估该领域的新模型和新技术。

开始更新：遇到的问题。

## 最大问题：数据集

不像传统的VQA，采用的是coco2014、2017的数据集，这个用了非常复杂的textvqa、stvqa以及ocr的部分。下面将进行简单分析。

代码先可以不看，但一定要把配置文件读懂。

数据集文件整体结构：

├── 1600-400-20
│   └── objects_vocab.txt
├── detectron
│   ├── fc6
│   └── resnext152_fc6
├── feat_resx
│   ├── stvqa
│   ├── test
│   └── train
├── imdb
│   ├── cc
│   ├── m4c_textcaps
│   └── m4c_textvqa
├── m4c_captioner_vocabs
│   ├── coco
│   ├── textcaps
│   └── textcaps_coco_joint
├── m4c_vocabs
│   ├── ocrvqa
│   ├── stvqa
│   └── textvqa
├── ocr_feat_resx
│   ├── stvqa_conf
│   └── textvqa_conf
└── original_dl
    ├── ST-VQA
    ├── TextCaps_0.1_train.json
    └── TextCaps_0.1_val.json

TAP这篇文章在**TextVQA**和**STVQA**上的Text-VQA任务以及TextCaps数据集上的TextCaption任务进行了TAP基准测试。本文使用（微软）提出的OCR-CC数据集进行大规模预训练。

TAP用的4个主要数据集：

### TextVQA

包含来自Open Image的28408个图片。使用之前工作相同的训练、验证、测试分割。

在文件中的表现：

image_features：faster-rcnn提取的图像的特征；

```
feat_resx/train(test)
```

imdb_files：ocr文件，主要保存的.npy的微软ocr文件；

```
imdb/m4c_textvqa/imdb_test(val、train)_ocr_en.npy
```

换了一个网络，我重新下载试试



