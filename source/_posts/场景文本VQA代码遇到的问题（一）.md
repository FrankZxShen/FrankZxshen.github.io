---
title: 场景文本VQA代码遇到的问题（一）
tags: 场景VQA代码
categories: 代码阅读
cover: url(/img/Pandora.jpg)
abbrlink: 468cc499
date: 2023-04-19 19:34:40
---

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





