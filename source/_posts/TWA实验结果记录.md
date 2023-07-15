---
title: TWA实验结果记录
abbrlink: 1b7a76a3
date: 2023-07-15 19:52:07
tags:
categories:
top_img:
---

## 目前做出的改进

1、将原来的bert的方法修改为layoutlm（可能存在问题（比如理解不太正确），需要进一步修改）

2、对抗训练，增加训练迭代次数

3、coteaching

4、混合精度训练（考虑丢到cpu上）可能导致精度下降



目前仅采用textvqa预训练只有86.6，finetuning还没出来。

