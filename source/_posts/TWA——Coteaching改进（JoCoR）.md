---
title: TWA——Coteaching改进（JoCoR）
abbrlink: ad813615
date: 2023-06-28 21:45:43
tags:
categories:
top_img:
---

## 改进方法

### JocoR方法

1、首先正常获取image和label，正常Forward

2、获取损失函数。有两个损失，但是这两个损失相同

损失需要5个参数：第一个输出（prepared_batch）、第二个输出（prepared_batch2）、gt（model_output）、rate_schedule（这是？）、co_lambda（0.1）

其中rate_schedule = np.ones(config['epochs']（200）) * forget_rate（forget_rate = config['percent'] （0.2）/ 2）

3、计算损失

这个单独的损失计算方法比我这个简单多了，就是两个cross_entropy

最烦的就是那个kl损失。必须要两个原来的y1和y2（两个预测的label）

改完了，但是爆显

目前损失的计算方法和jocor一致，每个model相当于都是单独训练的。肯定可以提，但是网络结构没法加了

### 新加入一个分支model_output2

对于这个新加入的model_output2，需要一个全新的prepared_batch2存放batch的数据。



计算损失时考虑将最后得到的损失放到loss外部，base_trainer做实验（我真的很担心爆显）

loss_pick1：loss

loss_pick2：作为另外一个模型的损失

