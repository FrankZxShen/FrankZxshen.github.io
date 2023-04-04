---
title: Efficientzero实验记录（一）
tags: 强化学习
categories: 实验记录
top_img: url(/img/ezero1.png)
abbrlink: b24f8d95
date: 2023-04-03 08:09:17
---

## 2023.04.03记录

### 记录1 啥都不改

**模型：**

ResNet

**参数设置：**

seed=0

lr_init=0.2,

lr_decay_rate=0.1

batch_size=64（输嘛了）

p_mcts_num=1

optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,weight_decay=config.weight_decay)

training_steps=100000

**实验结果：**

突出的就是一个字：差

Test Mean Score of PongNoFrameskip-v4: **9.28125**



### 记录2 seed=2023

**模型：**

SE-ResNet

**参数设置：**

seed=2023

lr_init=0.1,

lr_decay_rate=0.08

batch_size=64

p_mcts_num=1

optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,weight_decay=config.weight_decay)

training_steps=100000

**实验结果：**

Test Mean Score of PongNoFrameskip-v4: **11.96875**



### 记录3 seed=2025

**模型：**

SE-ResNet

**参数设置：**

seed=2025

lr_init=0.1,

lr_decay_rate=0.1

batch_size=64

p_mcts_num=4

optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,weight_decay=config.weight_decay)

training_steps=100000

**实验结果：**

Test Mean Score of PongNoFrameskip-v4: **9.84375**



### 记录4 seed=2025

**模型：**

SE-ResNet

**参数设置：**

seed=2025

lr_init=0.1,

lr_decay_rate=0.1

batch_size=128

p_mcts_num=4

optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,weight_decay=config.weight_decay)

training_steps=200000

**实验结果：**

Test Mean Score of PongNoFrameskip-v4: **19.25625**



### 记录5 seed=2023

**模型：**

SE-ResNet

**参数设置：**

seed=2023

lr_init=0.0001,

lr_decay_rate=0.01

weight_decay=1e-6

batch_size=128

p_mcts_num=4

optimizer = optim.Adam(model.parameters(), lr=config.lr_init ,weight_decay=config.weight_decay)

training_steps=200000

**实验结果：**

Test Mean Score of PongNoFrameskip-v4: **1.09375**



