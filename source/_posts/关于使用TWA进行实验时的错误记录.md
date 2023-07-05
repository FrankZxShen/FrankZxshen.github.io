---
title: 关于使用TWA进行实验时的错误记录
abbrlink: 6fbfcd62
date: 2023-07-02 16:58:54
tags:
categories:
top_img:
---

## 无语

### 1、要求添加的张量字段必须和SampleList已有字段大小相同。Passed size:8 Reqired_size:5

只能舍弃data后面的字段（可能出现问题，但是今天没有）

### 2、Nan（待解决）



### 3、~~不知道还加些什么~~ 加上layoutMv2的一部分

就是把所有token通过一次



### 4、升了一点点

每次修改修改4点：

1、base_trainer_魔改+coteaching（复制）

2、losses_twa_adv_jocor（复制）

3、twa_adv_jocor

4、config（mytwa<->twa）

