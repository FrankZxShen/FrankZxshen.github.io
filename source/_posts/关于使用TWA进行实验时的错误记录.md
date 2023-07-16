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

暂时没有发现除了nan之外的其他错误

现在需要把layoutmv2加进去，尽量，然后把图给找出来，画出来



不改，舍弃



### 4、升了一点点

每次修改修改4点：

1、base_trainer_魔改+coteaching（复制）

2、losses_twa_adv_jocor（复制）

3、twa_adv_jocor

4、config（mytwa<->twa）



### 5、精度很低，怀疑和batchsize有关

改albert

和学习率有关，albert不重要



想办法加layout

改图，问问题，还是没有解决

明后天做完再fineturn一下

#### 6、对比实验

画图，10-20号这期间把所有图给制作了，准备表格、消融实验

11号先用训练好的进行fineturn

### 7、我很怀疑是PrevPredEmbeddings的错误

修改为albert 按理说可以降低参数量？