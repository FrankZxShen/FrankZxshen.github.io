---
title: 关于训练损失d问题
abbrlink: e56d884d
date: 2023-07-31 23:07:46
tags:
categories:
top_img:
---

## 损失设置

```python
import torch.nn.functional as F
a=[8,35022,170]
b=[8,170]
F.cross_entropy(a,b,reduction='none',ignore_index=-1)#[8,170]
#b必须为long
#ignore_index表示忽略掉b中-1的
```

对于kl_loss，将原来[8,35022,170]维度的loss矩阵求mean(dim=1)?应该就行