---
title: Latr论文思路（二）
tags: latr
categories: 论文阅读
cover: url(/img/latr.png)
abbrlink: 72c16057
date: 2023-04-11 10:18:08
---

## 补充

### TextVQAAccuracyEvaluator（）类（M4C）

输入：prediction

```
predictions.append({
        "pred_answer": pred_answer,
        "gt_answers": gt_answers,
})
```

目标：改进val部分

将其用于验证，但是现在效果好像很糟糕？？？

