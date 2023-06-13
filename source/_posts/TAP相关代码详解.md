---
title: TAP相关代码详解
abbrlink: 6810bde8
date: 2023-05-31 16:45:11
tags:
categories:
top_img:
---

OPT（用于对抗训练）

```
# adversarial training related
    parser.add_argument('--adv_training', action='store_true',
                        help="Whether to use adversarial training or not")
    parser.add_argument("--adv_modality", default=['text'],
                        help="add pertubation on text or image modality")
    parser.add_argument('--adv_lr_txt', type=float, default=0)
    parser.add_argument('--adv_lr_img', type=float, default=0)
    parser.add_argument('--adv_steps', type=int, default=1, help="should be at least 1")
    parser.add_argument('--norm_type', type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument('--adv_max_norm', type=float, default=0, help="set to 0 to be unlimited")
    parser.add_argument('--adv_kl_weight', type=float, default=0, help="set to 0 to be unlimited")
```



## Class M4C 主类

###  sample_list

['question_id', 'image_id', 'image_feature_0', 'image_info_0', 'image_feature_1', 'image_info_1', 'text_mask_label', 'text', 'text_len', 'obj_bbox_coordinates', 'objtext_mask_label', 'obj_text', 'obj_text_len', 'ocrtext_mask_label', 'ocrtag_pollute', 'langtag_pollute', 'objtag_pollute', 'tag_pollute', 'ocr_text', 'ocr_text_len', 'context', 'context_tokens', 'context_tokens_enc', 'context_feature_0', 'context_info_0', 'context_feature_1', 'context_info_1', 'order_vectors', 'ocr_bbox_coordinates', 'overlap', 'overlap_obj', 'overlap_ocr', 'gt_answers_enc', 'targets', 'sampled_idx_seq', 'train_prev_inds', 'train_loss_mask', 'dataset_type', 'dataset_name', 'cmb_text', 'cmb_text_len', 'cmb_text_mask_label', 'dataset_type_', 'dataset_name_']

具体的类别和内容放到TAP文件夹中

**gt_answer_enc：**

**[32, 4094]**

tensor([[  0,  38, 128,  ...,   0,   0,   0],
        [  0, 136, 128,  ...,   0,   0,   0],
        [  0,  57, 128,  ...,   0,   0,   0],
        ...,
        [  0,  43, 128,  ...,   0,   0,   0],
        [  0, 112, 128,  ...,   0,   0,   0],
        [  0, 157, 128,  ...,   0,   0,   0]], device='cuda:1',
       dtype=torch.uint8)

tensor([[  0, 105, 128,  ...,   0,   0,   0],
        [  0, 167, 128,  ...,   0,   0,   0],
        [  0, 141, 128,  ...,   0,   0,   0],
        ...,
        [  0,  97, 128,  ...,   0,   0,   0],
        [  0,  50, 128,  ...,   0,   0,   0],
        [  0, 156, 128,  ...,   0,   0,   0]], device='cuda:0',
       dtype=torch.uint8)

**image_feature_0/image_feature_1：**

[32, 100, 2048]

**cmb_txt：**

[32, 170]

**model_output['score']/targets:**

torch.Size([32, 12, 5100])

主要是model_output

### model_output

['scores', 'textcls_scores', 'pollutecls_scores', 'overlapcls_scores']

torch.Size([32, 12, 5100])





对抗学习加入对抗干扰

**words_embeddings = words_embeddings + adv_delta直接在embedding后编码后加上同一个维度的干扰**

### 目标（6.2）

看fwd_results字典元素维度

1、直接创造一个fwd_results['txt_emb']出来->txt_embeds_init（用来创造0干扰矩阵）

2、一个视觉的（明天补充）

3、一个OCR的（后续）

4、gt_answer_scores（model_output['scores']）（用于与干扰矩阵的anwer输出计算KL散度）



### prepared_batch（sample_list）

['question_id', 'image_id', 'image_feature_0', 'image_info_0', 'image_feature_1', 'image_info_1', 'text_mask_label', 'text', 'text_len', 'obj_bbox_coordinates', 'objtext_mask_label', 'obj_text', 'obj_text_len', 'ocrtext_mask_label', 'ocrtag_pollute', 'langtag_pollute', 'objtag_pollute', 'tag_pollute', 'ocr_text', 'ocr_text_len', 'context', 'context_tokens', 'context_tokens_enc', 'context_feature_0', 'context_info_0', 'context_feature_1', 'context_info_1', 'order_vectors', 'ocr_bbox_coordinates', 'overlap', 'overlap_obj', 'overlap_ocr', 'gt_answers_enc', 'targets', 'sampled_idx_seq', 'train_prev_inds', 'train_loss_mask', 'dataset_type', 'dataset_name', 'cmb_text', 'cmb_text_len', 'cmb_text_mask_label', 'dataset_type_', 'dataset_name_']

现在遇到的问题

为什么不能在tap.py中修改？可以 换CPU上执行



### 主循环

### batch

['question_id', 'image_id', 'image_feature_0', 'image_info_0', 'image_feature_1', 'image_info_1', 'text_mask_label', 'text', 'text_len', 'obj_bbox_coordinates', 'objtext_mask_label', 'obj_text', 'obj_text_len', 'ocrtext_mask_label', 'ocrtag_pollute', 'langtag_pollute', 'objtag_pollute', 'tag_pollute', 'ocr_text', 'ocr_text_len', 'context', 'context_tokens', 'context_tokens_enc', 'context_feature_0', 'context_info_0', 'context_feature_1', 'context_info_1', 'order_vectors', 'ocr_bbox_coordinates', 'overlap', 'overlap_obj', 'overlap_ocr', 'gt_answers_enc', 'targets', 'sampled_idx_seq', 'train_prev_inds', 'train_loss_mask', 'dataset_type', 'dataset_name', 'cmb_text', 'cmb_text_len', 'cmb_text_mask_label', 'dataset_type_', 'dataset_name_']

#### 大致思路：

首先在外层定义delta，然后将delta作为prepaerd_batch字典的参数传入，在模型内部更新delta

### loss

1、第一步

设定两个report，第一个report记录原来没有加入干扰的model_output loss;

​								第二个report_adv记录加入干扰的model_adv_output loss;

2、

计算完成所有损失后，准备混合精度回溯。



### 目标（6.7-6.10）

对OCR进行对抗训练



OCR共有4个特征

1、OCR_fasttext特征

2、OCR_PHOC特征

3、OCR Faster RCNN特征

4、还有一个bbox边框特征

目标是把这几个特征全部加起来，再加上一个对抗干扰

5、考虑加上OCR纠错（AAAI那个） 看看OCR特征的具体内容

看看维度

co-teaching（预训练）

#### 特征

txt_delta [16, 170, 768]

img_delta [16, 100, 2048]

ocr_delta  [16, 100, 2048]

ocr_fasttext_delta [16, 100, 300]

ocr_phoc_delta [16, 100, 604]

不行，效果堪忧（为什么？）

考虑更换去噪网络+colearning

## 算法具体流程（6.10）

1、获取sample

也就是当前



## 6.14

提了一点点，但是没有完全提，这是为什么？

强烈建议把colearing加进去

最后最后（我没办法了）再加入ocr增强（直接加不知道怎么写）