---
title: 'Lora:从代码分析'
tags: Lora
categories: 代码阅读
cover: url(/img/Lora.png)
abbrlink: e57dac46
date: 2023-05-06 09:19:29
---

什么是Lora技术？

https://arxiv.org/pdf/2106.09685.pdf

​	在大型语言模型的传统微调中（当然，不只是大型语言模型），整个模型通常使用大量特定于任务的数据对新任务进行微调。这个过程在计算上可能很昂贵，并且需要大量内存，这可能是在资源受限的设备上进行实际部署的瓶颈。

​	LoRA方法提出了一种低秩自适应技术，该技术在微调期间仅适应语言模型中的一部分参数，同时保持其余参数不变。这降低了微调的计算成本和内存需求，同时保持了与传统微调相似的性能。

​	该方法将语言模型中某一层的权重矩阵分解为低秩矩阵和残差矩阵。然后在微调期间调整低秩矩阵，而残差矩阵保持固定。这使模型能够使用更少的特定于任务的数据快速适应新任务。

![](/img/Lora.png)

### 原理

​	在模型的Linear层旁增加一个右边的分支，该分支将代替原有的参数矩阵$$W$$进行训练。例如在Transformer模型中，$$x$$可能是Embedding的输出，也可能是上一层Transformer layer的输出。

​	比如一个输出维度d，我们将它传入右边的分支，Linear层A将数据从d维降到r（LoRA的秩，一般为2，4，8，16），再通过第二Linear层B将r变回d维。最后将左右两部分结果相加，得到下一个输出的h（隐藏状态）。

​	对于左右两部分，右侧可以看作为左侧原有矩阵$$W$$的分解，将参数量从$$d*d$$维降维至$$d*r+d*r$$，由于r一般远远小于d，参数量就大幅度降低。
$$
h=W_0x +\Delta{Wx}=W_0x+BAx
$$
​	LoRA在训练过程中保留了原来的矩阵$$W$$，但是不让$$W$$参与训练，所以需要计算梯度的部分就只剩下旁支的A和B两个小矩阵。在加入LoRA前，模型优化表示为：
$$
max_{\Phi}\sum_{(x,y)\in Z}\sum_{t=1}^{|y|}log(P_{\Phi}(y_{t}|x,y_{<t})
$$
模型参数采用$$\Phi$$表示，优化表示为：
$$
max_{\Theta}\sum_{(x,y)\in Z}\sum_{t=1}^{|y|}log(P\Phi_{0}+\Delta\Phi(\Theta)(y_{t}|x,y_{<t})
$$
可以看到，对于LoRA，只需要训练后面的$$\Delta\Phi(\Theta)$$，不需要训练整体的$$P_{\Phi}$$。

### 代码部分

最简单的方式：

PEFT的LoRA实现：在PEFT实现LoRA时，将语言模型中Transformer层的权矩阵分解为低秩矩阵和残差矩阵。然后使用一小部分训练数据自适应微调低秩矩阵，而残差矩阵保持固定。这降低了微调过程的计算成本和内存需求，同时保持了与传统微调方法相似的性能。PEFT的LoRA实现还包括一种机制，用于在训练期间根据验证集上的性能动态调整低秩矩阵的秩。这使得模型能够适应任务的复杂性，并相应地调整低秩矩阵的秩。用于替换"q_proj",
    "v_proj"。

```
# 设置超参数及配置
LORA_R = 4 # LoRA秩
LORA_ALPHA = 16 # 
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# 创建基础transformer模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
# 加入PEFT策略
model = get_peft_model(model, config)
```

####  loralib实现：layers.py

LoRALayer()：定义基类，各种超参数定义，其中最重要的是r，LoRA的秩。

```
class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
```



##### Linear：

用于替换nn.Linear()，继承上面的基类。

```
class Linear(nn.Linear, LoraLayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)     
```

参数初始化：

lora_A采用kaiming初始化；lora_B采用零初始化，确保$$\Delta W=BA$$值为0；

```
def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
```

训练：

设置mode=True，这块作用是采用LoRA实现权重张量的合并；检测权重是否尚未合并。如果是，该方法减去矩阵自身的乘积；如果不是，则将添加矩阵自身的乘积。权重最后进行scaling的值缩放。前者消去了权值并更新了权值张量，后者合并了权值并更新了权值张量。

```
def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True  
```

forward:

```
def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
```

使用方法（包括Conv2d，Conv1d）：

```
# ===== Before =====
# layer = nn.Linear(in_features, out_features)

# ===== After ======
import loralib as lora
# Add a pair of low-rank adaptation matrices with rank r=16
layer = lora.Linear(in_features, out_features, r=16)
```

什么意思呢？比如我们定义一个全连接层nn.Linear(200, 500)，输入一个矩阵[100, 200]，我们可以得到全连接层的输出[100, 500]，这时算法需要计算一个[100, 500] * [500, 200]的矩阵乘法和一个矩阵加法，计算复杂度为$$10^7(100*500*200)$$，当我们使用lora.Linear时，同样得到输出矩阵[100, 500]，lora计算矩阵[100, 200]与lora_A[200, 16]的乘积，得到矩阵[100, 16]，再与lora_B[16, 500]的乘积得到[100, 500]，但这时矩阵计算复杂度已经降到$$10^6(100*16*200+100*16*500)$$，当然，设置更小的r，计算复杂度将继续下降。



##### Conv2d：

用于替换nn.Conv2d()

首先我们来看一个传统的例子（https://blog.csdn.net/qq_42079689/article/details/102642610）

例如输入输入四维向量[20, 16, 50, 100] #[N, C, H, W]

```
conv2d = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3)
```

输出计算：
$$
out(N,C_{out})=bias(C_{out})+\sum_{k=0}^{C_{in}-1}weight(C_{out},k)*input(N,K)
$$
N为batch size，C为通道数量（灰度图为1），HW是高和宽；

输出维度计算：
$$
[C, outchannels, \frac{H-K+2P}{S}+1,\frac{W-K+2P}{S}+1]
$$
K(ernel)=3，P(adding)=0，S(tride)=0

输出：[20, 64, 48, 98]

Lora的方法与Linear相似，这里分析不同的部分：

Lora_A变为了[r * kernel_size, in_channels * kernel_size]，Lora_B变为了[out_channels * kernel_size, r * kernel_size]

```
def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
```

在forward部分，以相同的方式改变weight进行传播。

```
def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x, 
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return nn.Conv2d.forward(self, x)
```



#### 实验&Conv1d：

类似的，官方没有给出Conv1d的实现，但是很多词向量需要采用Conv1d来处理，我就在想Conv1d可以依葫芦画瓢？

只是将Conv2d替换为Conv1d，采用以下代码进行测试：

```
import torch
import time
import torch.nn as nn
from loralib_tmp import lora_layers
from memory_profiler import profile

@profile
def c2d():
    time_start1 = time.time() 
    conv2d = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3)
    input = torch.randn(20, 16, 50, 100)
    conv2d(input)

    time_end1 = time.time()  
    time_sum1 = time_end1 - time_start1  
    print('Conv2d time:',time_sum1)

###################################################################################
@profile
def lc2d():
    time_start2 = time.time()  
    input = torch.randn(20, 16, 50, 100)
    lora_conv2d = lora_layers.Conv2d(in_channels=16,out_channels=64,kernel_size=3, r=4)
    lora_conv2d(input)

    time_end2 = time.time() 
    time_sum2 = time_end2 - time_start2  
    print('lora_Conv2d time:',time_sum2)

####################################################################################
@profile
def c1d():
    time_start3 = time.time()   
    conv1d = nn.Conv1d(16, 33, 3)
    input2 = torch.randn(20, 16, 50)
    conv1d(input2)
    time_end3 = time.time()  
    time_sum3 = time_end3 - time_start3  
    print('Conv1d time:',time_sum3)

#####################################################################################
@profile
def lc1d():
    time_start4 = time.time()  
    input2 = torch.randn(20, 16, 50)
    lora_conv1d = lora_layers.Conv1d(16, 33, 3)
    lora_conv1d(input2)

    time_end4 = time.time() 
    time_sum4 = time_end4 - time_start4  
    print('lora_Conv1d time:',time_sum4)

#####################################################################################
@profile
def lin():
    time_start5 = time.time()  
    input3 = torch.randn(100, 200)
    linear = nn.Linear(200, 500)
    linear(input3)

    time_end5 = time.time() 
    time_sum5 = time_end5 - time_start5  
    print('Linear time:',time_sum5)

#####################################################################################
@profile
def llin():
    time_start6 = time.time()  
    input3 = torch.randn(100, 200)
    lora_linear = lora_layers.Linear(200, 500)
    lora_linear(input3)
    time_end6 = time.time() 
    time_sum6 = time_end6 - time_start6  
    print('lora_Linear time:',time_sum6)

if __name__ == '__main__':
    c1d()
    lc1d()
    c2d()
    lc2d()
    lin()
    llin()
```



实验结果：

##### Line #    Mem usage    Increment  Occurrences   Line Contents

    31    110.2 MiB    110.2 MiB           1   @profile
    32                                         def c1d():
    33    110.2 MiB      0.0 MiB           1       time_start3 = time.time()
    34    110.7 MiB      0.5 MiB           1       conv1d = nn.Conv1d(16, 33, 3)
    35    110.8 MiB      0.2 MiB           1       input2 = torch.randn(20, 16, 50)
    36    112.9 MiB      2.1 MiB           1       conv1d(input2)
    37    112.9 MiB      0.0 MiB           1       time_end3 = time.time()
    38    112.9 MiB      0.0 MiB           1       time_sum3 = time_end3 - time_start3
    39    112.9 MiB      0.0 MiB           1       print('Conv1d time:',time_sum3)


lora_Conv1d time: 0.0013039112091064453

    42    112.9 MiB    112.9 MiB           1   @profile
    43                                         def lc1d():
    44    112.9 MiB      0.0 MiB           1       time_start4 = time.time()
    45    112.9 MiB      0.0 MiB           1       input2 = torch.randn(20, 16, 50)
    46    112.9 MiB      0.0 MiB           1       lora_conv1d = lora_layers.Conv1d(16, 33, 3)
    47    112.9 MiB      0.0 MiB           1       lora_conv1d(input2)
    48
    49    112.9 MiB      0.0 MiB           1       time_end4 = time.time()
    50    112.9 MiB      0.0 MiB           1       time_sum4 = time_end4 - time_start4
    51    112.9 MiB      0.0 MiB           1       print('lora_Conv1d time:',time_sum4)


Conv2d time: 0.041425228118896484

     7    112.9 MiB    112.9 MiB           1   @profile
     8                                         def c2d():
     9    112.9 MiB      0.0 MiB           1       time_start1 = time.time()
    10    112.9 MiB      0.0 MiB           1       conv2d = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3)
    11    119.0 MiB      6.1 MiB           1       input = torch.randn(20, 16, 50, 100)
    12    119.1 MiB      0.1 MiB           1       conv2d(input)
    13
    14    119.1 MiB      0.0 MiB           1       time_end1 = time.time()
    15    119.1 MiB      0.0 MiB           1       time_sum1 = time_end1 - time_start1
    16    119.1 MiB      0.0 MiB           1       print('Conv2d time:',time_sum1)


lora_Conv2d time: 0.032353878021240234

    19    113.0 MiB    113.0 MiB           1   @profile
    20                                         def lc2d():
    21    113.0 MiB      0.0 MiB           1       time_start2 = time.time()
    22    119.1 MiB      6.1 MiB           1       input = torch.randn(20, 16, 50, 100)
    23    119.3 MiB      0.1 MiB           1       lora_conv2d = lora_layers.Conv2d(in_channels=16,out_channels=64,kernel_size=3, r=4)
    24    120.0 MiB      0.0 MiB           1       lora_conv2d(input)
    25
    26    120.0 MiB      0.0 MiB           1       time_end2 = time.time()
    27    120.0 MiB      0.0 MiB           1       time_sum2 = time_end2 - time_start2
    28    120.0 MiB      0.0 MiB           1       print('lora_Conv2d time:',time_sum2)


Linear time: 0.0029897689819335938

    54    113.9 MiB    113.9 MiB           1   @profile
    55                                         def lin():
    56    113.9 MiB      0.0 MiB           1       time_start5 = time.time()
    57    113.9 MiB      0.0 MiB           1       input3 = torch.randn(100, 200)
    58    114.3 MiB      0.4 MiB           1       linear = nn.Linear(200, 500)
    59    114.9 MiB      0.6 MiB           1       linear(input3)
    60
    61    114.9 MiB      0.0 MiB           1       time_end5 = time.time()
    62    114.9 MiB      0.0 MiB           1       time_sum5 = time_end5 - time_start5  
    63    114.9 MiB      0.0 MiB           1       print('Linear time:',time_sum5)


lora_Linear time: 0.001997709274291992

    66    114.9 MiB    114.9 MiB           1   @profile
    67                                         def llin():
    68    114.9 MiB      0.0 MiB           1       time_start6 = time.time()
    69    114.9 MiB      0.0 MiB           1       input3 = torch.randn(100, 200)
    70    114.9 MiB      0.0 MiB           1       lora_linear = lora_layers.Linear(200, 500)
    71    114.9 MiB      0.0 MiB           1       lora_linear(input3)
    72 
不出所料

还真的可以降低计算成本，内存消耗。



接下来就是实战，比如alpaca-lora、某些基于Transformer的模型。尝试采用LoRA降低大模型训练成本。

未完待续。。。



### 参考资料

https://arxiv.org/pdf/2106.09685.pdf

https://github.com/microsoft/LoRA/

https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

https://blog.csdn.net/u012193416/article/details/129427242

https://blog.csdn.net/emphmeral/article/details/129184347

https://blog.csdn.net/weixin_44826203/article/details/129733930

