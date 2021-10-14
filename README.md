# PyTorch-Basic

## 第三章
### 3.1 神经网络基本组成

***卷积层***
```
from torch import nn
# 搭建卷积层
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
```
