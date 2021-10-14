# PyTorch-Basic

## 第三章
### 3.1 神经网络基本组成
***卷积层***
```
from torch import nn
# 搭建卷积层
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
```
***激活函数层***
```
input = torch.randn(1,1,2,2)
>>> input
sigmoid = nn.Sigmoid()
>>> sigmoid(input)

# nn.ReLU()可以实现inplace操作，即可以直接将运算结果覆盖到输入中，以节省内存
relu = nn.ReLU(inplace=True)
>>> relu(input)

leakyrelu = nn.LeakyReLU(0.04, True) #True代表in-place操作
>>> leakyrelu(input)
```

Softmax函数
```
import torch.nn.functional as F
score = torch.randn(1, 4)
>>> score
F.softmax(score, 1) #第二个参数表示按照第几个维度进行
```

***池化层***
```
# 池化主要需要两个参数，第一个参数代表池化区域大小，第二个参数表示步长
max_pooling = nn.MaxPool2d(2, stride=2)
aver_pooling = nn.AvgPool2d(2, stride=2)
input = torch.randn(1,1,4,4)
>>> input
>>> max_pooling(input)
>>> aver_pooling(input)
```

***Dropout层***
```
dropout = nn.Dropout(0.5, inplace=False)
input = torch.randn(2, 64, 7, 7)
output = dropout(input)
```

***BN层***
```
# 使用BN层需要传入一个参数为num_features，即特征的通道数
bn = nn.BatchNorm2d(64)
>>> bn
>>> input = torch.randn(4, 64, 224, 224)
>>> output = bn(input)
```

***全连接层***
# 第一维表示一共有4个样本
```
>>> input = torch.randn(4, 1024)
>>> linear = nn.Linear(1024, 4096)
>>> output = linear(input)
>>> input.shape
torch.Size([4, 1024])
>>> output.shape
torch.Size([4, 4096])
```

### 3.2 走向深度：VGGNet
***vgg.py***
```
from torch import nn
class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        layers = []
        in_dim = 3
        out_dim = 64
        # 循环构造卷积层，一共有13个卷积层
        for i in range(13):
            layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dim = out_dim
            # 在第2、4、7、10、13个卷积层后增加池化层
            if i==1 or i==3 or i==6 or i==9 or i==12:
                layers += [nn.MaxPool2d(2,2)]
                # 第10个卷积后保持和前边的通道数一致，都为512，其余加倍
                if i!=9:
                    out_dim *= 2
        self.features = nn.Sequential(*layers)
        # VGGNet的3个全连接层，中间有ReLU与Dropout层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        # 这里是将特征图的维度从[1, 512, 7, 7]变到[1, 512*7*7]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#调用
>>> import torch
>>> from vgg import VGG
# 实例化VGG类，在此设置输出分类数为21，并转移到GPU上
>>> vgg = VGG(21).cuda()
>>> input = torch.randn(1, 3, 224, 224).cuda()
>>> input.shape
torch.Size([1, 3, 224, 224])
# 调用VGG，输出21类的得分
>>> scores = vgg(input)
>>> scores.shape
torch.Size([1, 21])
# 也可以单独调用卷积模块，输出最后一层的特征图
>>> features = vgg.features(input)
>>> features.shape
torch.Size([1, 512, 7, 7])
# 打印出VGGNet的卷积层，5个卷积组一共30层
>>> vgg.features

# 打印出VGGNet的3个全连接层
>>> vgg.classifier
```
