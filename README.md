# PyTorch-Basic


## 第二章
### 2.3 神经网络工具箱 torch.nn 
#### 两个全连接层组成的感知机
```
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_dim, out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))

    def forward(self, x):
        x = x.matmul(self.w)
        y = x + self.b.expand_as(x)
        return y

class Perception(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Perception, self).__init__()
        self.layer1 = Linear(in_dim, hid_dim)
        self.layer2 = Linear(hid_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        y = torch.sigmoid(x)
        y = self.layer2(y)
        y = torch.sigmoid(y)
        return y
>>>import torch
>>>from perception import Perception
>>>perception=Perception(2, 3, 2)
```
#### nn.Sequential()快速搭建感知机
```
from torch import nn

class Perception(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Perception, self).__init__()
        self.layer = nn.Sequential(
          nn.Linear(in_dim, hid_dim),
          nn.Sigmoid(),
          nn.Linear(hid_dim, out_dim),
          nn.Sigmoid()
)
    def forward(self, x):
        y = self.layer(x)
        return y
```
#### 损失函数
```
>>> from torch import nn
>>> import torch.nn.functional as F
# 设置标签
>>> label = torch.Tensor([0,1,1,0]).long()
# 实例化nn中的交叉熵损失函数
>>> criterion = nn.CrossEntropyLoss()
# 调用交叉熵损失
>>> loss_nn = criterion(output, label)

# 由于F.cross_entropy是一个函数，可以直接调用，不需要实例化
>>> loss_functional = F.cross_entropy(output, label) 
```
#### 优化器 nn.optim (3层感知机)
```
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim1, hid_dim2, out_dim):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
          nn.Linear(in_dim, hid_dim1),
          nn.ReLU(),
          nn.Linear(hid_dim1, hid_dim2),
          nn.ReLU(),
          nn.Linear(hid_dim2, out_dim),
          nn.ReLU()
       )
    def forward(self, x):
        x = self.layer(x)
        return x
        
>>>from mlp import MLP
>>>from torch import optim
>>>from torch import nn
>>>model = MLP(28*28, 300, 200, 10)
>>>optimizer = optim.SGD(params = model.parameters(), lr=0.1)
>>>data = torch.randn(10, 28*28)
>>>output = model(data)
>>>label = torch.Tensor([1,0,4,7,9,3,4,5,3,2])
>>>criterion = nn.CrossEntropyLoss()
>>>loss = criterion(output, label)
#清空梯度，在每次优化前都需要进行此操作
>>>optimizer.zero_grad()
#损失反向传播
>>>loss.backward()
# 利用优化器进行梯度更新
>>>optimizer.step()
```

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
```
# 第一维表示一共有4个样本
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

### ResNet
[**残差块**](https://zh-v2.d2l.ai/chapter_convolutional-modern/resnet.html) 
```
import torch
from torch.nn import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
     
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
```

***resnet_bottleneck.py***
```
import torch.nn as nn
class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(Bottleneck, self).__init__()
        # 网路堆叠层是由1×1、3×3、1×1这3个卷积组成的，中间包含BN层
        self.bottleneck = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 1, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim, in_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
            )
        self.relu = nn.ReLU(inplace=True)
        # Downsample部分是由一个包含BN层的1×1卷积组成
        self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, 1),
                nn.BatchNorm2d(out_dim),
            )
    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        identity = self.downsample(x)
        # 将identity（恒等映射）与网络堆叠层输出进行相加，并经过ReLU后输出
        out += identity
        out = self.relu(out)
        return out
```

```
>>> import torch
>>> from resnet_bottlenect import Bottleneck
# 实例化Bottleneck，输入通道数为64，输出为256，对应第一个卷积组的第一个Bottleneck
>>> bottleneck_1_1 = Bottleneck(64, 256).cuda()
>>> bottleneck_1_1
# Bottleneck作为卷积堆叠层，包含了1×1、3×3、1×1这3个卷积层
Bottleneck(
  (bottleneck): Sequential(
    (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_
    running_stats=True)
    (2): ReLU(inplace)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
    bias=False)
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_
    stats=True)
    (5): ReLU(inplace)
    (6): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_
    running_stats=True)
  )
  (relu): ReLU(inplace)
  # 利用Downsample结构将恒等映射的通道数变为与卷积堆叠层相同，保证可以相加
  (downsample): Sequential(
    (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_
    running_stats=True)
  )
)
>>> input = torch.randn(1, 64, 56, 56).cuda()
>>> output = bottleneck_1_1(input)  # 将输入送到Bottleneck结构中
>>> input.shape
torch.Size([1, 64, 56, 56])
>>> output.shape
# 相比输入，输出的特征图分辨率没变，而通道数变为4倍
torch.Size([1, 256, 56, 56])
```

### FPN
***fpn.py***

## 目标检测

### IoU

```
def IOU( box1, box2 ):
    """
    :param box1:[x1,y1,x2,y2] 左上角的坐标与右下角的坐标
    """
    width1, height1 = abs(box1[2] - box1[0]), abs(box1[1] - box1[3])
    width2, height2 = abs(box2[2] - box2[0]), abs(box2[1] - box2[3])
    
    # 并区域的x，y的最大最小值
    x_max, x_min = max(box1[0],box1[2],box2[0],box2[2]), min(box1[0],box1[2],box2[0],box2[2])
    y_max, y_min = max(box1[1],box1[3],box2[1],box2[3]), min(box1[1],box1[3],box2[1],box2[3])
    
    iou_width = x_min + width1 + width2 - x_max
    iou_height = y_min + height1 + height2 - y_max
    
    if iou_width <= 0 or iou_height <= 0:
        iou_ratio = 0
    else:
        iou_area = iou_width * iou_height # 交集的面积
        box1_area, box2_area = width1 * height1, width2 * height2
        iou_ratio = iou_area / (box1_area + box2_area - iou_area) # 并集的面积
    return iou_ratio
    
box1 = [1,3,4,1]
box2 = [2,4,5,2]
print(IOU(box1,box2))
```

### NMS [(出处)](https://blog.csdn.net/a1103688841/article/details/89711120)
```
def nms(self, bboxes, scores, thresh=0.5):
    x1, y1 = bboxes[:,0], bboxes[:,1]
    x2, y2 = bboxes[:,2], bboxes[:,3]
    areas = (y2-y1+1)*(x2-x1+1)
    scores = bboxes[:,4]
    
    keep = []
    index = scores.argsort()[::-1]
    
    while index.size > 0:
        i = index[0] # 取出第一个方框进行和其他方框比对，看看有没有合并，第一个总是最大的
        
        keep.append(i) # keep保留的是索引值，不是分数
        # 计算交集的左上角和右下角
        x_lt, y_lt = np.maximum(x1[i], x1[index[1:]]), np.maximum(y1[i], y1[index[1:]])
        x_rb, y_rb = np.minimum(x2[i], x2[index[1:]]), np.minimum(y2[i], y2[index[1:]])
        
        # 如果两个方框相交，x_rb-x_lt和y_rb-y_lt是正的，如果两个方框不相交，x_rb-x_lt和y_rb-y_lt是负的，我们把不相交的W和H设为0.
        w, h = np.maximum(0, x_rb-x_lt+1), np.maximum(0, y_rb-y_lt+1)
        overlaps = w * h
        IoU = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        
        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框，合并这些方框只保留下分数最高的。经过排序当前我们操作的方框就是分数最高的，所以剔除其他和当前重叠度最高的方框
        idx = np.where(IoU <= thresh)[0]
        
        #把留下来框在进行NMS操作，这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框，每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[idx+1]
    
    return keep
  
```



### [锚框](https://zh-v2.d2l.ai/chapter_computer-vision/anchor.html)
#### 生成多个锚框
