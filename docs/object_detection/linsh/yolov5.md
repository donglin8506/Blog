## 说明
整理yolov5s的整体结构


## 1. 第一层

输入情况
```python
配置文件中的参数："[-1, 1, 'Conv', [64, 6, 2, 2]]"

args: [64, 6, 2, 2]
gd: 0.33
dw: 0.5
n = 1

m: models.common.Conv

c1, c2 = 3, 64
args: [3, 32, 6, 2, 2] # 经过更新; 32是64 * 0.5
m_ = m(*args)
```

模块代码
```python
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True): 
        """
        c1, c2, k, s, p = 3, 32, 6, 2, 2
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
```

```python
输出情况：
t: models.common.Conv
np: 3520 (参数量)
    - 计算方式：(卷积部分参数)3*32*6*6+(Bn参数)32+32= 3520
    - 包括卷积核的参数和BN层的两个参数
输出打印情况：
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]    

from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv]     
ch: [32] # 当前循环结束后的值
```

## 2. 第二层

输入情况：
```python
配置文件中的参数："[-1, 1, 'Conv', [128, 3, 2]]"
f, n, m, args = -1, 1, 'Conv', [128, 3, 2]
m = eval(m) # models.common.Conv
n = n_ = 1

c1 = ch[-1] = 32 # 输入通道数，从上一层的输出通道数获取
c2 = args[0] = 128 # 输出通道数，从当前层的配置文件中获取, 其中的第一项
c2 = c2 * 0.5 = 64 # 因为是s版本，所以要折半

args: [32, 64, 3, 2]

```

```python
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True): 
        """
        c1, c2, k, s, p = 32, 64, 3, 2, None
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
# 参数量计算：c1 * c2 * k * k + c2

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    # k, p = 3, None 
    # p自动设置为 k的一半并向下取整
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)  # auto-pad
    return p
```

输出情况：

```python
t: 'models.common.Conv'
np: 18560 (参数量)
    - 计算方式：(卷积部分参数)32*64*3*3+(Bn参数)64+64= 18560
    - 包括卷积核的参数和BN层的两个参数
输出打印情况：
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]  

from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv, Conv]     
ch: [32, 64] # 当前循环结束后的值
```

## 3. 第三层

输入情况：
```python
配置文件中的参数："[-1, 3, 'C3', [128]]"
f, n, m, args = -1, 3, 'C3', [128]
m = eval(m) # models.common.C3
n = n_ = 1 # 这里虽然原始n等于3，但由于是s版本，3 * .33 = 1，即还是1个C3模块

c1 = ch[-1] = 32 # 输入通道数，从上一层的输出通道数获取
c2 = args[0] = 128 # 输出通道数，从当前层的配置文件中获取, 其中的第一项
c2 = c2 * 0.5 = 64 # 因为是s版本，所以要折半

args: [64, 64] # C3为什么只有通道数，没有卷积核这些参数
args: [64, 64, 1] # 执行过args.insert(2, n)的操作，所以最后一项变为1
```

```python
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # 输入参数：[64, 64, 1] 输入和输出通道数，以及当前模块的堆叠个数；其他参数使用默认
    # 1.两个1x1卷积分成两部分；2.一部分经过Botteneck，一部分保持不变；3.Cat两部分；4. 最后经过一个1x1卷积
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))) # 输入和输出通道数相等且e=1，这种情况能一直堆叠。
        # self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
# C3的参数量计算：(64 * 32 * 1 * 1 + 32*2) + (64 * 32 * 1 * 1 + 32*2) + (64 * 64 * 1 * 1 + 64*2) + B = 8448 + B = 18816
class Bottleneck(nn.Module):
    # Standard bottleneck
    # 一个包含有 1x1卷积+3x3卷积+shortcut的块
    # 如果输入通道和输出通道数相等，则使用shortcut
    # 1x1卷积的输出通道数*e
    # 输入参数：[32, 32, e=1.0]
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
# Bottleneck参数量计算：(32 * 32 * 1 * 1 + 32*2) + (32 * 32 * 3 * 3 + 32*2) = 10368
```

输出情况：
```python
t: 'models.common.C3'
np: 18816 (参数量)
    - 计算方式：(64 * 32 * 1 * 1 + 32*2) + (64 * 32 * 1 * 1 + 32*2) + (64 * 64 * 1 * 1 + 64*2) + (32 * 32 * 1 * 1 + 32*2) + (32 * 32 * 3 * 3 + 32*2) = 18816
    - 同上
输出打印情况：
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2] # [c1, c2, k, s, p]
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2] # [c1, c2, k, s]               
  2                -1  1     18816  models.common.C3                        [64, 64, 1] # [c1, c2, n]

from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv, Conv, C3]     
ch: [32, 64, 64] # 当前循环结束后的值
```

## 4. 第四层

输入情况：
```python
配置文件中的参数："[-1, 1, 'Conv', [256, 3, 2]]"
f, n, m, args = -1, 1, 'Conv', [256, 3, 2]
m = eval(m) # models.common.Conv
n = n_ = 1  

c1 = ch[-1] = 64 # 输入通道数，从上一层的输出通道数获取
c2 = args[0] = 256 # 输出通道数，从当前层的配置文件中获取, 其中的第一项
c2 = c2 * 0.5 = 128 # 因为是s版本，所以要折半

args: [64, 128] # C3为什么只有通道数，没有卷积核这些参数
args: [64, 128, 3, 2] 
```

```python
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True): 
        """
        c1, c2, k, s, p = 64, 128, 3, 2, None
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
```

```python
t: 'models.common.Conv'
np: 18816 (参数量)
    - 计算方式：64 * 128 * 3 * 3 + 128 * 2 = 73984

输出打印情况：
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2] # [c1, c2, k, s, p]
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2] # [c1, c2, k, s]               
  2                -1  1     18816  models.common.C3                        [64, 64, 1] # [c1, c2, n]
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2] # [c1, c2, k, s]

from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv, Conv, C3, Conv]     
ch: [32, 64, 64, 128] # 当前循环结束后的值
```

## 5. 第5层

```python
配置文件中的参数："[-1, 6, 'C3', [256]]"
f, n, m, args = -1, 6, 'C3', [256]
m = eval(m) # models.common.C3
n = n_ = 6 * 0.33 = 2

c1 = ch[-1] = 128 # 输入通道数，从上一层的输出通道数获取
c2 = args[0] = 256 # 输出通道数，从当前层的配置文件中获取, 其中的第一项
c2 = c2 * 0.5 = 128 # 因为是s版本，所以要折半

args: [128, 128] # 
args: [128, 128, 2] 
```

```python
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # 输入参数：[128, 128, 2] 输入和输出通道数，以及当前模块的堆叠个数；其他参数使用默认
    # 1.两个1x1卷积分成两部分；2.一部分经过Botteneck，一部分保持不变；3.Cat两部分；4. 最后经过一个1x1卷积
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))) # 输入和输出通道数相等且e=1，这种情况能一直堆叠。
        # self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
# C3的参数量计算：(128 * 64 * 1 * 1 + 64*2) * 2 + (128 * 128 * 1 * 1 + 128*2) + B = 33280 + B = 115712
class Bottleneck(nn.Module):
    # Standard bottleneck
    # 一个包含有 1x1卷积+3x3卷积+shortcut的块
    # 如果输入通道和输出通道数相等，则使用shortcut
    # 1x1卷积的输出通道数*e
    # 输入参数：[128, 128, e=1.0]
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
# Bottleneck参数量计算：(64 * 64 * 1 * 1 + 64*2) + (64 * 64 * 3 * 3 + 64*2) = 41216
# 两个Bottleneck : 41216 * 2 = 82432
```

```python
t: 'models.common.C3'
np: 115712 (参数量)
    - 计算方式：上面

输出打印情况：
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2] # [c1, c2, k, s, p]
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2] # [c1, c2, k, s]               
  2                -1  1     18816  models.common.C3                        [64, 64, 1] # [c1, c2, n]
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2] # [c1, c2, k, s]
  4                -1  2    115712  models.common.C3                        [128, 128, 2] # [c1, c2, n] n=2
from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv, Conv, C3, Conv, C3]     
ch: [32, 64, 64, 128, 128] # 当前循环结束后的值
```

## 6. 第6层

```python
配置文件中的参数："[-1, 1, 'Conv', [512, 3, 2]]"
f, n, m, args = -1, 1, 'Conv', [512, 3, 2]
m = eval(m) # models.common.Conv
n = n_ = 1

c1 = ch[-1] = 128 # 输入通道数，从上一层的输出通道数获取
c2 = args[0] = 512 # 输出通道数，从当前层的配置文件中获取, 其中的第一项
c2 = c2 * 0.5 = 256 # 因为是s版本，所以要折半

args: [512, 3, 2] # 
args: [128, 256, 3, 2] 
```

```python
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True): 
        """
        c1, c2, k, s, p = 128, 256, 3, 2, None
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
# 参数量： 128 * 256 * 3 * 3 + 256 * 2 = 295424
```

```python
t: 'models.common.Conv'
np: 295424 (参数量)
    - 计算方式：上面

输出打印情况：
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2] # [c1, c2, k, s, p]
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2] # [c1, c2, k, s]               
  2                -1  1     18816  models.common.C3                        [64, 64, 1] # [c1, c2, n]
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2] # [c1, c2, k, s]
  4                -1  2    115712  models.common.C3                        [128, 128, 2] # [c1, c2, n] n=2
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2] # [c1, c2, k, s]  
from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv, Conv, C3, Conv, C3, Conv]     
ch: [32, 64, 64, 128, 128, 256] # 当前循环结束后的值
```

## 7. 第7层

```python
配置文件中的参数："[-1, 9, 'C3', [512]]"
f, n, m, args = -1, 9, 'C3', [512]
m = eval(m) # models.common.C3
n = n_ = 9 * 0.33 = 3

c1 = ch[-1] = 256 # 输入通道数，从上一层的输出通道数获取
c2 = args[0] = 512 # 输出通道数，从当前层的配置文件中获取, 其中的第一项
c2 = c2 * 0.5 = 256 # 因为是s版本，所以要折半

args: [256, 256] # 
args: [256, 256, 3] 
```

```python
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # 输入参数：[256, 256, 3] 输入和输出通道数，以及当前模块的堆叠个数；其他参数使用默认
    # 1.两个1x1卷积分成两部分；2.一部分经过Botteneck，一部分保持不变；3.Cat两部分；4. 最后经过一个1x1卷积
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))) # 输入和输出通道数相等且e=1，这种情况能一直堆叠。
        # self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
# C3的参数量计算：(256 * 128 * 1 * 1 + 128*2) * 2 + (256 * 256 * 1 * 1 + 256*2) + B = 132096 + B = 625152
class Bottleneck(nn.Module):
    # Standard bottleneck
    # 一个包含有 1x1卷积+3x3卷积+shortcut的块
    # 如果输入通道和输出通道数相等，则使用shortcut
    # 1x1卷积的输出通道数*e
    # 输入参数：[128, 128, e=1.0]
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
# Bottleneck参数量计算：(128 * 128 * 1 * 1 + 128*2) + (128 * 128 * 3 * 3 + 128*2) = 164352
# 两个Bottleneck : 164352 * 3 = 493056
```

```python
t: 'models.common.C3'
np: 625152 (参数量)
    - 计算方式：上面

输出打印情况：
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2] # [c1, c2, k, s, p]
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2] # [c1, c2, k, s]               
  2                -1  1     18816  models.common.C3                        [64, 64, 1] # [c1, c2, n]
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2] # [c1, c2, k, s]
  4                -1  2    115712  models.common.C3                        [128, 128, 2] # [c1, c2, n] n=2
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2] # [c1, c2, k, s]  
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv, Conv, C3, Conv, C3, Conv, C3]     
ch: [32, 64, 64, 128, 128, 256, 256] # 当前循环结束后的值
```

## 8. 第8层

```python
配置文件中的参数："[-1, 1, 'Conv', [1024, 3, 2]]"
f, n, m, args = -1, 1, 'Conv', [1024]
m = eval(m) # models.common.Conv
n = n_ = 1

c1 = ch[-1] = 256 # 输入通道数，从上一层的输出通道数获取
c2 = args[0] = 1024 # 输出通道数，从当前层的配置文件中获取, 其中的第一项
c2 = c2 * 0.5 = 512 # 因为是s版本，所以要折半

args: [256, 512, 3, 2] # 
```

```python
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True): 
        """
        c1, c2, k, s, p = 256, 512, 3, 2, None
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
# 参数量： 256 * 512 * 3 * 3 + 512 * 2 = 1180672
```

```python
t: 'models.common.Conv'
np: 1180672 (参数量)
    - 计算方式：上面

输出打印情况：
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]          
from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv, Conv, C3, Conv, C3, Conv, C3, Conv]     
ch: [32, 64, 64, 128, 128, 256, 256, 512] # 当前循环结束后的值
```

## 9. 第9层

```python
配置文件中的参数："[-1, 3, 'C3', [1024]]"
f, n, m, args = -1, 3, 'C3', [1024]]
m = eval(m) # models.common.C3
n = n_ = 3 * 0.33 = 1

c1 = ch[-1] = 512 # 输入通道数，从上一层的输出通道数获取
c2 = args[0] = 1024 # 输出通道数，从当前层的配置文件中获取, 其中的第一项
c2 = c2 * 0.5 = 512 # 因为是s版本，所以要折半

args: [512, 512] # 
args: [512, 512, 1] 
```

```python
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # 输入参数：[512, 512, 1] 输入和输出通道数，以及当前模块的堆叠个数；其他参数使用默认
    # 1.两个1x1卷积分成两部分；2.一部分经过Botteneck，一部分保持不变；3.Cat两部分；4. 最后经过一个1x1卷积
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))) # 输入和输出通道数相等且e=1，这种情况能一直堆叠。
        # self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
# C3的参数量计算：(512 * 256 * 1 * 1 + 256*2) * 2 + (512 * 512 * 1 * 1 + 512*2) + B = 526336 + B = 
class Bottleneck(nn.Module):
    # Standard bottleneck
    # 一个包含有 1x1卷积+3x3卷积+shortcut的块
    # 如果输入通道和输出通道数相等，则使用shortcut
    # 1x1卷积的输出通道数*e
    # 输入参数：[256, 256, e=1.0]
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
# Bottleneck参数量计算：(256 * 256 * 1 * 1 + 256*2) + (256 * 256 * 3 * 3 + 256*2) = 656384
```
```python
t: 'models.common.C3'
np: 1182720 (参数量)
    - 计算方式：上面

输出打印情况：
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                            
from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv, Conv, C3, Conv, C3, Conv, C3, Conv, C3]     
ch: [32, 64, 64, 128, 128, 256, 256, 512, 512] # 当前循环结束后的值
```

## 10. 第10层

是一种新层 SPPF

```python
配置文件中的参数："[-1, 1, 'SPPF', [1024, 5]]"
f, n, m, args = -1, 1, 'SPPF', [1024, 5]]
m = eval(m) # models.common.SPPF
n = n_ = 3 * 0.33 = 1

c1 = ch[-1] = 512 # 输入通道数，从上一层的输出通道数获取
c2 = args[0] = 1024 # 输出通道数，从当前层的配置文件中获取, 其中的第一项
c2 = c2 * 0.5 = 512 # 因为是s版本，所以要折半

args: [512, 512, 5] 
```

```python
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    # [512, 512, 5] 
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# 参数量：(512 * 256 * 1 * 1 + 256 * 2) + (1024 * 512 * 1 * 1 + 512 * 2) = 656896
```

```python
t: 'models.common.SPPF'
np: 656896 (参数量)
    - 计算方式：上面

输出打印情况：
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                  
from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv, Conv, C3, Conv, C3, Conv, C3, Conv, C3, SPPF]     
ch: [32, 64, 64, 128, 128, 256, 256, 512, 512, 512] # 当前循环结束后的值
```


## 11. 第11层

输入参数：
```python
f, n, m, args = [-1, 1, 'Conv', [512, 1, 1]]
i = 10

c1 = ch[f] = 512
c2 = args[0] * 0.5 = 256
args = [512, 256, 1, 1]
t = 'models.common.Conv'
np = 131584
```

```python
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True): 
        """
        c1, c2, k, s, p = 512, 256, 1, 1, None
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
# 参数量： 512 * 256 * 1 * 1 + 256 * 2 = 131584
```

```python
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1] # [c1, c2, k, s]
 ```

 ```python
 from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv, Conv, C3, Conv, C3, Conv, C3, Conv, C3, SPPF, Conv]     
ch: [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256] # 当前循环结束后的值
```

## 12. 第12层

Upsample层没有参数量，
```python
i = 11
f, n, m, args = -1, 1, 'nn.Upsample', ['None', 2, 'nearest']

n = n_ = 1
c1 = 512 # c1没有发生变化
c2 = ch[f] # 256
t = 'torch.nn.modules.upsampling.Upsample'
```

```python
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']  

from: -1 # 含义是当前层的输入来自上一层的输出
save: [] # 因为当前层不会进行推理，所以不必保存
layers: [Conv, Conv, C3, Conv, C3, Conv, C3, Conv, C3, SPPF, Conv, Upsample]     
ch: [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256, 256] # 当前循环结束后的值

```

## 13. 第13层

Concat层没有参数
```python
i = 12 
f, n, m, args = [[-1, 6], 1, 'Concat', [1]]
```

```python
elif m in Concat:
    c2 = sum(ch[x] for x in f) = sum(ch[-1], ch[6]) = 512

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x):
        return torch.cat(x, self.d)
```

```python
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1] 

from: [-1, 6] # 输入来自上一层和索引第6层
save: [6] # 第一个不等于-1的值
layers: [Conv, Conv, C3, Conv, C3, Conv, C3, Conv, C3, SPPF, Conv, Upsample, Concat]     
ch: [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256, 256, 512] # 当前循环结束后的值
```


## 14. 第14层

```python
i = 13
f, n, m, args = -1, 3, 'C3', [512, False]

args = [512, 256, False]
args = [512, 256, 1, False]
```

```python
layers: [Conv, Conv, C3, Conv, C3, Conv, C3, Conv, C3, SPPF, Conv, Upsample, Concat, C3] 
ch: [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256, 256, 512, 256]
```

## 15. 第15层

```python
i = 14
f, n, m, args = -1, 1, 'Conv', [256, 1, 1]

args = [256, 128, 1, 1]
```

```python
layers: [Conv, Conv, C3, Conv, C3, Conv, C3, Conv, C3, SPPF, Conv, Upsample, Concat, C3, Conv] 
ch: [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256, 256, 512, 256, 128]
```

## 16. 第16层

```python
i = 15
f, n, m, args = -1, 1, 'nn.Upsamle', ['None', 2, 'nearest']

```

```python
layers: [Conv, Conv, C3, Conv, C3, Conv, C3, Conv, C3, SPPF, Conv, Upsample, Concat, C3, Upsample] 
ch: [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256, 256, 512, 256, 128, 128]
```

## 17. 第17层

```python
i = 16
f, n, m, args = [[-1, 4], 1, 'Concat', [1]]

```

```python
layers: [Conv, Conv, C3, Conv, C3, Conv, C3, Conv, C3, SPPF, Conv, Upsample, Concat, C3, Upsample, Concat] 
ch: [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256, 256, 512, 256, 128, 128]
```


## 最后一层
Detect

```python
i = 24
f, n, m, args = [17, 20, 23], 1, 'Detect', ['nc', 'anchors']

args = [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]]

args = [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [ch[17], ch[20], ch[23]]]

```

```python
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        # nc = 80; anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        # ch = [128, 256, 512]
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

self.nc = 80
self.no = nc + 5 = 85
self.nl = len(anchors) = 3
self.na = len(anchors[0]) // 2 = 3
self.grid = [torch.zeros(1)] * self.nl = [tensor([0.]), tensor([0.], tensor([0.]))] 
self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2)) # 第一项是层数nl；每层可以有多个anchor；但每个anchor都是有两个值
# [[[10, 13],
    [16, 30],
    [33, 23]],
   [[30, 61],
    [62, 45],
    [59, 119]],
   [[116, 90],
    [156, 198],
    [373, 326]]]
# shape (3, 3, 2)
```


-------------------------------------

模型的推理过程：

```python
def _forward_once(self, x):
    y, dt = [], []
    for m in self.model:
        if m.f != -1:
            x = y[m.f] if isinstance(x, int)
        x = m(x)
        y.append(x if m.i in self.save else None)
    return x
```

```python
输入是[tensor(1,128,32,32), tensor(1, 256, 16, 16), tensor(1, 512, 8, 8)]
```

```python
class Detect(nn.Module):
    ...
    def forward(self, x):
        # 输入是[tensor(1,128,32,32), tensor(1, 256, 16, 16), tensor(1, 512, 8, 8)]
        z = []
        for i in range(self.nl): # 几层特征图，3层
            x[i] = self.m[i](x[i]) # self.m[i] 由于self.m是一个nn.ModuleList，self.m[i]是其第i项；将第特征图tensor(1,128,32,32)传入到self.m[0]中; shape: [1, 128, 32, 32] -> [1, 255, 32, 32]
            bs, _, ny, nx = x[i].shape # x[0].shape : [1, 255, 32, 32]
            # [1, 255, 32, 32] -> [1, 3, 85, 32, 32] -> [1, 3, 32, 32, 85]
            # 1: 一张图片的特征图
            # 3: 包括3种anchor
            # 32, 32: 32 x 32每个点上都有这么3种anchor
            # 85: 每个anchor有85种输出
            x[i] = x[i].view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()

i = 0 ; [1, 128, 32, 32] -> [1, 255, 32, 32] -> [1, 3, 32, 32, 85]
i = 1 ; [1, 256, 16, 16] -> [1, 255, 16, 16] -> [1, 3, 16, 16, 85]
i = 2 ; [1, 512, 8, 8]   -> [1, 255, 8, 8]   -> [1, 3, 8, 8, 85]
```

