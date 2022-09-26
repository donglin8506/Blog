

`parameters()`的使用

类`Conv``来自yolov5中常用的一个组件，依次为例，我们感受下pytorch模块中parameters参数的使用。可以发现：

- 继承nn.Module的自定义类的实例对象，具有`parameters()`方法；
- `model.parameters()`返回的是生成器对象
- `model.parameters()`包含这个类中可学习的参数，包括卷积核的参数和BN层的参数

```python
import torch
from torch import nn

class Conv(nn.Module):
    def __init__(self, c1=3, c2=64, k=6, s=2, p=2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

if __name__ == "__main__":
    model = Conv()
    print(model.parameters()) # <generator object Module.parameters at 0x7fa68016f850>

    for idx, para in enumerate(model.parameters()):
        print("============", idx)
        print(para)
        print("shape: ", para.shape)
        print("total paras: ", para.numel()) # 获取para的参数总个数
```

```
============ 0
Parameter containing:
tensor([[[[-4.9318e-02,  6.9207e-02, -6.3467e-02, -6.5287e-02,  7.6723e-02,
           -3.8378e-04],
          [ 7.2446e-02,  3.1003e-02,  8.9773e-02, -8.0289e-02,  6.2464e-02,
           -7.7457e-02],
          [-5.8755e-02, -3.6876e-02,  1.6566e-02,  5.2497e-02,  5.6697e-02,
           -9.5655e-03],
          [ 7.1956e-02,  6.3590e-02,  2.7415e-02, -6.0436e-02,  3.3391e-02,
            6.7367e-02],
          [ 2.1894e-02, -5.5162e-02,  8.6821e-03, -7.9369e-02, -7.6835e-02,
            8.5203e-03],
          [ 2.0913e-02,  1.5397e-02, -2.7012e-02,  4.1439e-02,  9.0220e-02,
           -9.3959e-02]],

        ...,

         [[ 8.0491e-02, -7.4360e-02,  5.7617e-03,  7.9103e-03,  9.0219e-02,
           -4.4207e-03],
          [ 8.2175e-02,  2.4835e-02,  4.7966e-02, -2.8035e-02,  8.0372e-02,
            4.0228e-02],
          [ 1.4321e-02,  3.6747e-02, -1.5620e-02,  7.2155e-03, -2.5480e-02,
           -2.4253e-02],
          [ 2.9245e-02, -6.4244e-02, -5.4636e-03,  2.8508e-02, -7.1834e-02,
           -7.2294e-02],
          [ 1.3893e-02, -8.6789e-02, -2.7554e-02, -5.6497e-02,  3.6276e-02,
            8.4411e-02],
          [-1.0794e-02,  8.5295e-03,  7.9786e-02, -1.3315e-02, -9.4185e-02,
            7.1432e-02]]]], requires_grad=True)
shape:  torch.Size([64, 3, 6, 6])
============ 1
Parameter containing:
tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], requires_grad=True)
shape:  torch.Size([64])
============ 2
Parameter containing:
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       requires_grad=True)
shape:  torch.Size([64])
```