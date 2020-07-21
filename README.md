# Training_model_with_Data_Parallelism
Introduce several ways of data parallelism training, train model on one machine and seven GPUs.

   此repo以ImageNet数据集上的图像分类为例，介绍单机多卡以及多机多卡的训练代码。在repo( https://github.com/fourierer/classification_of_ImageNet_Resnet_Pytorch )中介绍了一部分，但不详细，这个repo将详细讲述各个并行原理。

一、单机多卡

并行训练分为模型并行和数据并行，模型并行一般适用于模型很大的情况，将模型参数分为几个部分，每块卡负责一部分模型参数的训练；数据并行是将数据放在多块卡上进行相同模型的训练。在pytorch中，数据并行主要有三种方式可以实现。

1.nn.DataParallel

​    DataParallel是几种并行训练方式中最慢的，但也是最简单的一种方式，使用DataParallel加载模型的方式如下：

```python
model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
```

（1）第一个参数是定义好的model，如model = ghost_net(width_mult=1.0)，此处的model要先load进GPU；

（2）第二个参数是$CUDA$列表，可以为$torch.device$类型，也可以是编号组成的$int$列表，如[5,6,7,8]；

（3）第三个参数是用于汇总梯度的GPU；

（4）参数默认情况下，即device_ids=None，output_device=None，此时使用所有的GPU训练，汇总梯度的GPU是第一块(gpus[0])；

（5）一台机器上有多张卡的情况下，只需更改gpus的id列表即可指定规定GPU训练，如gpus=[3,4,5,6,7]表示指定第3，4，5，6，7块GPU训练；

（6）DataParallel工具是单进程控制多个GPU，在训练时可以发现每次输出只有一行（每10次迭代输出一次精度）；

训练代码如下保存为datapallel.py，关键代码部分如下：

```python
import torch.nn as nn

def main():
  ...
  gpus = [4,5,6,7] # 指定gpu序号进行训练
  main_work(gpus=gpus,args=args)
  
def main_work(gpus,args):
  ...
  model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
  ...
```



2.torch.distributed

