# Training_model_with_Data_Parallelism
Introduce several ways of data parallelism training, train model on one machine and seven GPUs.

   此repo以ImageNet数据集上的图像分类为例，介绍单机多卡以及多机多卡的训练代码。在repo( https://github.com/fourierer/classification_of_ImageNet_Resnet_Pytorch )中介绍了一部分，但不详细，这个repo将详细讲述各个并行原理，大部分都参考了知乎链接( https://zhuanlan.zhihu.com/p/98535650 )，也会增加、删减和修改一些内容和代码，同时做一些脚本的测试。

#### 一、单机多卡

并行训练分为模型并行和数据并行，模型并行一般适用于模型很大的情况，将模型参数分为几个部分，每块卡负责一部分模型参数的训练；数据并行是将数据放在多块卡上进行相同模型的训练。在pytorch中，数据并行主要有三种方式可以实现。

##### 1.nn.DataParallel

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
  model = ghost_net(width_mult=1.0)
  torch.cuda.set_device('cuda:{}'.format(gpus[0]))
  model.cuda()
  model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
  ...
  optimizer = optim.SGD(model.parameters())

  for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

运行治疗较简单：

```
python dataparallel.py
```



##### 2.torch.distributed

​    与nn.DataParallel一个进程控制多块GPU不一样，torch.distributed是多个进程，每个进程控制一块GPU。在 API 层面，pytorch 为我们提供了 torch.distributed.launch 启动器，用于在命令行分布式地执行 python 文件。在执行过程中，启动器会将当前进程的（其实就是 GPU的）index 通过参数传递给 python，可以写一个测试脚本获得当前进程的 index：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
print(args.local_rank)
```

测试指令：

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 test.py
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 test.py 
python test.py
```

输出：

```
2
1
3
0
```

```
2
1
0
```

```
0
1
```

```
-1
```

除了第四个指令，其余的输出是进程号随机的排列，所以--local_rank实际上就是启动器传递给python的当前进程的（其实就是 GPU的）index。

接着，使用 init_process_group 设置GPU 之间通信使用的后端和端口：（这一步不清楚原理）

```python
dist.init_process_group(backend='nccl')
```

之后，使用 DistributedSampler 对数据集进行划分，将每个batch划分成几个part，在当前进程中只需要获取和rank对应的那个part进行训练：

```python
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
```

然后，使用 DistributedDataParallel 包装模型，它能为不同GPU上求得的梯度进行all reduce（即汇总不同 GPU 计算所得的梯度，并同步计算结果）。all reduce 后不同 GPU 中模型的梯度均为 all reduce 之前各 GPU 梯度的均值：

```python
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
```

最后，把数据和模型加载到当前进程使用的 GPU 中，正常进行正反向传播：

```python
torch.cuda.set_device(args.local_rank)

model.cuda()

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

汇总一下，torch.distributed 并行训练部分主要与如下代码段有关：

```python
import torch
import argparse
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')


def main():
  args = parser.parse_args()
  ngpus_per_node = torch.cuda.device_count()
  main_worker(args.local_rank, ngpus_per_node, args)
  
def main_worker(gpu, ngpus_per_node, args):
  dist.init_process_group(backend='nccl')
  
  
  torch.cuda.set_device(gpu)

  train_dataset = ...
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

  model = ...
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

  optimizer = optim.SGD(model.parameters())

  for epoch in range(100):
     for batch_idx, (data, target) in enumerate(train_loader):
       images = images.cuda(non_blocking=True)
       target = target.cuda(non_blocking=True)
       
       output = model(images)
       loss = criterion(output, target)
       
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
```

运行指令稍复杂，需要torch.distributed.launch 启动器，用于命令行分布式地执行 python 文件：

```python
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 distributed.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 distributed.py
```

或者省略执行指令中的CUDA_VISIBLE_DEVICES=1,2,3，在distributed.py脚本的导入模块的地方指定哪些GPU可见，如下：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
```



##### 3.torch.multiprocessing

​    多进程的GPU训练也可以手动使用torch.multiprocessing，绕开 torch.distributed.launch启动器。使用时，只需要调用 torch.multiprocessing.spawn，torch.multiprocessing会自动创建进程。如下面的代码所示，spawn 开启了 nprocs=ngpus_per_node个线程，每个线程执行 main_worker 并向其中传入 local_rank（当前进程 index）和 args（即 ngpus_per_node 和 args）作为参数：

```python
import torch.multiprocessing as mp
mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
```

这里，我们直接将原本需要torch.distributed.launch管理的执行内容，封装进 main_worker 函数中，其中gpu对应local_rank（当前进程 index），ngpus_per_node 对应mp.spawn参数args中的ngpus_per_node， args对应mp.spawn参数args中的args：

```python
def main_worker(gpu, ngpus_per_node, args):
```

由于没有 torch.distributed.launch 读取的默认环境变量作为配置，在main_worker函数中需要手动为init_process_group指定参数：（和distributed.py一样，不清楚dist.init_process_grocess_group的含义）

```python
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=ngpus_per_node, rank=gpu)
```

汇总一下，使用multiprocessing并行训练部分主要与如下代码段有关：

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
def main():
  ngpus_per_node = torch.cuda.device_count()
  mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):

   dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=ngpus_per_node, rank=gpu)
   torch.cuda.set_device(gpu)

   train_dataset = ...
   train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

   model = ...
   model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

   optimizer = optim.SGD(model.parameters())

   for epoch in range(100):
      for batch_idx, (data, target) in enumerate(train_loader):
          images = images.cuda(non_blocking=True)
          target = target.cuda(non_blocking=True)
          ...
          output = model(images)
          loss = criterion(output, target)
          ...
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
```

运行指令：

```shell
python multiprocessing_distributed.py
```

这里直接运行默认使用所有的GPU，是可以跑通的；但当制定特定的GPU训练时会报错，如：

```shell
CUDA_VISIBLE_DEVICES=1,2,3 python multiprocessing_distributed.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python multiprocessing_distributed.py
```

或者省略执行指令中的CUDA_VISIBLE_DEVICES=1,2,3，在multiprocessing_distributed.py脚本的导入模块的地方指定哪些GPU可见，如下：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
```

这两种方式都会报错，具体原因需要进一步研究。



#### 二、多机多卡

​    这部分直接介绍pytorch官方给出的训练代码main_pytorch.py，链接（ https://github.com/pytorch/examples/blob/master/imagenet/main.py ）。这个脚本是为了集群训练而写的，功能非常强大，由于本人没有多个服务器做尝试，这里利用该脚本做个单机多卡的训练。

使用main_pytorch.py脚本做单机多卡训练，阅读代码可以发现实际上是第一部分单机多卡训练中nn.DataParallel,nn.distributed和torch.multiprocesing三个工具的结合体，通过设置参数达到选择的目的。

给出主要代码部分：

```python
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/data/mhy/imagenet',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
```

运行指令：

```shell
CUDA_VISIBLE_DEVICES=0,3 python main_pytorch.py
```

或者直接在脚本中添加：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'
```

然后直接python main_pytorch.py

这样不设置任何的参数，实际上默认的是采用torch.dataparallel这个工具来训练。



