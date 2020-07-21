# Training_model_with_Data_Parallelism
Introduce several ways of data parallelism training, train model on one machine and seven GPUs.

   此repo以ImageNet数据集上的图像分类为例，介绍单机多卡以及多机多卡的训练代码。在repo( https://github.com/fourierer/classification_of_ImageNet_Resnet_Pytorch )中介绍了一部分，但不详细，这个repo将详细讲述各个并行原理。

一、单机多卡

并行训练分为模型并行和数据并行，模型并行一般适用于模型很大的情况，将模型参数分为几个部分，每块卡负责一部分模型参数的训练；数据并行是将数据放在多块卡上进行相同模型的训练。在pytorch中，数据并行主要有三种方式可以实现。

1.

