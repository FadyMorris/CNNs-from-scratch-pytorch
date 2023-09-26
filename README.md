# CNNs from Scratch in PyTorch

This repository contains a code to build PyTorch famous CNN architectures.

The notebooks were created for educational purposes, they show the architecures in detail and trace the inputs and the outputs of the networks.

Architectures in this repository:
* VGG
* ResNet

The dataloader loads both CIFAR10 and CIFAR100 datasets and the code was run on hardware with the following specifications:

```
CPU model name	: 12th Gen Intel(R) Core(TM) i7-12650H
CPU Count:  16
GPU Model:  NVIDIA GeForce RTX 3070 Laptop GPU
```
The models were built using Pytorch `Sequential` function and Python `OrderedDict`.

The architecture parts are well-named for educational purposes.

The code takes advantage of GPU computing and the dataloader is optimized to use the maximum number of processor cores and the GPU for transformations.

The code was inspired by the following two blog posts:  
* [Writing ResNet from Scratch in PyTorch](https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/)  
* [Writing VGG from Scratch in PyTorch](https://blog.paperspace.com/vgg-from-scratch-pytorch/)
