# CNNs from Scratch in PyTorch

This repository contains a code to build PyTorch famous CNN architectures.

The notebooks were created for educational purposes, they show the architecures in detail and trace the inputs and the outputs of the networks.

Architectures in this repository:
* VGG16
* ResNet18

The dataloader loads both CIFAR10 and CIFAR100 datasets and the code was run on Amazon SageMaker `ml.g4dn.xlarge` instance with the following hardware specifications:

```
CPU model name	: Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz
CPU Count:  4
GPU Model:  Tesla T4
```
The models were built using Pytorch `Sequential` function and Python `OrderedDict`.

## Transfer Learning
These two models are identical to PyTorch impledmentation architecture. I copied the `IMAGENET1K_V1` pre-trained weights from PyTorch implementation and using the transfer learning technique, I trained the models to achieve higher performance scores.

## Accelerated Computing
The architecture parts are well-named for educational purposes.

The code takes advantage of GPU computing and the dataloader is optimized to use the maximum number of processor cores and the GPU for transformations.

## Resources
The code was inspired by the following two blog posts:  
* [Writing ResNet from Scratch in PyTorch](https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/)  
* [Writing VGG from Scratch in PyTorch](https://blog.paperspace.com/vgg-from-scratch-pytorch/)
