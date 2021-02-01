# AutoEncoders & GANs

Experimentations with AutoEncoders and GANs, implementations in TensorFlow 2.0.

The notebooks are used to test the models on the test sets and to visualize the results.
The train.py script is used to train the available models.

>python train.py -h

Implemented models:
* AutoEncoder (Standard implementation)
* U-Net [Ronneberger 2015](https://arxiv.org/pdf/1505.04597.pdf)
* (In progress) SRCNN [Dong 2015](https://arxiv.org/pdf/1501.00092.pdf)
* (In progress) DnCNN [Zhang 2016](https://arxiv.org/pdf/1608.03981.pdf)
* (In progress) DCGAN [Radford 2016](https://arxiv.org/pdf/1511.06434.pdf)
* (In progress) Deep Image Prior [Ulyanov 2020](https://arxiv.org/pdf/1711.10925v4.pdf)


Used datasets:
* MNIST
* Fashion MNIST
* CIFAR
* Labeled Faces in the Wild (LFW)

## AutoEncoder

This model is just a standard encoder decoder.

## U-Net

My implementation of the U-Net is a lighter version of the one from the paper (a total of 11 conv layers against 23 originaly).
The paper implementation had a segmentation goal, whereas mine has an image quality improving goal.
The architecture of the U-Net makes use of skip connection that transfers low-level information to high-level layers, in the objective to improve the quality of output images.