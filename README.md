# COMI : Correction of Out-of-focus Microscopic Images by Deep Learning 

Overview of CycleGAN-based deep learning for high resolution microscopic image reconstruction. The proposed method contains two generators (Source Generator and Target Generator)and two discriminators(Source Discriminator and Target Discriminator). Source Generator translates out-of-focus image to in-focus image and Target Discriminator tries to distinguish real in-focus image and generated in-focus image.Target Generator translates in-focus image to out-of-focus image and Source Discriminator tries to distinguish real out-of-focus image and generated out-of-focus image.

![](https://github.com/jiangdat/COMI/raw/main/figure/figure2.png)




As the figure below shows,A contains a pair of images which represent In-focus and out-of-focus image of Leishmania.B contains In-focus and out-of-focus images of nucleus, actin and mitochondria, we collected these by a confocal microscope in Z stack module. 

![](https://github.com/jiangdat/COMI/raw/main/figure/figure1.png)




Leishmania microscopic images deblur results from different methods. Our method significantly improves the quality of out-of-focus blurred images and generates relatively sharp deblurred compared to original in-focus images. Details, like flagella, can be visible, demonstrating the generalizability of our network on a new type of sample (Leishmania parasite) that it has never seen before.

![](https://github.com/jiangdat/COMI/raw/main/figure/figure3.png)
## Prerequisites

## Installation

## Train

## Test

## Apply a pre-trained model

## Datasets
