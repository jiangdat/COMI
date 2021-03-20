# COMI : Correction of Out-of-focus Microscopic Images by Deep Learning 

Overview of CycleGAN-based deep learning for high resolution microscopic image reconstruction. The proposed method contains two generators (Source Generator and Target Generator)and two discriminators(Source Discriminator and Target Discriminator). 

Source Generator translates out-of-focus image to in-focus image and Target Discriminator tries to distinguish real in-focus image and generated in-focus image.

Target Generator translates in-focus image to out-of-focus image and Source Discriminator tries to distinguish real out-of-focus image and generated out-of-focus image.

![](https://github.com/jiangdat/COMI/raw/main/figure/figure2.png)






Leishmania microscopic images deblur results from different methods. Our method significantly improves the quality of out-of-focus blurred images and generates relatively sharp deblurred compared to original in-focus images. Details, like flagella, can be visible, demonstrating the generalizability of our network on a new type of sample (Leishmania parasite) that it has never seen before.

![](https://github.com/jiangdat/COMI/raw/main/figure/figure3.png)






Deep-learning-based images reconstruction for subcellular structures. Out-of-focus images come from layer 1 and in-focus images come from layer 4. A. Nucleus. B. Actin. C. Mitochondria. The in-focus image has a very sharp structure, while the out-of-focus images are blurred and noisy. Obviously, the images corrected by our method is very similar to the in-focus image in terms of smoothness, continuity, and thickness. Experiments were repeated, achieving similar results.

![](https://github.com/jiangdat/COMI/raw/main/figure/figure4.png)






## Prerequisites
#### Python ≥ 3.6 

#### Keras ≥ 2.2.4 

#### Tensorflow ≥ 1.14.0

## Installation
### Clone this repo:
     git clone https://github.com/jiangdat/COMI
     cd COMI

## Train
### Download train datasets

### Train model
    python deblur.py

## Test
### Download test datasets

### Test model
    python test.py

## Apply a pre-trained model


