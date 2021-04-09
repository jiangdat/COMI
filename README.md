# COMI : Correction of Out-of-focus Microscopic Images by Deep Learning 
   Microscopic images are widely used in scientific research and medical discovery. However, images obtained by low cost microscope are often out-of-focus, resulting poor performance in research and diagnosis. 
   
   We present a Cycle Generative Adversarial Network (CycleGAN) based model and a multi-component weighted loss function to address this issue. Our method shows good generalization capabilities across diverse research fields by analyzing various cellular structures ranging from protozoan parasite to nucleus, actin and mitochondria of mammalian cells, demonstrating a great promise for bioimaging. 
  
   The proposed method contains two generators(Source Generator and Target Generator)and two discriminators(Source Discriminator and Target Discriminator). Source Generator translates out-of-focus image to in-focus image and Target Discriminator tries to distinguish real in-focus image and generated in-focus image.Target Generator translates in-focus image to out-of-focus image and Source Discriminator tries to distinguish real out-of-focus image and generated out-of-focus image.
![figure2.png](https://github.com/jiangdat/COMI/raw/main/figure/figure2.png)


## 1. Prerequisites

#### Ubuntu 16.04 

#### Tesla K40C GPU

#### Python 3.6 

#### Keras 2.2.4 

#### Tensorflow 1.14.0

## 2. Installation
### Clone this repo:
     git clone https://github.com/jiangdat/COMI
     cd COMI

## 3. Datasets
  
   We collect and publish two datasets for correcting out-of-focus microscopic images, including Leishmania parasite dataset  and Bovine Pulmonary Artery Endothelial Cells (BPAEC) dataset.
   
   
![table1.png](https://github.com/jiangdat/COMI/raw/main/figure/table1.png)
   
#### download our datasets:
    https://data.mendeley.com/datasets/m3jxgb54c9/4

## 4. Train

#### set the parameter and path in deblur.py, then using:
    python deblur.py

## 5. Test

#### set the parameter and path in test.py, then using:
    python test.py

## 6. Apply a pre-trained model

#### download our pre-trainde model from link:
    https://drive.google.com/drive/folders/13R9fZ45IyPdJrq-ATHatPc_j_977qsT3?usp=sharing

the pre-trained model can be used directly for testing.


## 7. Results


Our method significantly improves the quality of out-of-focus blurred images.
   
![result of deblured image ](https://github.com/jiangdat/COMI/raw/main/figure/result_github.png)


## 8. Related Projects
### links of other models we have compared

#### link for "Unnatural L0 Sparse Representation for Natural Image Deblurring"(L0-regularized) : 
    http://www.cse.cuhk.edu.hk/~leojia/projects/l0deblur/

#### link for "Deblurgan: Blind motion deblurring using conditional adversarial networks" : 
    https://github.com/KupynOrest/DeblurGAN

#### link for "Deblurgan-v2: Deblurring (orders-of-magnitude) faster and better" : 
    https://github.com/TAMU-VITA/DeblurGANv2

#### link for "Image-to-Image Translation with Conditional Adversarial Networks"(pix2pix) : 
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

#### link for"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"(CycleGAN):
    https://junyanz.github.io/CycleGAN


## 9. Object Detection
#### For object detection, you should follow the instruction of Yolov3. Link for Yolov3 : 
    https://pjreddie.com/darknet/yolo/
