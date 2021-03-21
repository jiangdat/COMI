# COMI : Correction of Out-of-focus Microscopic Images by Deep Learning 
   Microscopic images are widely used in scientific research and medical discovery. However, images obtained by low cost microscope are often out-of-focus, resulting poor performance in research and diagnosis. We present a Cycle Generative Adversarial Network (CycleGAN) based model and a multi-component weighted loss function to address this issue. The CycleGAN learns the mapping from the low-quality out-of-focus images to the corresponding high-quality in-focus images to achieve the correction for out-of-focus microscopic image. The proposed model reached state-of-the-art performance in correction and generation of the visually comfortable images compared to other methods. Our method shows good generalization capabilities across diverse research fields by analyzing various cellular structures ranging from protozoan parasite to nucleus, actin and mitochondria of mammalian cells, demonstrating a great promise for bioimaging. In addition, the proposed method can also improve the accuracy of human visual diagnosis. We extend the application range of simple microscopy beyond the limitations of hardware and human operation, which will facilitate the development of cell and molecular biology research, especially in the remote areas and developing countries.  




   
   Overview of CycleGAN-based deep learning for high resolution microscopic image reconstruction. The proposed method contains two generators(Source Generator and Target Generator)and two discriminators(Source Discriminator and Target Discriminator). Source Generator translates out-of-focus image to in-focus image and Target Discriminator tries to distinguish real in-focus image and generated in-focus image.Target Generator translates in-focus image to out-of-focus image and Source Discriminator tries to distinguish real out-of-focus image and generated out-of-focus image.
![figure2.png](https://github.com/jiangdat/COMI/raw/main/figure/figure2.png)









## 1. Prerequisites
#### Python ≥ 3.6 

#### Keras ≥ 2.2.4 

#### Tensorflow ≥ 1.14.0


## 2. Installation
### Clone this repo:
     git clone https://github.com/jiangdat/COMI
     cd COMI


## 3. Datasets
  
   We collect and publish two datasets for correcting out-of-focus microscopic images, including Leishmania parasite dataset which captured by captured under two bright-field light microscopes (Leica DM2700P and Olympus IX53), and Bovine Pulmonary Artery Endothelial Cells (BPAEC) dataset which captured by a confocal laser scanning microscopy in z-stack module.
   ![table1.png](https://github.com/jiangdat/COMI/raw/main/figure/table1.png)
   
   BPAEC dataset includes 100 sets images for 3 components, which means 297 in-focus images and 1,782 out-of-focus images. Examples of each categories are shown in Figure 1B. The details of images used in this study can be summarized at Table 1.
![figure1.png](https://github.com/jiangdat/COMI/raw/main/figure/figure1.png)

#### download our datasets:
    https://data.mendeley.com/drafts/m3jxgb54c9

## 4. Train

#### set the parameter and path in file deblur.py, then using:
    python deblur.py



## 5. Test

#### set the parameter and path in file test.py, then using:
    python test.py

## 6. Apply a pre-trained model

#### download our pre-trainde model from link:

#### here can download the picture named "Out-of-focus image" in Figure3.

## 7. Results


Leishmania microscopic images deblur results from different methods. Our method significantly improves the quality of out-of-focus blurred images and generates relatively sharp deblurred compared to original in-focus images. Details, like flagella, can be visible, demonstrating the generalizability of our network on a new type of sample (Leishmania parasite) that it has never seen before.

![figure3.png](https://github.com/jiangdat/COMI/raw/main/figure/figure3.png)





Deep-learning-based images reconstruction for subcellular structures. Out-of-focus images come from layer 1 and in-focus images come from layer 4. A. Nucleus. B. Actin. C. Mitochondria. The in-focus image has a very sharp structure, while the out-of-focus images are blurred and noisy. Obviously, the images corrected by our method is very similar to the in-focus image in terms of smoothness, continuity, and thickness. Experiments were repeated, achieving similar results.

![figure4.png](https://github.com/jiangdat/COMI/raw/main/figure/figure4.png)


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
