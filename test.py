from __future__ import print_function, division
import numpy as np
import os
import cv2
from PIL import Image
import random
from functools import partial

import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers.merge import _Merge
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Conv2D, BatchNormalization, UpSampling2D, Activation
from keras.layers import Reshape, Dropout, Concatenate, Lambda, Multiply, Add, Flatten, Dense
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras import backend as K
import keras
import cv2
from sklearn.utils import shuffle
import random
import datetime
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
import math
from skimage.measure import compare_psnr, compare_ssim
from keras.utils import multi_gpu_model
from scipy.stats import pearsonr

def load_confocal(input_shape=None, set=None, z_depth=None):
    dir = './confocal/' + set
    lr_lq_set = []
    hr_lq_set = []
    lr_hq_set = []
    hr_hq_set = []
    for _, _, files in os.walk(dir+'/'+z_depth):
        for file in files:
            if int(file.split('_')[-1].split('.')[0]) < len(files) * 0.8:
                img_lq = cv2.imread(dir+'/'+z_depth + '/' + file)
                img = cv2.resize(img_lq, (input_shape[0], input_shape[1]))
                lr_lq_set.append(img)
                img = cv2.resize(img_lq, (input_shape[0]*4, input_shape[1]*4))
                hr_lq_set.append(img)

                file = 'Z7_' + file.split('_')[1]
                img_hq = cv2.imread(dir+'/Z007' + '/' + file)
                img = cv2.resize(img_hq, (input_shape[0]*4, input_shape[1]*4))
                hr_hq_set.append(img)
                img = cv2.resize(img_hq, (input_shape[0], input_shape[1]))
                lr_hq_set.append(img)
    hrhq, lrhq, hrlq, lrlq = hr_hq_set, lr_hq_set, hr_lq_set, lr_lq_set

    hrhq_train = hrhq
    lrhq_train = lrhq
    hrlq_train = hrlq
    lrlq_train = lrlq

    lr_lq_set = []
    hr_lq_set = []
    lr_hq_set = []
    hr_hq_set = []
    for _, _, files in os.walk(dir+'/'+z_depth):
        for file in files:
            if int(file.split('_')[-1].split('.')[0]) >= len(files) * 0.8:
                img_lq = cv2.imread(dir+'/'+z_depth + '/' + file)
                img = cv2.resize(img_lq, (input_shape[0], input_shape[1]))
                lr_lq_set.append(img)
                img = cv2.resize(img_lq, (input_shape[0]*4, input_shape[1]*4))
                hr_lq_set.append(img)

                file = 'Z7_' + file.split('_')[1]
                img_hq = cv2.imread(dir+'/Z007' + '/' + file)
                img = cv2.resize(img_hq, (input_shape[0]*4, input_shape[1]*4))
                hr_hq_set.append(img)
                img = cv2.resize(img_hq, (input_shape[0], input_shape[1]))
                lr_hq_set.append(img)

    hrhq, lrhq, hrlq, lrlq = hr_hq_set, lr_hq_set, hr_lq_set, lr_lq_set

    hrhq_test = hrhq
    lrhq_test = lrhq
    hrlq_test = hrlq
    lrlq_test = lrlq

    hrhq_train = np.array(hrhq_train)
    hrhq_train = hrhq_train.astype('float32') /127.5 - 1.
    hrhq_test = np.array(hrhq_test)
    hrhq_test = hrhq_test.astype('float32')  /127.5 - 1.

    lrhq_train = np.array(lrhq_train)
    lrhq_train = lrhq_train.astype('float32') /127.5 - 1.
    lrhq_test = np.array(lrhq_test)
    lrhq_test = lrhq_test.astype('float32') /127.5 - 1.

    hrlq_train = np.array(hrlq_train)
    hrlq_train = hrlq_train.astype('float32') /127.5 - 1.
    hrlq_test = np.array(hrlq_test)
    hrlq_test = hrlq_test.astype('float32') /127.5 - 1.

    lrlq_train = np.array(lrlq_train)
    lrlq_train = lrlq_train.astype('float32') /127.5 - 1.
    lrlq_test = np.array(lrlq_test)
    lrlq_test = lrlq_test.astype('float32') /127.5 - 1.

    print(hrhq_train.shape)
    print(hrhq_test.shape)
    return hrhq_train, hrhq_test, lrhq_train, lrhq_test, hrlq_train, hrlq_test, lrlq_train, lrlq_test

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def define_batch_size(self, bs):
        self.bs = bs

    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.bs, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class StarGAN(object):
    def __init__(self):

        # Model configuration.
        self.channels = 3
        self.lr_height = 128                 # Low resolution height
        self.lr_width = 128                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height*4   # High resolution height
        self.hr_width = self.lr_width*4     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        self.n_residual_blocks = 9

        optimizer = Adam(0.0001, 0.5, 0.99)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg_hq = self.build_vgg_hr(name='vgg_hq')
        self.vgg_hq.trainable = False
        self.vgg_lq = self.build_vgg_hr(name='vgg_lq')
        # Calculate output shape of D (PatchGAN)
        patch_hr_h = int(self.hr_height / 2 ** 4)
        patch_hr_w = int(self.hr_width / 2 ** 4)
        self.disc_patch_hr = (patch_hr_h, patch_hr_w, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        self.discriminator_hq = self.build_discriminator(name='dis_hq')
        self.discriminator_lq = self.build_discriminator(name='dis_lq')
        # Build the generator
        self.generator_lq2hq = self.build_generator(name='gen_lq2hq')
        self.generator_hq2lq = self.build_generator(name='gen_hq2lq')

        # High res. and low res. images
        img_lq = Input(shape=self.hr_shape)
        img_hq = Input(shape=self.hr_shape)

        fake_hq = self.generator_lq2hq(img_lq)
        fake_lq = self.generator_hq2lq(img_hq)

        reconstr_lq = self.generator_hq2lq(fake_hq)
        reconstr_hq = self.generator_lq2hq(fake_lq)

        img_lq_id = self.generator_hq2lq(img_lq)
        img_hq_id = self.generator_lq2hq(img_hq)

        fake_hq_features = self.vgg_hq(fake_hq)
        fake_lq_features = self.vgg_lq(fake_lq)

        reconstr_hq_features = self.vgg_hq(reconstr_hq)
        reconstr_lq_features = self.vgg_lq(reconstr_lq)

        self.discriminator_hq.trainable = False
        self.discriminator_lq.trainable = False

        validity_hq = self.discriminator_hq(fake_hq)
        validity_lq = self.discriminator_lq(fake_lq)

        validity_reconstr_hq = self.discriminator_hq(reconstr_hq)
        validity_reconstr_lq = self.discriminator_lq(reconstr_lq)

        self.combined_hq = Model([img_lq, img_hq], [validity_hq, validity_reconstr_lq,
                                                    fake_hq_features, reconstr_lq_features, img_lq_id])
        self.combined_hq_m = multi_gpu_model(self.combined_hq, gpus=4)
        self.combined_hq_m.compile(loss=['mse', 'mse', 'mse', 'mse', 'mse'],
                                   loss_weights=[1e-3, 1e-3, 1, 1, 1],
                                   optimizer=optimizer)
        self.combined_lq = Model([img_lq, img_hq], [validity_lq, validity_reconstr_hq,
                                                    fake_lq_features, reconstr_hq_features, img_hq_id])
        self.combined_lq_m = multi_gpu_model(self.combined_lq, gpus=4)
        self.combined_lq_m.compile(loss=['mse', 'mse', 'mse', 'mse', 'mse'],
                                   loss_weights=[1e-3, 1e-3, 1, 1, 1],
                                   optimizer=optimizer)

    def build_vgg_hr(self, name=None):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """

        vgg = VGG19(include_top=False, weights="./model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
        vgg.outputs = [vgg.layers[9].output]
        img = Input(shape=self.hr_shape)

        # Extract image features
        img_features = vgg(img)
        model = Model(img, img_features, name=name)
        model.summary()
        return model


    def build_generator(self, name=None):
        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = InstanceNormalization()(d)
            d = Activation('relu')(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = InstanceNormalization()(d)
            d = Add()([d, layer_input])
            return d

        # Low resolution image input
        img_lr = Input(shape=self.hr_shape)

            # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = InstanceNormalization()(c1)
        c1 = Activation('relu')(c1)

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            c1 = Conv2D(filters=64 * mult * 2, kernel_size=(3, 3), strides=2, padding='same')(c1)
            c1 = InstanceNormalization()(c1)
            c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf * (n_downsampling ** 2))
        for _ in range(8):
            r = residual_block(r, self.gf * (n_downsampling ** 2))

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            r = UpSampling2D()(r)
            r = Conv2D(filters=int(64 * mult / 2), kernel_size=(3, 3), padding='same')(r)
            r = InstanceNormalization()(r)
            r = Activation('relu')(r)

            # Post-residual block
        c2 = Conv2D(self.channels, kernel_size=7, strides=1, padding='same')(r)
        c2 = Activation('tanh')(c2)
        c2 = Add()([c2, img_lr])
        model = Model(img_lr, [c2], name=name)


        model.summary()
        return model


    def build_discriminator(self, name=None):
        n_layers, use_sigmoid = 3, False
        inputs = Input(shape=self.hr_shape)
        ndf=64
        x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
        x = LeakyReLU(0.2)(x)

        nf_mult, nf_mult_prev = 1, 1
        for n in range(n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            x = Conv2D(filters=ndf * nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)

        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        x = Conv2D(filters=ndf * nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
        if use_sigmoid:
            x = Activation('sigmoid')(x)

        x = Dense(1024, activation='tanh')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=x, name=name)


        model.summary()
        return model

    def build_discriminator_lr(self):
        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            if bn:
                d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        # Input img
        d0 = Input(shape=self.lr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d4 = d_block(d2, self.df * 2, strides=2)

        d9 = Dense(self.df * 4)(d4)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)
        model = Model(d0, validity)

        model.summary()
        return Model(d0, validity)

    def test(self, model, epochs, batch_size, sample_interval, set=None, z_depth=None):
        input_shape = (128, 128, 3)
        start_time = datetime.datetime.now()
        weigths_dir = model + '_weights'
        img_dir = model + '_img'
        log_dir = model + '_logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_dir = model + '_predict_img'
        if not os.path.exists(weigths_dir):
            os.makedirs(weigths_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Load the dataset
        hrhq_train, hrhq_test, lrhq_train, lrhq_test, hrlq_train, hrlq_test, lrlq_train, lrlq_test = load_confocal(
            input_shape=input_shape,
            set=set, z_depth=z_depth)

        lq2hq = load_model(weigths_dir + '/' + 'generator_l2h.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
        hq2lq = load_model(weigths_dir + '/' + 'generator_h2l.h5', custom_objects={'InstanceNormalization': InstanceNormalization})

        print('original hrlq')
        self.compute(hrhq_test, hrlq_test)

        gen_hrhq = lq2hq.predict(hrlq_test, batch_size=1)
        print('save_model generate hrhq : ')
        self.compute(hrhq_test, gen_hrhq)

        gen_hrlq = hq2lq.predict(hrhq_test, batch_size=1)
        print('save_model generate hrlq : ')
        self.compute(hrlq_test, gen_hrlq)

        reconstr_hrhq = lq2hq.predict(gen_hrlq, batch_size=1)
        print('save_model reconstr hrhq : ')
        self.compute(hrhq_test, reconstr_hrhq)

        dir = './confocal/' + set + '/' + z_depth
        num = 0
        for _, _, files in os.walk(dir):
            for file in files:
                num += 1
                img = []
                img_lq = cv2.imread(dir + '/' + file)
                img_lq = cv2.resize(img_lq, (input_shape[0] * 4, input_shape[1] * 4))
                img.append(img_lq)
                img = np.array(img)
                img = img.astype('float32') / 127.5 - 1.
                img = lq2hq.predict(img, batch_size=1)
                cv2.imwrite(output_dir + '/' + file, (0.5 * img[0] + 0.5) * 255)

    def compute(self, set1, set2):
        PSNR = 0
        SSIM = 0
        Pearson = 0
        for i in range(set1.shape[0]):
            a = []
            b = []
            for x in range(set1[i].shape[0]):
                for y in range(set1[i].shape[1]):
                    for z in range(set1[i].shape[2]):
                        a.append(set1[i, x, y, z])
                        b.append(set2[i, x, y, z])
            Pearson += pearsonr(a, b)[0]
            PSNR += self.PSNR(0.5 * set1[i] + 0.5, 0.5 * set2[i] + 0.5)
            SSIM += self.SSIM(0.5 * set1[i] + 0.5, 0.5 * set2[i] + 0.5)
        print('PSNR : ' + str(PSNR / set1.shape[0]))
        print('SSIM : ' + str(SSIM / set1.shape[0]))
        print('Pearson : ' + str(Pearson / set1.shape[0]))

    def PSNR(self, img1, img2):
        psnr = 0
        for i in range(img1.shape[2]) :
            psnr += compare_psnr(img1[:,:,i], img2[:,:,i], 1)
        return psnr / img1.shape[2]

    def SSIM(self, img1, img2):
        return compare_ssim(img1, img2, data_range=1, multichannel=True)

if __name__ == '__main__':
    # acgan + mnist dataset
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dcgan = StarGAN()
    save_num = 500
    epoch = 25000
    set = 'C0depth'
    z_depth = 'Z005'
    model = 'deblursrgan4' + '_' + set + '_' + z_depth
    batch_size = 4
    dcgan.test(model=model, epochs=epoch, batch_size=batch_size, sample_interval=int(epoch / save_num), set=set,
                z_depth=z_depth)
