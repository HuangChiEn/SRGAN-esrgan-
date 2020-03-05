#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import  ## execute the following code in python3.x enviroment

"""
Created on Tue Feb  18 15:32:00 2020
@author: Josef-Huang

@@ The following code is unstable ' V 0.4 '
Signature :
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@                                                                  @@
@@    JJJJJJJJJ    OOOO     SSSSSSS   EEEEEE   PPPPPPP   HH    HH   @@
@@       JJ       O    O    SSS       E        PP   PP   HH    HH   @@
@@       JJ      O      O    SSS      EEEEEE   PPPPPPP   HHHHHHHH   @@
@@       JJ       O    O         SS   E        PP        HH    HH   @@
@@     JJJ         OOOO     SSSSSSS   EEEEEE   PP        HH    HH   @@
@@                                                                  @@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    

Description :
    The following code is the implementation of super resolution GAN by keras mainly with tensorflow backend
    engining.
    The kernel module be divided into  generator, discriminator, auxiliary_featuretor. The different module 
    can load different type of basic block :  
    (1) generator -> RRDB(Residual in Residual Dense block)
        -> The subpixel conv are eliminated(currently), due to their effect are not good~(see problem sample)
        -> However, respect with paper proposed model, I trace back to ECPNetwork, that use tanh as activation.
        (main stream)@->
            @-> I use keras Upsampling2D function to upsampling(nearest), but replace the generator structure.
                the effect are as good as original structure, but better brightness.
    (2) discriminator -> RaGAN(Relativistic GAN), Conv block.
    
        (main stream)@->
            @-> I'm trying to build the RaGAN for recover more detail texture .
            
    (3) auxiliary_featuretor -> vgg19 before, vgg19 after.
        (main stream)@->
            @-> I'm trying to extract the before activated feature.
    
Acknowledegment :
    The following code are referenced from a lot of released source code and unreleased code.
    I want to greatly thank for the author on github who released the srgan.py source code, 
    and my senior k0 who offer some scheme to confirm the re-use ability, log-mechnism, 
    exception handler, and parallel GPU env to make the process of training be quickly!!
    
    At the result, of course my prof. Liu, who offer the lab to support our research.
    Every research can not be done completely without the env support!!
    
Notice : 
    As the file name, the edsr gan module will be implement in this code.
    The RRDB( Residual in Residual Dense Block ) has already be add in generator.

            Josef-Huang...2020/02/22(Donnerstag)
"""

""" tensorflow backend--GPU env seeting : """
import tensorflow as tf
from keras.utils import multi_gpu_model
import keras.backend.tensorflow_backend as KTF  
import os

'''keras package : the high-level module for deep learning '''
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.applications import VGG19  ## keras pretraining model : for extract activated feature
''' setting callback mechnism'''
'''from keras.callbacks import EarlyStopping, TensorBoard
ensure_dir( './TensorBoard/' + file_name )
path = './TensorBoard/' + file_name + 'logs'  
tensorboard = TensorBoard(log_dir=path, 
                          histogram_freq=0,
                          batch_size=batch_size, 
                          write_graph=True, 
                          write_grads=True, 
                          write_images=True )

earlyStop = EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=500, 
                          verbose=1 , 
                          mode='auto')'''

'''other package'''
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import sys       
sys.path.insert(1, '../tmpShar') ## Grabe the path of shared-module 

''' self-define package '''
## Extandable for data preprocessing
from dataloader import DataLoader
from modelManager import ModMang
from submodule import  load_module
from metricEvaluator import evaluator

class SRGAN():
    def __init__(self, **params):
        ## Input structure :
        self.channels = params['lr_channel']
        self.lr_height = params['lr_hei']       # Low resolution height
        self.lr_width = params['lr_wid']        # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.img_scalar = params['img_scalr']
        self.hr_height = self.lr_height*self.img_scalar   # High resolution height
        self.hr_width = self.lr_width*self.img_scalar     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.data_set_name = params['data_set_name']
        
        ## Configure data loader :
        self.data_loader = DataLoader(data_set_name=self.data_set_name,
                                      #prepro_dir=params['prepro_dir'],
                                      hr_img_size=(self.hr_height, self.hr_width), 
                                      scalr=self.img_scalar)
        ## Configure model manager :
        self.model_man =  ModMang(save_path=params['pre_model_dic'])
        
        ### Model setting ###
        optimizer = Adam(0.0002, 0.5) ## with best setting, do not change will be better
        ## Auxiliry_Model structure: 
            #   We use a pre-trained VGG19 model to extract image features from the high resolution
            #   and the generated high resolution images and minimize the mse between them
        aux_params = {
                ##self-define :
                'hr_shape' : self.hr_shape
                }
        self.auxMod = load_module('feature_dis', aux_params, False)
        self.auxMod.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.auxMod.trainable = False
        
        ## Discriminator structure :  
        ''' Calculate output shape of D (PatchGAN) 
            For each patch have a lot of true/false value
            So the discriminator output value according 
            to each pixel state ''' 
        patch_hei = int(self.hr_height / (2**4))
        patch_wid = int(self.hr_width / (2**4))
        self.disc_patch = (patch_hei, patch_wid, 1)
            #   Build and compile the discriminator
        
        ## Discriminator parameters setting :
        D_params = {
                ##self-define :
                'num_of_filts' : params['n_D_filt'],
                ## default :
                'hr_shape' : self.hr_shape
                }
        
        self.discriminator = load_module('discriminator', D_params, False)
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        ## Generator structure : 
        G_params = {
                ## self-define :
                'num_of_filts' : params['n_G_filt'],
                'num_of_RRDB' : params['n_RRDB'],
                'lr_shape' : self.lr_shape,
                ## default :
                'num_of_DB' : 3,
                'upScalar' : 2,
                'resScal' : 0.2
                }
        #upScalar=math.floor(math.sqrt(self.imgScal))
        #print()
        self.generator = load_module('generator', G_params, False)
        
        ## Model relation stucture : 
            #   High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

            #   Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

            #   Extract image features of the generated img
        fake_features = self.auxMod(fake_hr)

            #   For the combined model we will only train the generator
        self.discriminator.trainable = False

            #   Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)
        
        ## Build the combined model to combind 
        ##  the generator, discriminator and auxiliary_model
        '''(1) fea-dis eat sr-img, not hr-img, and the loss modify by [][][]'''
        """ For parallel model : """
        mod = Model([img_lr, img_hr], [validity, fake_features])
        self.combined = multi_gpu_model(mod, gpus=4)  ## for parallel seeting
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)
        """ load G and D model weights : """
        self.generator, self.discriminator = self.model_man.simple_load(generator=self.generator, discriminator=self.discriminator)
        
        
    ''' The following code describe the action about model '''
    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr, _ = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)
            ''' discriminator path size with batch size (look patch GAN comment)'''
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)
            
            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr, _ = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.auxMod.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the tranning progress
            print ("%d time: %s" % (epoch, elapsed_time))
            print("D loss : {} ; G loss : {} ".format(d_loss, g_loss))
            
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        
        self.model_man.save_all(generator=self.generator, discriminator=self.discriminator)
        
    ## modification        
    def generated_image(self, batSiz=2, lr_path=None):
        os.mkdirs('../images/%s' % self.data_set_name, exist_ok=True)
        # = self.data_loader.load_data(batch_size=batSiz, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)
        pass
        
    def sample_images(self, epoch):
        os.makedirs('../images/%s' % self.data_set_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr, imgs = self.data_loader.load_data(batch_size=2)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5
        imgs = 0.5 * imgs + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("../images/%s/%d.png" % (self.data_set_name, epoch))
        plt.close()
        
        for idx in range(r):
            ''' Setting the figure(s) '''
            fig0 = plt.figure(num=0)
            fig1 = plt.figure(num=1)
            fig2 = plt.figure(num=2)
            ''' Plot on each figure '''
            ## plot lower resolution image
            plt.figure(fig0.number)
            plt.imshow(imgs_lr[idx])
            ## plot super resolution image
            plt.figure(fig1.number)
            plt.imshow(fake_hr[idx])
            ## plot high resolution image (original image)
            plt.figure(fig2.number)
            plt.imshow(imgs[idx])
            ''' Saving the image in each figure '''
            fig0.savefig('../images/%s/%d_lowres%d.png' % (self.data_set_name, epoch, idx))
            fig1.savefig('../images/%s/%d_super%d.png' % (self.data_set_name, epoch, idx))
            fig2.savefig('../images/%s/%d_original%d.png' % (self.data_set_name, epoch, idx))
            '''close the figure..'''
            plt.close(fig0)
            plt.close(fig1)
            plt.close(fig2)
        
## For unit test..
if __name__ == '__main__':
    main()    