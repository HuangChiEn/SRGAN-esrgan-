#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import  ## execute the following code in python3.x enviroment

"""
Created on Feb  28  2020
@author: Josef-Huang

@@ The following code is unstable ' V 1.0 (stable version) '
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
    And all your logical control can implement in here.
            
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

            Josef-Huang...2020/02/28(Freitag)
"""

""" tensorflow backend--GPU env seeting : """
import tensorflow as tf
from keras.utils import multi_gpu_model
import keras.backend.tensorflow_backend as KTF  
import os

import argparse  ## User-define parameter setting
import sys   
sys.path.insert(1, '../SRGAN') ## Grabe the path of shared-module 

''' self-define package '''
from SRGAN import SRGAN



'''Main function : (implement your parameter setting and control logic)'''
def main():
    '''(1) Parameter setting stage : the parameters used in module are defined by following code'''
    parser = argparse.ArgumentParser(description="None")

    '''Enviroment Setting : '''
    parser.add_argument('--cuda', type=str, default="0, 1, 2, 3", help='a list of gpus')
        
    '''Image specification : '''
    parser.add_argument('--dataSet', type=str, default='../dataSet/', help="The path of data set.")
    parser.add_argument('--lrImgHei', type=int, default=120, help="The height of lower resoultion input image (pixel value).")  
    parser.add_argument('--lrImgWid', type=int, default=160, help="The width of lower resoultion input image (pixel value).")
    parser.add_argument('--imgScal', type=int, default=4, help="The size of image scalar (from low resolution to super resolution).")
    
    '''Model specification : '''
    parser.add_argument('--numOfRRDB', type=int, default=7, help="The number of residual block.")
    parser.add_argument('--GFilt', type=int, default=64, help="The number of filter in Generator.")
    parser.add_argument('--DFilt', type=int, default=64, help="The number of filter in Discriminator.")
    parser.add_argument('--DPatSiz', type=int, default=4, help="The patch size of discriminator (see comment patch GAN).")
    
    '''Training setting : '''
    parser.add_argument('--traSetNam', type=str, default='traSet', help="The name of training data set.")
    parser.add_argument('--tstSetPath', type=str, default='tstSet', help="The name of testing data set.")
    parser.add_argument('--epochs', type=int, default=10001, help="The number of epoch during training.")
    parser.add_argument('--batSiz', type=int, default=4, help="The number of batch size during training.")
    parser.add_argument('--samInv', type=int, default=100, help="The interval of saveing the image during training.")
    
    '''Prediction setting : '''
    parser.add_argument('--lrNum', type=int, default=4, help="The number of lower resolution with generate sr images.")
    parser.add_argument('--lrPath', type=str, default=None, help="The loading image path of lower resolution.")
    
    args = parser.parse_args() ## parser the arguments, get parameter via arg.parameter

    '''(2) GPU enviroment setting stage : '''
    if args.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
        os.system('echo $CUDA_VISIBLE_DEVICES')
        config = tf.ConfigProto()  
        config.gpu_options.allow_growth=True   ## Avoid to run out all memory, allocate it depend on the requiring
        sess = tf.Session(config=config)
        KTF.set_session(sess)
        n_gpus = len(args.cuda.split(','))
        
    else:
        print('Training without gpu. It is recommended using at least one gpu.')
        n_gpus = 0

    '''(3) Initialize parameter list stage : ''' ## divide the parameters list : init_params, train_params
    params = {
        ## Env setting :
        'n_gpus': n_gpus,
        
        ## Image setting :
        'lr_hei': args.lrImgHei,
        'lr_wid': args.lrImgWid,
        'lr_channel': 3,
        'img_scalr': args.imgScal,
            
        ## Model specification :
        'n_RRDB': args.numOfRRDB,
        'n_G_filt': args.GFilt,
        'n_D_filt': args.DFilt,
        'D_patch_size': args.DPatSiz,
        'pre_model_dic': '../pretrain/',
        
        ## Training phase setting : 
        'data_set_name': 'IOM',
        'prepro_dir': '../preprocess_img/',
        #'tra_hr_dir': os.path.join(args.dataSet, args.traSetNam, 'HR'),
        #'tra_lr_dir': os.path.join(args.dataSet, args.traSetNam, 'LR'),
        'tst_lr_dir': args.tstSetPath,
        'ext': '.png',
        'epochs': args.epochs,
        'batch_size': args.batSiz,
        'sample_intval': args.samInv,
        'per_loss_w': 1e-3,
        
        ## Prediction setting : 
        'n_lr_img':args.lrNum,
        'lr_path':args.lrPath
    }   
    
    '''(4) Control model stage : ''' 
    deepMod = SRGAN(**params)
    deepMod.train(epochs=args.epochs, batch_size=args.batSiz, sample_interval=args.samInv)
    #deepMod.generate_images(args.lr_path, s3args.lrNum)

if __name__ == '__main__':
    main()