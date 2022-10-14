# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:35:24 2020

@author: MEMS_Santosh_WS
"""

from __future__ import print_function, division
import scipy

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, AveragePooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling3D, Conv2D, Conv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from random import shuffle
import numpy as np
import cv2
import random
import glob

from keras.preprocessing import image as KImage
from keras.callbacks import Callback
from keras.layers import concatenate
from keras.layers.core import Lambda
from keras.models import Model
from keras.utils.data_utils import Sequence
from keras import backend as K
from keras.preprocessing import image as KImage
from keras.callbacks import Callback
from keras.layers import concatenate
from keras.layers.core import Lambda
from keras.models import Model
from keras.utils.data_utils import Sequence
from keras import backend as K

from keras.layers import Add, Subtract
from keras.layers import MaxPooling3D as mp3d
from keras.layers import AveragePooling3D as ap3d
from keras.utils import plot_model
from keras.optimizers import SGD

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
os.environ["CUDA_VISIBLE_DEVICES"]="3"; 

dataset_1 = [ 'dynamicBackground']
#, 'PTZ'
dataset = {
            'badWeather':['wetSnow', 'skating', 'snowFall','blizzard'],                                                             #, 'wetSnow'blizzard
            'baseline':['highway', 'pedestrians', 'PETS2006','office'],                                                            #, 'office'
            'cameraJitter':['badminton', 'sidewalk', 'traffic','boulevard'],                                                          #, 'boulevard'
            'dynamicBackground':['fountain02', 'canoe', 'fall', 'fountain01', 'overpass','boats'],                                     #, 'boats'
            'intermittentObjectMotion':['abandonedBox', 'sofa', 'parking', 'tramstop', 'winterDriveway','streetLight'],             #, ''streetLight
            'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps','turnpike_0_5fps'],                                  #, 'turnpike_0_5fps'
            'nightVideos':['bridgeEntry','busyBoulvard','fluidHighway','streetCornerAtNight', 'winterStreet','tramStation'],            #,'tramStation'
            #'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
            'shadow':['backdoor', 'bungalows', 'copyMachine', 'cubicle', 'peopleInShade','busStation'],                                #, 'busStation'
            'thermal':['lakeSide', 'diningRoom', 'library', 'park','corridor'],                                                      #, ''corridor
            'turbulence':['turbulence0', 'turbulence2', 'turbulence3','turbulence1']                                                   #, 'turbulence1'
}

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels1 = 50
        self.channels2 = 1
        self.img_shape = (self.channels1, self.img_rows, self.img_cols, 1)
        self.img_shape2 = (self.channels2, self.img_rows, self.img_cols, 1)
        self.size_shape = [self.img_rows, self.img_cols]
        self.train_gan_model = True
        self.epochs = 50
        self.batch_size = 1
        self.training_samples = 5190  #88831, 4217, 1182 , 951 , 1451 , 302 , 433, 266, 4252 , 1951 , 48285 , 5953, 5753
        self.epoch_resume = 1

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (1, patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 16
        self.df = 16

        # Loss weights
        self.lambda_cycle = 100.0                    # Cycle-consistency loss
        #self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.00001, 0.5)

        # Build and compile the discriminators
        self.d_1 = self.build_discriminator1()
        self.d_2 = self.build_discriminator2()
        self.d_1.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_2.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_1 = self.build_generator1()
        self.g_2 = self.build_generator2()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape) #(50, 256, 256, 1)- Input
        img_B = Input(shape=self.img_shape2) #(1, 256, 256, 1)- Input
        img_C = Input(shape=self.img_shape2) #(1, 256, 256, 1)- BG Groundtruth
        img_D = Input(shape=self.img_shape2)#Groundtruth (1, 256, 256, 1)
        """
        # By conditioning on B generate a fake version of A
        fake_A = self.generator([img_A, img_B])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_A, img_B])

        self.combined = Model(inputs=[img_C, img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
        """
        # Translate images to the other domain
        fake_Bg = self.g_1(img_A)
        fake_Cd = self.g_2([img_B, fake_Bg])
        # Translate images back to original domain
        #reconstr_A = self.g_BA(fake_B)
        #reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        #img_A_id = self.g_BA(img_A)
        #img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_1.trainable = False
        self.d_2.trainable = False

        # Discriminators determines validity of translated images
        valid_1 = self.d_1([fake_Bg, img_C])
        valid_2 = self.d_2([fake_Cd, img_D])

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B, img_C, img_D],
                              outputs=[valid_1, valid_2,
                                        fake_Bg, fake_Cd])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mse'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle],
                            optimizer=optimizer)
        
    def save_model(self, model, path='Data/Model/', model_name = 'model', weights_name = 'weights'):
        if not os.path.exists(path):
            os.makedirs(path)
        model_json = model.to_json()
        with open(path+model_name+'.json', 'w') as model_file:
            model_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(path+weights_name+'.h5')
        print('Model and weights saved to ' + path+model_name+'.json and ' + path+weights_name+'.h5')
        return    

    def OurGenerator(self): # , resize_shape, crop_shape, horizontal_flip, vertical_flip , brightness, rotation, zoom , resize_shape = ( 256, 256)

    #print(mode)
        void_label = 0
        folder='/home/users/student21/projectMNIT/data/dataset2014'

       
        X = np.zeros((self.batch_size,self.channels1, self.size_shape[1], self.size_shape[0], 1), dtype='float32')
        main_X = np.zeros((self.batch_size, 1, self.size_shape[1], self.size_shape[0], 1), dtype='float32')    
        Y = np.zeros((self.batch_size, 1, self.size_shape[1], self.size_shape[0], 1), dtype='float32')    
        Bg = np.zeros((self.batch_size, 1, self.size_shape[1], self.size_shape[0], 1), dtype='float32') 
        

        #while True: 
        a=0
        random.shuffle(dataset_1)
        for category in dataset_1:   

            scene_list = list(dataset[category])
            random.shuffle(scene_list)

            for scene in scene_list:

                image_path_list = glob.glob(os.path.join(folder+'/'+category+'/'+scene+'/input/'+'*jpg'))
                image_path_list = sorted(image_path_list)
                label_path_list = glob.glob(os.path.join(folder+'/'+category+'/'+scene+'/groundtruth/'+'*png'))
                label_path_list = sorted(label_path_list)
                mode_path = os.path.join(folder+'/'+category+'/'+scene+'/mode/'+'mod000100.png')
                stop = int(0.5 * len(image_path_list[49:]))
                stop = 49 + stop

                for i,(image_path, label_path) in enumerate(zip(image_path_list[49:stop],label_path_list[49:stop])):

                    
                    
                    label_ac = cv2.imread(label_path_list[49+i], 0)
                    label_ac = KImage.img_to_array(label_ac)
                    shape = label_ac.shape
                    label_ac /= 255.0
                    label_ac = label_ac.reshape(-1)
                    idx = np.where(np.logical_and(label_ac>0.25, label_ac<0.8))[0] # find non-ROI
                    if (len(idx)>0):
                        label_ac[idx] = void_label
                    label_ac = label_ac.reshape(256, 256, 1)
                    label_ac = np.floor(label_ac)
                    Y[a][0] = label_ac
                    
                    bckgd = cv2.imread(mode_path, 0)
                    bckgd = KImage.img_to_array(bckgd)
                    #bckgd = bckgd.reshape(-1)
                    #if (len(idx)>0):
                    #    bckgd[idx] = void_label
                    bckgd = bckgd.reshape(256, 256, 1)
                    Bg[a][0] = bckgd

                    main_img = cv2.imread(image_path_list[49+i], 0)
                    main_img = KImage.img_to_array(main_img)
                    main_img = main_img.reshape(-1)
                    if (len(idx)>0):
                        main_img[idx] = void_label
                    main_img = main_img.reshape(256, 256, 1)
                    main_X[a][0] = main_img
                    
                    for x in range(50):
                        """
                        label = cv2.imread(label_path_list[49+i-x], 0)
                        label = KImage.img_to_array(label)
                        shape = label.shape
                        label /= 255.0
                        label = label.reshape(-1)
                        idx_l = np.where(np.logical_and(label>0.25, label<0.8))[0] # find non-ROI
                        """

                        image = cv2.imread(image_path_list[49+i-x], 0)
                        image = KImage.img_to_array(image)
                        #image = image.reshape(-1)
                        #if (len(idx_l)>0):
                        #    image[idx_l] = void_label
                        image = image.reshape(256, 256, 1)
                        X[a][x] = image

                    a = a+1
                    if(a==self.batch_size):
                        a=0
                        yield [X, main_X, Bg], Y

    def load_network(self, e, index_max, files, files2, files3, files4):
        self.g_1.load_weights(files[index_max]+"/weights.h5")
        self.g_2.load_weights(files2[index_max]+"/weights.h5")
        self.d_1.load_weights(files3[index_max]+"/weights.h5")
        self.d_2.load_weights(files4[index_max]+"/weights.h5")
        self.epoch_resume = e
        
        

    def build_generator1(self):

        def conv3d(layer_input, filters, f_size=(1, 3, 3), bn=True):
            """Layers used during downsampling"""
            d = Conv3D(filters, kernel_size=f_size, strides=(1, 2, 2), padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv3d(layer_input, skip_input, filters, f_size=(1, 3, 3), dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling3D(size=(1, 2, 2))(layer_input)
            u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        input_1 = Input(shape=self.img_shape)
        #input_2 = Input(shape =self.img_shape2)
                
        model = Conv3D(32, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last')(input_1)
        model = ap3d(pool_size = (5, 3, 3), strides = (5, 1, 1), padding = 'same', data_format = 'channels_last')(model)        
        model = Activation('relu')(model)

        model = Conv3D(16, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last')(model)
        model = ap3d(pool_size = (2, 3, 3), strides = (2, 1, 1), padding = 'same', data_format = 'channels_last')(model)
        model = Activation('relu')(model)


        model = Conv3D(8, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last')(model)
        model = ap3d(pool_size = (5, 3, 3), strides = (5, 1, 1), padding = 'same', data_format = 'channels_last')(model)
        model = Activation('relu')(model)


        bg = Conv3D(1, (1, 3, 3), padding = 'same', data_format = 'channels_last')(model)
        bg = Activation('relu')(bg)

        return Model(input_1, bg)
    
    def build_generator2(self):

        def conv3d(layer_input, filters, f_size=(1, 3, 3), bn=True):
            """Layers used during downsampling"""
            d = Conv3D(filters, kernel_size=f_size, strides=(1, 2, 2), padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv3d(layer_input, skip_input, filters, f_size=(1, 3, 3), dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling3D(size=(1, 2, 2))(layer_input)
            u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        input_1 = Input(shape=self.img_shape2) #Input
        input_2 = Input(shape =self.img_shape2) #Fake_bg

        b1 = Conv3D(64, (1, 3, 3), strides = 1, padding = 'same')(input_1)
        b1 = LeakyReLU(alpha = 0.2)(b1)
        #c1 = BatchNormalization(momentum = 0.8)(c1)

        b2 = Conv3D(32, (1, 3, 3), strides = 1, padding = 'same')(b1)
        b2 = LeakyReLU(alpha = 0.2)(b2)
        b2 = BatchNormalization(momentum = 0.8)(b2)

        b3 = Conv3D(16, (1, 3, 3), strides = 1, padding = 'same')(b2)
        b3 = LeakyReLU(alpha = 0.2)(b3)
        b3 = BatchNormalization(momentum = 0.8)(b3)

        c1 = Conv3D(64, (1, 3, 3), strides = 1, padding = 'same')(input_2)
        c1 = LeakyReLU(alpha = 0.2)(c1)
        #c1 = BatchNormalization(momentum = 0.8)(c1)

        c2 = Conv3D(32, (1, 3, 3), strides = 1, padding = 'same')(c1)
        c2 = LeakyReLU(alpha = 0.2)(c2)
        c2 = BatchNormalization(momentum = 0.8)(c2)

        c3 = Conv3D(16, (1, 3, 3), strides = 1, padding = 'same')(c2)
        c3 = LeakyReLU(alpha = 0.2)(c3)
        c3 = BatchNormalization(momentum = 0.8)(c3)

        concat = Concatenate()([b3, c3])

        a1 = Conv3D(16, (1, 3, 3), strides = 1, padding = 'same')(concat)
        a1 = LeakyReLU(alpha = 0.2)(a1)
        a1 = BatchNormalization(momentum = 0.8)(a1)
        
        a3 = Conv3D(1, (1, 3, 3), strides = 1, padding = 'same')(a1)
        a3 = LeakyReLU(alpha = 0.2)(a3)
        a3 = BatchNormalization(momentum = 0.8)(a3)

        #U-Net

        #Encoder

        e0 = conv3d(a3, self.gf)
        e1 = conv3d(e0, self.gf*2)
        e2 = conv3d(e1, self.gf*4)
        e3 = conv3d(e2, self.gf*8)
        e4 = conv3d(e3, self.gf*16)
        e5 = conv3d(e4, self.gf*16) # 4 X 4 X 256

        #Decoder

        d0 = deconv3d(e5, e4, self.df*16)
        d1 = deconv3d(d0, e3, self.df*8)
        d2 = deconv3d(d1, e2, self.df*4)
        d3 = deconv3d(d2, e1, self.df*2)
        d4 = deconv3d(d3, e0, self.df) # 128 X 128X 32

        d5 = UpSampling3D(size=(1, 2, 2))(d4)

        output_img = Conv3D(self.channels2, kernel_size=(1, 3, 3), strides=1, padding='same', activation='tanh')(d5)
        return Model([input_1, input_2], output_img)


    def build_discriminator1(self):

        def d_layer(layer_input, filters, f_size=(1,3,3), bn=False):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=(1, 2, 2), padding='same')(layer_input)
            #d = LeakyReLU(alpha=0.2)(d)
            d = Activation('relu')(d)

            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape2) # Fake_BG
        img_B = Input(shape=self.img_shape2) # Groundtruth BG

        ################################################################

        b1 = Conv3D(64, (1, 3, 3), strides = 1, padding = 'same')(img_A)
        #b1 = LeakyReLU(alpha = 0.2)(b1)
        b1 = Activation('relu')(b1)

        b2 = Conv3D(32, (1, 3, 3), strides = 1, padding = 'same')(b1)
        #b2 = LeakyReLU(alpha = 0.2)(b2)
        b2 = Activation('relu')(b2)


        b3 = Conv3D(16, (1, 3, 3), strides = 1, padding = 'same')(b2)
        #b3 = LeakyReLU(alpha = 0.2)(b3)
        b3 = Activation('relu')(b3)


        ################################################################

        c1 = Conv3D(64, (1, 3, 3), strides = 1, padding = 'same')(img_B)
        c1 = Activation('relu')(c1)

        c2 = Conv3D(32, (1, 3, 3), strides = 1, padding = 'same')(c1)
        #c2 = LeakyReLU(alpha = 0.2)(c2)
        c2 = Activation('relu')(c1)

        c3 = Conv3D(16, (1, 3, 3), strides = 1, padding = 'same')(c2)
        #c3 = LeakyReLU(alpha = 0.2)(c3)
        c3 = Activation('relu')(c3)

        ################################################################

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([b3, c3])

        d1 = d_layer(combined_imgs, self.df*2, bn=False)
        d2 = d_layer(d1, self.df)
        d3 = d_layer(d2, self.df//2)
        #d4 = d_layer(d3, self.df//4)

        validity = Conv3D(1, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(d3)

        return Model([img_A, img_B], validity)
    
    def build_discriminator2(self):

        def d_layer(layer_input, filters, f_size=(1,3,3), bn=False):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=(1, 2, 2), padding='same')(layer_input)
            #d = LeakyReLU(alpha=0.2)(d)
            d = Activation('relu')(d)

            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape2) # Fake_BG
        img_B = Input(shape=self.img_shape2) # Groundtruth BG

        ################################################################

        b1 = Conv3D(64, (1, 3, 3), strides = 1, padding = 'same')(img_A)
        #b1 = LeakyReLU(alpha = 0.2)(b1)
        b1 = Activation('relu')(b1)

        b2 = Conv3D(32, (1, 3, 3), strides = 1, padding = 'same')(b1)
        #b2 = LeakyReLU(alpha = 0.2)(b2)
        b2 = Activation('relu')(b2)


        b3 = Conv3D(16, (1, 3, 3), strides = 1, padding = 'same')(b2)
        #b3 = LeakyReLU(alpha = 0.2)(b3)
        b3 = Activation('relu')(b3)


        ################################################################

        c1 = Conv3D(64, (1, 3, 3), strides = 1, padding = 'same')(img_B)
        c1 = Activation('relu')(c1)

        c2 = Conv3D(32, (1, 3, 3), strides = 1, padding = 'same')(c1)
        #c2 = LeakyReLU(alpha = 0.2)(c2)
        c2 = Activation('relu')(c2)

        c3 = Conv3D(16, (1, 3, 3), strides = 1, padding = 'same')(c2)
        #c3 = LeakyReLU(alpha = 0.2)(c3)
        c3 = Activation('relu')(c3)

        ################################################################

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([b3, c3])

        d1 = d_layer(combined_imgs, self.df*2, bn=False)
        d2 = d_layer(d1, self.df)
        d3 = d_layer(d2, self.df//2)
        #d4 = d_layer(d3, self.df//4)

        validity = Conv3D(1, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(d3)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(self.epoch_resume, epochs+1):
            
            data_generator = self.OurGenerator()
            for z in range(int(self.training_samples/self.batch_size)):
                
                XY= next(data_generator)
                imgs_A = XY[0][0] # (50, 256, 256, 1)
                imgs_B = XY[0][1]
                real_Bg = XY[0][2]# (1, 256, 256, 1)
                real_Cd = XY[1]   # groundtruth
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_Bg = self.g_1.predict(imgs_A)
                fake_Cd = self.g_2.predict([imgs_B, fake_Bg])

                # Train the discriminators (original images = real / generated = Fake)
                d1_loss_real = self.d_1.train_on_batch([real_Bg, real_Bg], valid)
                d1_loss_fake = self.d_1.train_on_batch([fake_Bg, real_Bg], fake)
                d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)
                
                d2_loss_real = self.d_2.train_on_batch([real_Cd,real_Cd], valid)
                d2_loss_fake = self.d_2.train_on_batch([fake_Cd,real_Cd], fake)
                d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)
                
                # Total disciminator loss
                d_loss = 0.5 * np.add(d1_loss, d2_loss)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B, real_Bg, real_Cd], [valid, valid, real_Bg, real_Cd])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [D1 loss: %f, D2 loss: %f] [G loss: %05f, G1: %05f, G2: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            d1_loss[0], d2_loss[0],
                                                                            g_loss[0],
                                                                            g_loss[1],
                                                                            g_loss[2],
                                                                            elapsed_time))
                
            path1='/home/users/student21/projectMNIT/data/seq_gan_50p_v4/g_1/e'+str(epoch).zfill(3)
            if not os.path.exists(path1):
                os.makedirs(path1)
            path2='/home/users/student21/projectMNIT/data/seq_gan_50p_v4/g_2/e'+str(epoch).zfill(3)
            if not os.path.exists(path2):
                os.makedirs(path2)
            path3='/home/users/student21/projectMNIT/data/seq_gan_50p_v4/d_1/e'+str(epoch).zfill(3)
            if not os.path.exists(path3):
                os.makedirs(path3)
            path4='/home/users/student21/projectMNIT/data/seq_gan_50p_v4/d_2/e'+str(epoch).zfill(3)
            if not os.path.exists(path4):
                os.makedirs(path4)
                
            self.save_model(self.g_1, path='/home/users/student21/projectMNIT/data/seq_gan_50p_v4/g_1/e'+str(epoch).zfill(3)+'/', model_name = 'model', weights_name = 'weights')
            self.save_model(self.g_2, path='/home/users/student21/projectMNIT/data/seq_gan_50p_v4/g_2/e'+str(epoch).zfill(3)+'/', model_name = 'model', weights_name = 'weights')
            self.save_model(self.d_1, path='/home/users/student21/projectMNIT/data/seq_gan_50p_v4/d_1/e'+str(epoch).zfill(3)+'/', model_name = 'model', weights_name = 'weights')
            self.save_model(self.d_2, path='/home/users/student21/projectMNIT/data/seq_gan_50p_v4/d_2/e'+str(epoch).zfill(3)+'/', model_name = 'model', weights_name = 'weights')
            print('Segmentation model checkpoints saved to "Data/Chackpoints/GAN-Models/Generator/"')
        
        return 



if __name__ == '__main__':
    path1 = '/home/users/student21/projectMNIT/data/seq_gan_50p_v4/g_1/*'
    path2 = '/home/users/student21/projectMNIT/data/seq_gan_50p_v4/g_2/*'
    path3 = '/home/users/student21/projectMNIT/data/seq_gan_50p_v4/d_1/*'
    path4 = '/home/users/student21/projectMNIT/data/seq_gan_50p_v4/d_2/*'
    files= glob.glob(path1)
    files2= glob.glob(path2)
    files3= glob.glob(path3)
    files4= glob.glob(path4)
    if (files):
        gan = Pix2Pix()
        arr = []
        for fil in files:
            arr.append(int((fil.split('/')[-1]).split('e')[-1]))
        index_max = max(range(len(arr)), key=arr.__getitem__)
        e = arr[index_max] + 1
        gan.load_network(e, index_max, files, files2, files3, files4)
        
    else:   
        gan = Pix2Pix()
        
    gan.train(epochs=50, batch_size=1)
        
        
        
        
        

