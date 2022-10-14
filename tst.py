# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:18:26 2020

@author: MEMS_Santosh_WS
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:05:25 2020

@author: MEMS_Santosh_WS
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 18:12:37 2020

@author: vansh dhar
"""

import numpy as np
import tensorflow as tf
import random 
import os
import os.path
import json
import pickle
import gc
import glob
import numpy as np
import cv2
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#from final_MSFgNet_new import MSFgNet

import glob
from keras.preprocessing import image as kImage

import os
import numpy as np
from os import listdir
from random import shuffle
#from get_dataset import read_npy_dataset, split_npy_dataset
from keras.callbacks import ModelCheckpoint, TensorBoard
from get_models2 import get_segment_model, get_Discriminator, get_GAN, get_Generator, save_model




from random import shuffle
import numpy as np
import cv2

from keras.preprocessing import image as KImage
from keras.callbacks import Callback
from keras.layers import concatenate
from keras.layers.core import Lambda
from keras.models import Model
from keras.utils.data_utils import Sequence
from keras import backend as K

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Optimizer
#f=1
train_gan_model = True
my_epoch = 4
epochs = 4
my_batch_size = 1
batch_size = 2
training_samples =  70426 # 88831, 4217, 1182 , 951 , 1451 , 302 , 433, 266, 4252 , 1951 , 48285 , 5953, 5753
epoch_resume = 0
my_size_shape = (256, 256)
size_shape = (256, 256)
my_depth = 50
depth = 50

def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = y_true.flatten()
    flat_y_pred = y_pred.flatten()
    return (2. * np.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (np.sum(flat_y_true) + np.sum(flat_y_pred) + smoothing_factor)




dataset_1 = [ 'badWeather', 'baseline', 'cameraJitter', 'dynamicBackground', 'intermittentObjectMotion', 'lowFramerate', 'nightVideos', 'shadow', 'thermal', 'turbulence']
#, 'PTZ'
dataset = {
            'badWeather':['blizzard', 'skating', 'snowFall'],                                                             #, 'wetSnow'
            'baseline':['highway', 'pedestrians', 'PETS2006'],                                                            #, 'office'
            'cameraJitter':['badminton', 'sidewalk', 'traffic'],                                                          #, 'boulevard'
            'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'overpass'],                                     #, 'fountain02'
            'intermittentObjectMotion':['abandonedBox', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],             #, 'parking'
            'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps'],                                  #, 'turnpike_0_5fps'
            'nightVideos':['bridgeEntry','busyBoulvard','fluidHighway','streetCornerAtNight', 'winterStreet'],            #,'tramStation'
            #'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
            'shadow':['backdoor', 'bungalows', 'copyMachine', 'cubicle', 'peopleInShade'],                                #, 'busStation'
            'thermal':['corridor', 'diningRoom', 'library', 'park'],                                                      #, 'lakeSide'
            'turbulence':['turbulence0', 'turbulence2', 'turbulence3',]                                                   #, 'turbulence1'
}






def OurGenerator(batch_size, depth, size_shape): # , resize_shape, crop_shape, horizontal_flip, vertical_flip , brightness, rotation, zoom , resize_shape = ( 256, 256)

    #print(mode)
  
    a=0
    #while True: 
    for category in dataset_1:   

        scene_list = list(dataset[category])

        for scene in scene_list:

            image_path_list = glob.glob(os.path.join(folder+'\\'+category+'\\'+scene+'\input\\'+'*jpg'))
            image_path_list = sorted(image_path_list)
            label_path_list = glob.glob(os.path.join(folder+'\\'+category+'\\'+scene+'\groundtruth\\'+'*png'))
            label_path_list = sorted(label_path_list)

            for i,(image_path, label_path) in enumerate(zip(image_path_list[49:],label_path_list[49:])):

                a = a+1
                if(a==batch_size):
                    a=0
                    yield 1




    
    

for epoch in range(epochs):
    
    data_generator = OurGenerator(batch_size, depth, size_shape)
    print('Epoch: {0}/{1}'.format(epoch+1, epochs))
    x=0
    for z in range(int(training_samples/batch_size)):
        
        x=x+1
        XY_data = next(data_generator)
        
    print(x)