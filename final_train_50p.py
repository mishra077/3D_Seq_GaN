
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 13:19:32 2018

@author: vansh
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

#from extra import PolyDecay
#from extra import OurGenerator
import glob
from keras.preprocessing import image as kImage
#from net_model import Module
from final_MSFgNet_new import MSFgNet
#from New_Net_channel_last_AB import MSFgNet




#change



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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 


# Optimizer
my_epoch = 200
my_batch_size = 1
training_samples =  1200 #2586 #1451 # 88831, 4217, 1182 , 951 , 1451 , 302 , 433, 266, 4252 , 1951 , 48285 , 5953, 5753
epoch_resume = 0
my_size_shape = (256, 256)
my_depth = 50



dataset_1 = ['lowFramerate']
#dataset_1 = [ 'badWeather', 'baseline', 'cameraJitter', 'dynamicBackground', 'intermittentObjectMotion', 'lowFramerate', 'nightVideos', 'PTZ', 'shadow', 'thermal', 'turbulence']

dataset = {
            'badWeather':['blizzard', 'skating', 'snowFall', 'wetSnow'],
            'baseline':['highway', 'office', 'pedestrians', 'PETS2006'],
            'cameraJitter':['badminton', 'boulevard', 'sidewalk', 'traffic'],
            'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
            'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
            'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
            'nightVideos':['bridgeEntry','busyBoulvard','fluidHighway','streetCornerAtNight','tramStation', 'winterStreet'],
            'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
            'shadow':['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
            'thermal':['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
            'turbulence':['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3',]
}








def OurGenerator(batch_size = 1, depth = 50, size_shape = (256, 256)): # , resize_shape, crop_shape, horizontal_flip, vertical_flip , brightness, rotation, zoom , resize_shape = ( 256, 256)

    #print(mode)
    void_label = 0
    folder=r'C:\Users\santo\OneDrive\Desktop\soumendu_work\dataset2014'
#        image_path_list = sorted(glob.glob(os.path.join(folder, mode, '_input/*.png')))      # test code before
#        label_path_list = sorted(glob.glob(os.path.join(folder, mode, '_output/*.png')))
#    img_path_add   = folder+mode+'_input/*.png'
#        label_path_add = folder+mode+'_output/*.png'


    """
    horizontal_flip = horizontal_flip
    vertical_flip = vertical_flip
    brightness = brightness
    rotation = rotation
    zoom = zoom
    """

    # Preallocate memory
    """
    if mode == 'training' and resize_shape:
        X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 1), dtype='float32')
        Y = np.zeros((batch_size, resize_shape[1], resize_shape[0], 1), dtype='float32')

    elif resize_shape:
        X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 1), dtype='float32')
        Y = np.zeros((batch_size, crop_shape[1], crop_shape[0], n_classes), dtype='float32')
    else:
        raise Exception('No image dimensions specified!')
    """
    X = np.zeros((batch_size,depth, size_shape[1], size_shape[0], 1), dtype='float32')
    main_X = np.zeros((batch_size, 1, size_shape[1], size_shape[0], 1), dtype='float32')    
    Y = np.zeros((batch_size, 1, size_shape[1], size_shape[0], 1), dtype='float32')    
    

    while True:
        
        a=0
        random.shuffle(dataset_1)
        for category in dataset_1:   

            scene_list = list(dataset[category])
            random.shuffle(scene_list)

            for scene in scene_list:

                image_path_list = glob.glob(os.path.join(folder+'\\'+category+'\\'+scene+'\input\\'+'*jpg'))
                image_path_list = sorted(image_path_list)
                label_path_list = glob.glob(os.path.join(folder+'\\'+category+'\\'+scene+'\groundtruth\\'+'*png'))
                label_path_list = sorted(label_path_list)

                for i,(image_path, label_path) in enumerate(zip(image_path_list[49:],label_path_list[49:])):

                    label_ac = cv2.imread(label_path, 0)
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

                    main_img = cv2.imread(image_path, 0)
                    main_img = KImage.img_to_array(main_img)
                    main_img = main_img.reshape(-1)
                    if (len(idx)>0):
                        main_img[idx] = void_label
                    main_img = main_img.reshape(256, 256, 1)
                    main_X[a][0] = main_img
                    for x in range(50):
                        label = cv2.imread(label_path_list[49+i-x], 0)
                        label = KImage.img_to_array(label)
                        shape = label.shape
                        label /= 255.0
                        label = label.reshape(-1)
                        idx_l = np.where(np.logical_and(label>0.25, label<0.8))[0] # find non-ROI

                        image = cv2.imread(image_path_list[49+i-x], 0)
                        image = KImage.img_to_array(image)
                        image = image.reshape(-1)
                        if (len(idx_l)>0):
                            image[idx_l] = void_label
                        image = image.reshape(256, 256, 1)
                        X[a][x] = image

                    a = a+1
                    if(a==batch_size):
                        a=0
                        yield [X, main_X], Y


# print('i=='+str(i))


#### Train ####
def scheduler(epoch):
    if (epoch >= 21 and epoch <= 40):
        return 0.0004
    elif (epoch >= 41 and epoch <= 60):
        return 0.0002
    elif (epoch >= 61 and epoch <= 80):
        return 0.0001
    elif (epoch >= 81 and epoch <= 100):
        return 0.0001
    elif (epoch >= 101 and epoch <= 120):
        return 0.0001
    elif (epoch >= 121 and epoch <= 200):
        return 0.0001   
    else:
        return 0.0006
'''
def scheduler(epoch):
    if (epoch >= 21 and epoch <= 40):
        return 0.0001
    elif (epoch >= 41 and epoch <= 60):
        return 0.0001
    elif (epoch >= 61 and epoch <= 80):
        return 0.0001
    elif (epoch >= 81 and epoch <= 100):
        return 0.0001
    elif (epoch >= 101 and epoch <= 120):
        return 0.0001
    elif (epoch >= 121 and epoch <= 200):
        return 0.0001   
    else:
        return 0.0001
'''  

change_lr = LearningRateScheduler(scheduler)
# Workaround to forbid tensorflow from crashing the gpu and limits gpu memory usage

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))


# creating a new model or loading a partailly pretained one

path = r'C:\Users\santo\OneDrive\Desktop\soumendu_work\data\model_save\3dcd_murari_50p\lowFramerate\init_camj\*h5' 
files=glob.glob(path)


if (files):
    arr = []
    for fil in files:
        arr.append(int(fil.split('.')[-2]))
    index_max = max(range(len(arr)), key=arr.__getitem__)
    model = load_model(files[index_max])
    epoch_resume = int(files[index_max].split('.')[-2])
else:
    """
    for net_module.py
    img_shape = (512, 512, 1) 
    model = Module(img_shape)
    model = model.initModel()
    """
    #for new_res_net.py
    input_shape = (my_depth, my_size_shape[1], my_size_shape[0])
    input_shape2 = (1, my_size_shape[1], my_size_shape[0])
    model = MSFgNet(input_shape, input_shape2)
    model = model.initModel_3D()







# Callbacks

checkpoint = ModelCheckpoint(r'C:\Users\santo\OneDrive\Desktop\soumendu_work\data\model_save\3dcd_murari_50p\lowFramerate\init_camj\dsfmdl.{epoch:03d}.h5', monitor='loss', mode='min')
tensorboard = TensorBoard(log_dir = r'C:\Users\santo\OneDrive\Desktop\soumendu_work\data\log_dir', batch_size=my_batch_size)
#lr_decay = LearningRateScheduler(PolyDecay(my_lr, my_decay, my_epoch).scheduler)
#lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.5,
#                            patience=5, min_lr=0.0000005)                                                                                                                                                                                                                                                                                                                                                

# Generators
train_gen = OurGenerator( my_batch_size, my_depth, my_size_shape) # , my_resize_shape
#val_gen = OurGenerator('val', my_batch_size)# mode = 'training', 'val', 'test'




# Model
"""
if opt.checkpoint:
    net = load_model(opt.checkpoint)
else:
    net = model.build_bn(opt.image_width, opt.image_height, 66, train=True)
if opt.n_gpu > 1:
    net = multi_gpu_model(net, opt.n_gpu)
"""

# Training
 # steps_per_epoch =,
model.fit_generator(generator = train_gen, steps_per_epoch = (training_samples//my_batch_size), epochs = my_epoch, callbacks=[checkpoint, tensorboard, change_lr],
shuffle=False,  max_queue_size=5, initial_epoch= epoch_resume) #initial_epoch= my_epoch, validaton_steps = len(val_gen),, , , my_lr, validation_data = val_gen, validation_steps = (11216//my_batch_size), , initial_epoch= epoch_resume, , lr_decay


"""

image = cv2.imread('/home/cvprlab/background_subtraction/s805.png', 0)
image = kImage.img_to_array(image)
image = image.reshape(-1,512,512,1)

label = model.predict( image, batch_size=None, verbose=1, steps=None)
image = image.reshape(512,512,1)
image = kImage.array_to_img(image)
image.show()
label = np.round(label)
label = label.reshape(512,512,1)

label *=255
label = kImage.array_to_img(label)
label.show()
label.save('/home/cvprlab/background_subtraction/baseline_highway_output3.png')
"""

#saving model
#model.save('/home/cvprlab/backgroud-subtraction2/data/dataset_2014/model_save/new1/final_dsfmdl.h5')
#del model

#model = load_model('/home/cvprlab/background_subtraction/dataset_2014/model_save/trained_mdl.h5')

