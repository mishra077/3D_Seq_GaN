# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 01:21:25 2019

@author: MEMS_Santosh_WS
"""

import numpy as np
import tensorflow as tf
import os
import os.path
import glob
import numpy as np
import cv2
from PIL import *
from keras.preprocessing import image as KImage



resize_shape = (256, 256)
folder = r'G:\I2R'

"""
dataset_1 = [ 'badWeather', 'baseline', 'cameraJitter', 'dynamicBackground', 'intermittentObjectMotion', 'lowFramerate', 'nightVideos', 'shadow', 'thermal', 'turbulence']

dataset = {
            'badWeather':['blizzard'],
            'baseline':['pedestrians'],
            'cameraJitter':['sidewalk'],#boulevard, sidewalk, traffic
            'dynamicBackground':['boats'], #'canoe',
            'intermittentObjectMotion':['parking'],
            'lowFramerate':['turnpike_0_5fps'], #, 'turnpike_0_5fps'
            'nightVideos':['tramStation'],
            'PTZ':['zoomInZoomOut'],
            'shadow':['busStation'],
            'thermal':['corridor'],
            'turbulence':['turbulence1']
}

dataset1 = ['Dust', 'Fog', 'Low Light', 'Rain'] 

dataset2 = {
            'Dust': ['Dynamic Background', 'Flat Cluttered Background', 'Motion Camera'],
            'Fog': ['Dynamic Background', 'Flat Cluttered Background', 'Motion Camera'],
            'Low Light': ['Dynamic Background', 'Flat Cluttered Background', 'Motion Camera'],
            'Rain': ['Dynamic Background', 'Flat Cluttered Background']
}

dataset3 = {
            'Dynamic Background': ['D12_DB_14082018_GT', 'D13_DB_14082018_GT'],    #'D12_DB_14082018', 'D13_DB_14082018'
            'Flat Cluttered Background': ['D2_FCB_06072018_GT', 'D6_FCB_14082018_GT'],    #'D2_FCB_06072018', 'D6_FCB_14082018'
            'Motion Camera': ['D16_MC_14082018_GT']     #'D16_MC_14082018'
}

dataset4 = {
            'Dynamic Background': ['F8_DB_20012019_GT', 'F9_DB_20012019_GT'],  #'F8_DB_20012019', 'F9_DB_20012019'
            'Flat Cluttered Background': ['F1_FCB_26122018_GT', 'F5_FCB_20012019_GT'],    #'F1_FCB_26122018', 'F5_FCB_20012019'
            'Motion Camera': ['F12_MC_20012019_GT']    #'F12_MC_20012019'       
}

dataset5 = {
            'Dynamic Background': ['LL15_DB_03082018_GT', 'LL19_DB_19062019_GT'],  #'LL15_DB_03082018_GT', 'LL19_DB_19062019_GT''LL15_DB_03082018', 'LL19_DB_19062019'
            'Flat Cluttered Background': ['LL8_FCB_06072018_GT', 'LL11_FCB_19062019_GT'], #'LL8_FCB_06072018_GT', 'LL11_FCB_19062019_GT''LL8_FCB_06072018', 'LL11_FCB_19062019'
            'Motion Camera': ['LL22_MC_03082018_GT']   #'LL22_MC_03082018_GT''LL22_MC_03082018'
}

dataset6 = {
            'Dynamic Background': ['R4_DB_04092018_GT', 'R5_DB_13072019_GT'],  #'R4_DB_04092018_GT', 'R5_DB_13072019_GT''R4_DB_04092018', 'R5_DB_13072019'
            'Flat Cluttered Background': ['R1_FCB_04092018_GT', 'R3_FCB_13072019_GT']    #'R1_FCB_04092018_GT', 'R3_FCB_13072019_GT''R1_FCB_04092018', 'R3_FCB_13072019'
}
"""

dataset1 = {'Bootstrap','Campus','Curtain','Escalator','Fountain','Hall','Lobby','ShoppingMall','WaterSurface'}



for category in dataset1:   

       

    #if (category =='Dust'):
        #part_list = dataset3[scene]
    #elif (category =='Fog'):
        #part_list = dataset4[scene]
    #elif (category =='Low Light'):
        #part_list = dataset5[scene]
    #elif (category =='Rain'):
        #part_list = dataset6[scene]


    #for part in part_list:

	
        #print('RESIZING -->> '+category+' / '+scene+ '/'+part)
    print('RESIZING -->> '+category)
    part2Name = folder + '\\' + str(category) + '_GT_256'                                                                                       

    if not os.path.exists(part2Name):
        os.makedirs(part2Name)

    image_path_list = glob.glob(os.path.join(folder+'\\'+category+ '_GT\\'+'*p'))
    image_path_list = sorted(image_path_list)


    for image_path in image_path_list:
        image = cv2.imread(image_path,0)
        image = cv2.resize(image, resize_shape)
        image = image.reshape(256,256,1)
        image =  KImage.array_to_img(image)
            #cv2.imshow('image',image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        image.save(folder + '\\' + str(category) + '_GT_256\\'+((os.path.basename(image_path)).split('.')[-2])+'.bmp')


print('END')