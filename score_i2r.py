# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:59:24 2020

@author: MEMS_Santosh_WS
"""

import csv
import pickle
import os
import glob
from PIL import Image 
import numpy as np
import cv2
from keras.preprocessing import image as kImage


img_save_loc = r'G:\I2R\results_txt\Results_i2r_2dcd.txt'



dataset_1 = {'Bootstrap','Campus','Curtain','Escalator','Fountain','Hall','Lobby','ShoppingMall','WaterSurface'}


"""

dataset_1 = [ 'intermittentObjectMotion']
#, 'PTZ'
dataset = {
            'badWeather':['wetSnow', 'skating', 'snowFall', 'blizzard'],        #['wetSnow', 'skating', 'snowFall','blizzard']                                                     #, 'wetSnow'blizzard
            'baseline':['highway', 'pedestrians', 'PETS2006','office'],                                                            #, 'office'
            'cameraJitter':['badminton','sidewalk', 'traffic','boulevard'],                                                          #, 'boulevard'
            'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],                                     #, 'boats'
            'intermittentObjectMotion':[ 'abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],             #, ''streetLight
            'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps'],                                  #, 'turnpike_0_5fps'
            'nightVideos':['bridgeEntry','busyBoulvard','fluidHighway','streetCornerAtNight', 'winterStreet','tramStation'],            #,'tramStation'
            #'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
            'shadow':['backdoor', 'bungalows', 'copyMachine', 'cubicle', 'peopleInShade','busStation'],                                #, 'backdoor', 'bungalows', 'copyMachine', 'cubicle', 'peopleInShade','busStation'
            'thermal':['corridor', 'diningRoom', 'lakeSide','library', 'park'],                                                      #, ''corridor
            'turbulence':['turbulence0', 'turbulence2', 'turbulence3','turbulence1']                                                   #, 'turbulence1'
}
"""




epochs= [84]
res_list = []

 
for category in dataset_1:   

    print ('Score calculation for ->>> ' + category)
    
    for e in epochs: 
    
        tp = 0
        tn = 0
        fp = 0
        fn = 0
    
    
        lbl_add = r'G:\I2R\\'+category+'_GT_256\\'
        pst_add = r'G:\I2R\\'+category+'_test\predicted_loop_2dcd\e'+str(e).zfill(3)+'\\'
    
        L_list = glob.glob(os.path.join(lbl_add,'*p'))
        L_list = sorted(L_list)
        #L_list = L_list[49:]
        P_list = glob.glob(os.path.join(pst_add,'*p'))
        P_list = sorted(P_list)
        
    
        for k,(lbl,pst) in enumerate(zip(L_list, P_list)):
            #print('comp no. ->>> '+str(k+1)+' / '+str(len(L_list)))
            y_true = cv2.imread(lbl, 0)
            y_true = kImage.img_to_array(y_true)
    
            y_pred = cv2.imread(pst, 0)
            y_pred = kImage.img_to_array(y_pred)
    
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)
    
            for j in range(len(y_true)):
    	
                if(y_true[j]==255 and y_pred[j]==255):
                    tp +=1
                elif(y_true[j]<=50 and y_pred[j]==0):
                    tn +=1
                elif(y_true[j]<=50 and y_pred[j]==255):
                    fp +=1
                elif(y_true[j]==255 and y_pred[j]==0):
                    fn+=1
    		
    		
    		
        print('\n'+'epoch= '+str(e)+' category= '+category+' tp= '+str(tp)+' fp= '+str(fp)+' tn= '+str(tn)+' fn= '+str(fn)+'\n')
    
    		
        res_list.append('epoch ->>> '+str(e)+'category ->>> '+category+' , True Positive ->>> '+str(tp)+' , False Positive ->>> '+str(fp)+' , True Negative ->>> '+str(tn)+' , False Negative ->>> '+str(fn))
    		
    				
		
with open(img_save_loc, 'wb') as fp:      # img_save_loc[i]
	pickle.dump(res_list, fp)



    
