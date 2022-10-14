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


img_save_loc = r'G:\FgSegNet_50p_S\Results_FgSegNet_50p-50p_S\results_txt\Results_FgSegNet_50p-50p_S_v1.txt'



dataset_1 = [ 'badWeather']
"""
dataset = {
            'badWeather':['blizzard'],
            'baseline':['pedestrians'],
            'cameraJitter':['sidewalk'],#boulevard, sidewalk, traffic
            'dynamicBackground':['boats'], #'canoe',
            'intermittentObjectMotion':['parking'],
            'lowFramerate':['turnpike_0_5fps'], #, 'turnpike_0_5fps'
            'nightVideos':['tramStation'],
            #'PTZ':['zoomInZoomOut'],
            'shadow':['busStation'],
            'thermal':['corridor'],
            'turbulence':['turbulence1']
}
"""
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
            'turbulence':['turbulence0', 'turbulence2', 'turbulence3','turbulence1']
}




#epochs= [50,55,60,65,70,75,80,85,90,95,100]
epochs= [40,45,50,55,60,65,70,75,80,85]
res_list = []

 
for category in dataset_1:   

            scene_list = dataset[category]

            for scene in scene_list:
                
                print ('Score calculation for ->>> ' + category + ' / ' + scene)
                
                for e in epochs: 

                    tp = 0
                    tn = 0
                    fp = 0
                    fn = 0


                    lbl_add = r'C:\Users\santo\OneDrive\Desktop\soumendu_work\dataset2014\\'+category+'\\'+scene+'\groundtruth\\'
                    pst_add = r'G:\FgSegNet_50p_S\Results_FgSegNet_50p-50p_S\Results_FgSegNet_S_v1\\'+category+'\\'+scene+'\predicted_loop\e'+str(e).zfill(3)+'\\'

                    L_list = glob.glob(os.path.join(lbl_add,'*g'))
                    L_list = sorted(L_list)
                    stop = int(0.5 * len(L_list[49:]))
                    stop = 49 + stop
                    L_list = L_list[stop:]
                    P_list = glob.glob(os.path.join(pst_add,'*g'))
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
		
		
		
                    print('\n'+'epoch= '+str(e)+' category= '+category+'scene= '+scene+' tp= '+str(tp)+' fp= '+str(fp)+' tn= '+str(tn)+' fn= '+str(fn)+'\n')

		
                    res_list.append('epoch ->>> '+str(e)+'category ->>> '+category+' , scene ->>> '+scene+' , True Positive ->>> '+str(tp)+' , False Positive ->>> '+str(fp)+' , True Negative ->>> '+str(tn)+' , False Negative ->>> '+str(fn))
		
				
		
with open(img_save_loc, 'wb') as fp:      # img_save_loc[i]
	pickle.dump(res_list, fp)



    
