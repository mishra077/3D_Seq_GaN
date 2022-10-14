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


img_save_loc = r'G:\results_3d_seq_gan_100p_TUVDN.txt'


#epoch = [25, 30, 35, 37, 40, 45, 50]

dataset_1 = ['Dust', 'Fog', 'Low Light', 'Rain'] #

dataset2 = {
            'Dust':['Dynamic Background', 'Flat Cluttered Background'],     #             
            'Fog':['Dynamic Background', 'Flat Cluttered Background'], 
            'Low Light':['Dynamic Background', 'Flat Cluttered Background'],     #             
            'Rain':['Dynamic Background', 'Flat Cluttered Background'], #  
                
}

dataset3 = {
            'Dynamic Background':['D12_DB_14082018', 'D13_DB_14082018'],   #, 'I_BS_02'
            'Flat Cluttered Background':['D2_FCB_06072018', 'D6_FCB_14082018']
}
dataset4 = {
            'Dynamic Background':['F8_DB_20012019', 'F9_DB_20012019'],   #, 'I_BS_02'
            'Flat Cluttered Background':['F1_FCB_26122018', 'F5_FCB_20012019']
}
dataset5 = {
            'Dynamic Background':['LL15_DB_03082018', 'LL19_DB_19062019'],   #, 'I_BS_02'
            'Flat Cluttered Background':['LL8_FCB_06072018', 'LL11_FCB_19062019']
}
dataset6 = {               

            'Dynamic Background':['R4_DB_04092018', 'R5_DB_13072019'],   #, 'I_BS_02'
            'Flat Cluttered Background':['R1_FCB_04092018', 'R3_FCB_13072019']
}





from_epoch=20
to_epoch=50
res_list = []

 


for category in dataset_1:   

      scene_list = dataset2[category]

      for scene in scene_list:
          
          if (category=='Dust'):
              part_list = dataset3[scene]
          elif (category=='Fog'):
              part_list = dataset4[scene]
          elif (category=='Low Light'):
              part_list = dataset5[scene]
          elif (category=='Rain'):
              part_list = dataset6[scene]
              
          for part in part_list:
          
              print ('Score calculation for ->>> ' + category + ' / ' + scene+ ' / ' + part)
          
              for e in range(from_epoch, to_epoch+1): 
              
                 tp = 0
                 tn = 0
                 fp = 0
                 fn = 0


                 lbl_add = r'G:\TU-VDN_1\\'+category+'\\'+scene+'\\'+part+'_GT_gt_256\\'
                 pst_add = r'G:\Results_100p_TUVDN\g_2'+'\\'+category+'\\'+scene+'\\'+part+'\predicted_loop_v2\e'+str(e).zfill(3)+'\Cd\\'
                 
                 L_list = glob.glob(os.path.join(lbl_add,'*p'))
                 L_list = sorted(L_list)
                 L_list = L_list[49:]
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
                         elif(y_true[j]==0 and y_pred[j]==0):
                             tn +=1
                         elif(y_true[j]==0 and y_pred[j]==255):
                             fp +=1
                         elif(y_true[j]==255 and y_pred[j]==0):
                             fn+=1
		
		
		
                 print('\n'+' epoch= '+str(e)+' category= '+category+' scene= '+scene+' part= '+part+' tp= '+str(tp)+' fp= '+str(fp)+' tn= '+str(tn)+' fn= '+str(fn)+'\n')

		
                 res_list.append('epoch ->>> '+str(e)+'category ->>> '+category+' , scene ->>> '+scene+' , part ->>> '+part+' , True Positive ->>> '+str(tp)+' , False Positive ->>> '+str(fp)+' , True Negative ->>> '+str(tn)+' , False Negative ->>> '+str(fn))
		
				
		
with open(img_save_loc, 'wb') as fp:      # img_save_loc[i]
	pickle.dump(res_list, fp)



    
