import csv
import pickle
import os
import glob
from PIL import Image 
import numpy as np
import cv2
from keras.preprocessing import image as kImage


img_save_loc = '/home/tyrone1/murari/Vansh Data/3d_lasiesta_results/3d_l_result_whole_v2_120.txt'



dataset_1 = ['Indoor Sequences', 'Outdoor Sequences']

dataset2 = {
            'Indoor Sequences':['bootstrap', 'camouflage', 'illumination changes', 'modified background', 'moving camera', 'occlusions', 'simple sequences', 'simulated motion'],                  
            'Outdoor Sequences':['cloudy conditions', 'moving camera', 'rainy conditions', 'simulated motion', 'snowy conditions', 'sunny conditions'],    
                
}

dataset3 = {
            'bootstrap':['I_BS_01', 'I_BS_02'],   
	    'camouflage':['I_CA_01', 'I_CA_02'],   
            'illumination changes':['I_IL_01', 'I_IL_02'],   
	    'modified background':['I_MB_01', 'I_MB_02'],   
            'moving camera':['I_MC_01', 'I_MC_02'],   
            'occlusions':['I_OC_01', 'I_OC_02'],   
            'simple sequences':['I_SI_01', 'I_SI_02'],   
            'simulated motion':['I_SM_01', 'I_SM_02', 'I_SM_03', 'I_SM_04', 'I_SM_05', 'I_SM_06', 'I_SM_07', 'I_SM_08', 'I_SM_09', 'I_SM_10', 'I_SM_11', 'I_SM_12'],   
}
dataset4 = {               

            'cloudy conditions':['O_CL_01', 'O_CL_02'],  
            'moving camera':['O_MC_01', 'O_MC_02'], 
            'rainy conditions':['O_RA_01', 'O_RA_02'], 
            'simulated motion':['O_SM_01', 'O_SM_02', 'O_SM_03', 'O_SM_04', 'O_SM_05', 'O_SM_06', 'O_SM_07', 'O_SM_08', 'O_SM_09', 'O_SM_10', 'O_SM_11', 'O_SM_12'], 
            'snowy conditions':['O_SN_01', 'O_SN_02'], 
            'sunny conditions':['O_SU_01', 'O_SU_02'],  
}



res_list = []


for category in dataset_1:   

    scene_list = dataset2[category]
   
    for scene in scene_list:
 
		if (category=='Indoor Sequences'):
		   part_list = dataset3[scene]
		elif (category=='Outdoor Sequences'):
		   part_list = dataset4[scene]

           	for part in part_list:
		  		print ('Score calculation for ->>> ' + category + ' / ' + scene + ' / ' + part)
				tp = 0
				tn = 0
				fp = 0
				fn = 0


				lbl_add = '/home/tyrone1/murari/Vansh Data/LASIESTA/'+category+'/'+scene+'/'+part+'/bw_groundtruth_256/'
				pst_add = '/home/tyrone1/murari/Vansh Data/LASIESTA/'+category+'/'+scene+'/'+part+'/predicted/'

				L_list = glob.glob(os.path.join(lbl_add,'*g'))
				L_list = sorted(L_list)
				L_list = L_list[49:]
				P_list = glob.glob(os.path.join(pst_add,'*g'))
				P_list = sorted(P_list)
                                print(len(L_list))
                                print(len(P_list))

				for k,(lbl,pst) in enumerate(zip(L_list, P_list)):
					print('comp no. ->>> '+str(k+1)+' / '+str(len(L_list)))
					y_true = cv2.imread(lbl, 0)
					y_true = kImage.img_to_array(y_true)

					y_pred = cv2.imread(pst, 0)
					y_pred = kImage.img_to_array(y_pred)

					y_true = y_true.reshape(-1)
					y_pred = y_pred.reshape(-1)

					for j in xrange(len(y_true)):
	
						if(y_true[j]==255 and y_pred[j]==255):
							tp +=1
						elif(y_true[j]<=50 and y_pred[j]==0):
							tn +=1
						elif(y_true[j]<=50 and y_pred[j]==255):
							fp +=1
						elif(y_true[j]==255 and y_pred[j]==0):
							fn+=1
		
		
		
				print('\n'+' tp= '+str(tp)+' fp= '+str(fp)+' tn= '+str(tn)+' fn= '+str(fn)+'\n')

		
				res_list.append('category ->>> '+category+' , scene ->>> '+scene+' ,  part ->>> '+part+' , True Positive ->>> '+str(tp)+' , False Positive ->>> '+str(fp)+' , True Negative ->>> '+str(tn)+' , False Negative ->>> '+str(fn))
		
				"""
				precision = tp/(tp + fp)
				recall = tp/(tp + fn)
				f_measure = (2 * precision * recall)/(precision + recall)
				pwc = 100*(fn + fp)/(fp + fn + tp + tn)
				fpr = fp/(fp + tn)
				fnr = fn/(fp + tn)
				specificity = tn/(tn + fp)


				print('precision ->> '+str(precision))
				print('recall ->> '+str(recall))
				print('pwc ->> '+str(pwc))
				print('fpr ->> '+str(fpr))
				print('fnr ->> '+str(fnr))
				print('specificity ->> '+str(specificity))
				print('f_measure ->> '+str(f_measure))
				"""
		
with open(img_save_loc, 'wb') as fp:      # img_save_loc[i]
	pickle.dump(res_list, fp)



    

