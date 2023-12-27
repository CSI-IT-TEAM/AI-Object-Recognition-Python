import cv2
import json
import os
from datetime import datetime, timedelta
import numpy as np
from scipy.spatial import distance

######### kHAI BÁO fILE LIÊN KẾT  ################
from criteria_main import main_check_criteria
from Check_KMeans_colors import KMeans_colors,KMeans_1colors
from CHECK_CLIP_V2 import Check_CosineSimilarity
#from fuc_tf_image import tf_image
#from check_image_vgg16 import tf_image_vgg16
#from fuc_checking import fuc_Checking

######### kHAI BÁO fILE LIÊN KẾT  ################
#C:\Users\tuhieu.it\AppData\Local\Programs\Python\Python39\Scripts\auto-py-to-exe

######### XỬ LÝ DỮ LIỆU  ################
def main_task(type_data,info_criteria_data,model,feat_extractor,device_torch,model_torch,preprocess_torch):
    #time.sleep(1)
    try:
        path = '/Users/it/Programing/AI-Object-Recognition-Python/AI-Object-Recognition-Python/' # use mac os 
        #path = ''
        print("-------------------begin---------------------")
    
        if type_data == "S": 

            check_vgg16,percentage_vgg16 = main_check_criteria(type_data,info_criteria_data,model,feat_extractor)
            data = {
                "AVGVGG": str(round(percentage_vgg16,2))
            }
            
        elif type_data == "F": 

            path_image_1 = path + 'criteria_images/' + info_criteria_data+ '/Criteria.jpg'
            path_image_2 = path + 'criteria_images/' + info_criteria_data+ '/Object1.jpg'

            check_vgg16,percentage_vgg16 = main_check_criteria(type_data,info_criteria_data,model,feat_extractor)
            avgvgg = float(percentage_vgg16)
            print(" ------------------------- VGG16 .NPY  -------------------------")
            print(check_vgg16,percentage_vgg16)
            print(" ------------------------- VGG16 .NPY-------------------------")
           
           
            image_1 = cv2.imread(path_image_1)
            image_2 = cv2.imread(path_image_2)

            delta_e,check_color = KMeans_1colors(image_1,'')
            delta_e_values = KMeans_colors(image_1,image_2,'')
            print("check_color",check_color)
            _flag = False

            if check_color == 1 : 
                print(" TRUONG HOP 1 MAU -- MAX delta_e", max(delta_e),)
                if(min(delta_e_values) <= max(delta_e)*2 and max(delta_e_values) <= max(delta_e)*2):
                    print("DUNG DIEU KIEN DETAE 1")
                    _flag = True

                
                percentage = Check_CosineSimilarity(path_image_1,path_image_2,device_torch,model_torch,preprocess_torch)
                if(float(percentage) >= avgvgg):
                    avgvgg = float(percentage)
               
                print("------------ VGG 1 COLOR---------" + str(avgvgg) + ' %')
                
                
                data = {
                    "AVGVGG": str(round(avgvgg,0))
                }
                print(data)  

            else : 

                print(" TRUONG HOP NHIEU MAU -- MAX delta_e", max(delta_e))

                percentage = Check_CosineSimilarity(path_image_1,path_image_2,device_torch,model_torch,preprocess_torch)
             
                print("------------ CHECK--------" + str(percentage) + ' %')

                #check_vgg16,percentage_vgg16 = main_check_criteria(type_data,info_criteria_data,model,feat_extractor)
                
                print("------------ VGG NHIEU MAU---------" + str(avgvgg) + ' %')
                
                if(round(avgvgg,0)<= round(percentage)):
                    avgvgg = round(percentage)
                    
                
                data = {
                        "AVGVGG": str(round(avgvgg,0))
                }


        json_string = json.dumps(data)

        return json_string
        #return "VGG16:" + str(round(percentage_vgg16,2)) +"|"+ "Difference:" + str(round(difference,2)) +"|"+ "VGG19:" + str(round(percentage_vgg19,2))
    except:
        return "ERROR"
