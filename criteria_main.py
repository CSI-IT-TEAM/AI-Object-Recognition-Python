import os
import random
import pickle as cPickle
import numpy as np
import matplotlib.pyplot
from matplotlib.pyplot import imshow
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from scipy.spatial import distance
from tqdm import tqdm
import cv2
import time
from numpy import savetxt, loadtxt, load, save
from keras.preprocessing.image import ImageDataGenerator
######### kHAI BÁO fILE LIÊN KẾT  ################
from Model_Fit_Accuracy import Model_Fit_Accuracy

from tf_create_image import main_tf_create
######### kHAI BÁO fILE LIÊN KẾT  ################


def main_check_criteria(type_data,info_criteria_data,model,feat_extractor):
    print("main_check_criteria................")


    path = '/Users/it/Programing/AI-Object-Recognition-Python/AI-Object-Recognition-Python/' # use mac os 
    #path = ''

    def delete_file(filename):
        if os.path.exists(filename):  # Kiểm tra xem file có tồn tại không
            os.remove(filename)       # Xóa file nếu nó tồn tại
            print(f"Đã xóa file '{filename}' thành công.")
        else:
            print(f"File '{filename}' không tồn tại.")

    try: 
        storage_vector_path = path + 'criteria_files/' + info_criteria_data + '.npy'
            
        if type_data == "S": 
            #xóa file đang tồn tại trong hệ thống
            
            delete_file(storage_vector_path)
            image_path = path + 'criteria_images/' + info_criteria_data+ '/Criteria.jpg'
            image_path_1 = path + 'criteria_images/' + info_criteria_data+ '/Criteria_1.jpg'
            output_directory = path + 'criteria_images/' + info_criteria_data+ '/Criteria/'

            check = main_tf_create(image_path,image_path_1,output_directory,storage_vector_path,model,feat_extractor)

            #tạo ra 1 file đó 
            #Argument_criteria_create(storage_vector_path,storage_image_path)
            if check:
                return True,100
            else:
                return False,0
            
        elif type_data == "F":

            path_image = path + 'criteria_images/' + info_criteria_data+ '/'

            similarity_max = Model_Fit_Accuracy(storage_vector_path,path_image,model,feat_extractor)

            return True,round(similarity_max,2)
    except:
        return False,0

if __name__ == '__main__':
    try:
        model = keras.applications.VGG16(weights='imagenet', include_top=True)
        feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
        Type = "S"
        #TYPE = "Save Criteria"
        image_1 = cv2.imread('criteria_images/SPTESTSTYLETEST_4/Criteria_1.jpg')
        image_2 = cv2.imread('criteria_images/SPTESTSTYLETEST_4/Criteria_1.jpg')
        info_criteria = "storage_feature_vector_1"
        main_check_criteria(Type,info_criteria,image_1,image_1,model,feat_extractor)
    except:
        pass