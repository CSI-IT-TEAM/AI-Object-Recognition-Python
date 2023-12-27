import tensorflow as tf
import os
import random
import pickle as cPickle
import numpy as np
from numpy import savetxt, loadtxt, load, save
import matplotlib.pyplot
from matplotlib.pyplot import imshow
import keras
from keras.preprocessing import image
from keras.preprocessing.image import  save_img
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance
import cv2
import time

######### kHAI BÁO fILE LIÊN KẾT  ################
from Check_KMeans_colors import KMeans_1colors
######### kHAI BÁO fILE LIÊN KẾT  ################

def Model_Fit_Accuracy(storage_vector_path,path_image,model,feat_extractor):
    #model = keras.applications.VGG16(weights='imagenet', include_top=True)
    #feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    # model = keras.applications.ResNet50(weights='imagenet', include_top=True)
    # feat_extractor = Model(inputs=model.input, outputs=model.layers[-1].output)
    
    image_path = path_image +'Object1.jpg'

    q_image = image.load_img(image_path, target_size=(224, 224))
    image_object = image.img_to_array(q_image)

    num_variations = 20
    image_1 = cv2.imread(image_path)
    delta_e,check_color = KMeans_1colors(image_1,'')
    if(check_color == 1):
        num_variations = 5

    #feat_query = feat_extractor.predict(image_object.reshape(1, 224, 224, 3), verbose=0).flatten()

    features = []

    if (os.path.isfile(storage_vector_path)):
        features = load(storage_vector_path)
        if (len(features) == 0):
            print("Không tìm thấy Argument Vector Images")
    else:
        print("Không tìm thấy Argument Vector Images")
        # do a query on a random image
    distance_accuracy = []

    # Khởi tạo Sequential model để tạo biến thể
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.experimental.preprocessing.RandomRotation(
            factor=0.0873,  # Sử dụng góc 360 độ #radians = degrees * pi / 180
            #factor=(-1.5708, 1.5708),
            fill_mode='constant'
        )
    ])


    for i in range(num_variations):
            augmented_image = model(tf.cast(tf.expand_dims(image_object, 0), tf.float32))
            augmented = augmented_image[0]

            resized_image = tf.image.resize(augmented, (224, 224))
            output_directory = path_image + 'Object/'
            path = output_directory + str(i+1) + ".png"
            print(path)
            save_img(path, resized_image, scale=True)

            q_image = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(q_image)
            feat_query = feat_extractor.predict(x.reshape(1, 224, 224, 3), verbose=0).flatten()

            distances = [distance.cosine(feat_query, feat)
                                for feat in features]
            
            # euclideans = [distance.euclidean(feat_query, feat)
            #                     for feat in features]
            # print(min(euclideans))

            idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:1+1]
            
            for closest in idx_closest:
                distance_accuracy.append((distances[closest]))
            # print("(1)-------------------------------------------------------")
            # print(distance_accuracy)
            # print("(2)-------------------------------------------------------")
            # print(idx_closest)
            # print("(3)-------------------------------------------------------")
            #print(str(idx_closest) + " " + str(distance_accuracy) + " " + str(distances))
            
         
   
    similarity_max = ((1 - min(distance_accuracy)) * 100 )
    print("DATA: ",similarity_max)

    return round(similarity_max,2)

if __name__ == '__main__':
    storage_vector_path = 'criteria_files/SP343846100TEST_1.npy'
    path_image ='criteria_images/SP343846100TEST_1/'
    model = keras.applications.ResNet50(weights='imagenet', include_top=False)
    feat_extractor = Model(inputs=model.input, outputs=model.layers[-1].output)
    Model_Fit_Accuracy(storage_vector_path,path_image,model,feat_extractor)
    