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
from sklearn.metrics.pairwise import cosine_similarity
######### kHAI BÁO fILE LIÊN KẾT  ################
from Check_KMeans_colors import KMeans_1colors
######### kHAI BÁO fILE LIÊN KẾT  ################

def preprocess_and_extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    features_flat = features.flatten()
    return features_flat


def Model_Fit_2_Image(image_path,model):
   
    image_path_1 = image_path +'Object1.jpg'
    image_path_2 = image_path +'Criteria.jpg'

    # Extract features from the images using MobileNetV2
    img1_features = preprocess_and_extract_features(image_path_1, model)
    img2_features = preprocess_and_extract_features(image_path_2, model)

    # Reshape the features to have a consistent shape
    img1_features = img1_features.reshape(1, -1)
    img2_features = img2_features.reshape(1, -1)

    # Compute cosine similarity between the features
    similarity_score = cosine_similarity(img1_features, img2_features)[0, 0]

    print (round(similarity_score,2))
    return round(similarity_score,2)

if __name__ == '__main__':
    
    path_image ='criteria_images/SPTESTSTYLETEST_2/'
    
    model = keras.applications.ResNet50(weights='imagenet', include_top=False)
    feat_extractor = Model(inputs=model.input, outputs=model.layers[-1].output)
    Model_Fit_2_Image(path_image,model)
    