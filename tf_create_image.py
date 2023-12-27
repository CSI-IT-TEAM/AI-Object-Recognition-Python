import os
import tensorflow as tf
import keras
from keras.preprocessing.image import  save_img
from keras.preprocessing import image
from keras.models import Model

import numpy as np
from numpy import save
import cv2
import time

######### kHAI BÁO fILE LIÊN KẾT  ################
from Check_KMeans_colors import KMeans_1colors
######### kHAI BÁO fILE LIÊN KẾT  ################

def main_tf_create(image_path,image_path_1,output_directory,storage_vector_path,model,feat_extractor):
    # Đường dẫn đến ảnh gốc
    try:
        os.makedirs(output_directory, exist_ok=True)
        start_time = time.time()
        #model = keras.applications.VGG16(weights='imagenet', include_top=True)
        #feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)  # fc2 is layer

        #model = keras.applications.ResNet50(weights='imagenet', include_top=True)
        #feat_extractor = Model(inputs=model.input, outputs=model.layers[-1].output)

        # Đọc ảnh và chuyển thành tensor

        image_criteria = image.load_img(image_path, target_size=(224, 224))
        image_criteria = image.img_to_array(image_criteria)

        num_variations = 20

        image_1 = cv2.imread(image_path)
        delta_e,check_color = KMeans_1colors(image_1,'')
        if(check_color == 1):
            num_variations = 5
        # Số lượng biến thể bạn muốn tạo
        

        # Khởi tạo Sequential model để tạo biến thể
        model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
            tf.keras.layers.experimental.preprocessing.RandomRotation(
                #factor=1,  # Sử dụng góc 360 độ
                factor=(-1.5708, 1.5708), #Đối với góc -90 độ (trái) tương đương với -1.5708 radian.
                                        #Đối với góc 90 độ (phải) tương đương với 1.5708 radian.
                fill_mode='constant' #{"constant", "reflect", "wrap", "nearest"})
            )
        ])

        new_size = (224, 224)
        features = []

        image_criteria_1 = image.load_img(image_path_1, target_size=(224, 224))
        x = image.img_to_array(image_criteria_1)
        feat = feat_extractor.predict(x.reshape(1, 224, 224, 3), verbose=0).flatten()
        features.append(feat)



        for i in range(num_variations):
            augmented_image = model(tf.cast(tf.expand_dims(image_criteria, 0), tf.float32))
            augmented = augmented_image[0]
            resized_image = tf.image.resize(augmented, (224, 224))

          
            path = output_directory + str(i+1) + ".png"
            print(path)
            save_img(path, resized_image, scale=True)

            q_image = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(q_image)
            feat = feat_extractor.predict(x.reshape(1, 224, 224, 3), verbose=0).flatten()

            features.append(feat)

        features = np.array(features)
        save(storage_vector_path, features)

        #code anh phước

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Time create model: {elapsed_time:.2f} seconds')
        return True
    except:
        return False

if __name__ == '__main__':
    try:
        image_path = 'criteria_images/SP319116011TEST_1/Object/2.png'
        image_path_1 = 'criteria_images/SP319116011TEST_1/Object/2.png'
        output_directory = 'criteria_images/SP319116011TEST_1/Criteria/'
        storage_vector_path = 'criteria_files/SP319116011TEST_1.npy'
        model = keras.applications.ResNet50(weights='imagenet', include_top=False)
        feat_extractor = Model(inputs=model.input, outputs=model.layers[-1].output)
        value = main_tf_create(image_path,image_path_1,output_directory,storage_vector_path,model,feat_extractor)
        print(value)
    except:
        pass