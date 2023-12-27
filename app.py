from flask import Flask, request, jsonify
app = Flask(__name__)
import cv2
import base64
import numpy as np
import threading
import time
from datetime import datetime, timedelta
import sys
import tensorflow as tf
######### kHAI BÁO fILE LIÊN KẾT  ################
from Image_fuc import  main_task
from sbLog import sbLogWrite
import os

from create_image import create_or_clear_directory

######### kHAI BÁO fILE LIÊN KẾT  ################

######### kHAI BÁO BIẾN  ################
HOST = '0.0.0.0' 
PORT = 8005    
INDEX_SAVE = 0 
######### kHAI BÁO BIẾN  ################

######### kHAI BÁO VGG16 và VGG19  ################
start_time = time.time()

from keras.models import Model
import keras

model = keras.applications.ResNet50(weights='imagenet', include_top=False)
feat_extractor = Model(inputs=model.input, outputs=model.layers[-1].output)

#model = keras.applications.VGG16(weights='imagenet', include_top=True)
#feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)


import torch
import clip
device_torch = "cuda" if torch.cuda.is_available() else "cpu"
model_torch, preprocess_torch = clip.load("ViT-B/32", device=device_torch)

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Time load model: {elapsed_time:.2f} seconds')
######### kHAI BÁO VGG16 và VGG19  ################


########################################################################################

# Đọc ảnh gốc

def main_transparent_image(image):
    try:
        # Tạo một ảnh trống (màu trong suốt) cùng kích thước với ảnh gốc
        size = image.shape[0]
        transparent_image = np.zeros((size, size, 4), dtype=np.uint8)  # 4 channels (RGBA)
        # Tạo một mặt nạ hình tròn trắng
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (size // 2, size // 2), size // 2, 255, -1)
        # Sao chép hình tròn vào ảnh trong suốt (RGBA channels)
        transparent_image[:, :, :3][mask == 255] = image[:, :, :3][mask == 255]
        transparent_image[:, :, 3][mask == 255] = 255  # Set alpha channel to 255 for the circular region
        
        image_without_alpha = transparent_image[:, :, :3]

        return image_without_alpha
        #return transparent_image
    except:
        return image

def main_resize_object_image(input_image_path,output_image_path):
    try:
        # Đường dẫn đến tấm ảnh gốc
       
        # Đọc tấm ảnh gốc bằng OpenCV
        input_image = cv2.imread(input_image_path)
        input_image = cv2.resize(input_image, (224, 224))
        # Kích thước của tấm ảnh mới (128x128)
        new_size = (128, 128)
        # Tính toán điểm trung tâm
        height, width, _ = input_image.shape
        left = (width - new_size[0]) // 2
        top = (height - new_size[1]) // 2
        # Cắt tấm ảnh từ trung tâm
        cropped_image = input_image[top:top+new_size[1], left:left+new_size[0]]
        output_image = cv2.resize(cropped_image, (224, 224))
        # Lưu tấm ảnh đầu ra
        cv2.imwrite(output_image_path, output_image)
        print("main_resize_object_image OK ................")
    except:
        pass





thread_result = None


def worker_thread(type_data,info_criteria_data,img1, img2,data_log):
    global thread_result
    global model
    global feat_extractor
    global INDEX_SAVE

    global device_torch,model_torch,preprocess_torch #model CLIP
    try: 
        thread_result = ""
        path = '/Users/it/Programing/AI-Object-Recognition-Python/AI-Object-Recognition-Python/' # use mac os 
        #path = ''
        images_path = path + 'criteria_images/' + info_criteria_data
        print(images_path)

        if type_data == "F" or type_data == "S" :

            os.makedirs(images_path, exist_ok=True)
            os.makedirs(images_path+"/Criteria", exist_ok=True)
            os.makedirs(images_path+"/Object", exist_ok=True)

            if type_data == "S":
                filename = f'{images_path}/Criteria.jpg'
                print(filename)
                cv2.imwrite(filename, img1)
                output_image_path = f'{images_path}/Criteria_1.jpg'
                print(output_image_path)
                main_resize_object_image(filename,output_image_path)

            if type_data == "F":
                
                filename = f'{images_path}/Criteria.jpg'
                cv2.imwrite(filename, img1)
                filename = f'{images_path}/Object1.jpg'
                cv2.imwrite(filename, img2)

                output_image_path = f'{images_path}/Object.jpg'
                print(output_image_path)
                main_resize_object_image(filename,output_image_path)
            



            sbLogWrite("OK",str(data_log)+ "==(_img1)  "+ str(type_data) + "================("+str(img1.shape)+")===============")
            sbLogWrite("OK",str(data_log)+ "==(_img2)  "+ str(type_data) + "================("+str(img2.shape)+")===============")

            print("worker_thread main_task ................")
            start_time = time.time()
            result = main_task(type_data,info_criteria_data,model,feat_extractor,device_torch,model_torch,preprocess_torch)
            thread_result = result
            end_time = time.time()
            elapsed_time = end_time - start_time
            sbLogWrite("OK",str(data_log) + "==(OK)  "+ str(type_data) + "================(worker_thread Time: "+ str(elapsed_time) +")===============")
        
        elif type_data == "I":

            INDEX_SAVE = INDEX_SAVE + 1
            #thêm hình vào thư mục
            sbLogWrite("OK",str(data_log)+ "==(_img1)  "+ str(type_data) + "=====(Insert)===========("+str(img1.shape)+")===============")
            filename = f'{images_path}/Criteria_{INDEX_SAVE}.jpg'
            cv2.imwrite(filename, img1)

            thread_result = '{"INSERT": "OK"}'
            
        
        elif type_data == "D":
            INDEX_SAVE = 0
            check = create_or_clear_directory(images_path)
            if(check):
                print("Xóa thành công thư mục : " + images_path)
                thread_result = '{"DELETE": "OK"}'
            else:
                print("OK","Xóa không được thư mục : " + images_path)
                thread_result = '{"DELETE": "ERROR"}'
            
            
        
        else:
            thread_result = {"NULL": 0}

        
    except:
        sbLogWrite("Error",str(data_log) + "==(Error)  "+ str(type_data) + "================(worker_thread)===============")
        pass

# Định nghĩa một tuyến đường (route) POST
@app.route('/ai/image', methods=['POST'])
def handle_post_request():

    try:
        # Hàm xử lý log file vào thư mục LOG
        _time = datetime.now() + timedelta(days=0, hours=0)
        data_log = _time.strftime('%H-%M-%S-%f')

        sbLogWrite("OK",str(data_log)+ "==(OK)================(Begin)===============")
        data = request.json
        type_data = data.get('type')
        info_criteria_data = data.get('info_criteria')
        image1_data = data.get('image1')
        image2_data = data.get('image2')
        if image1_data is not None and image2_data is not None:
            #print("Dữ liệu image1:", image1_data)
            #print("Dữ liệu image2:", image2_data)
            if type_data == "F":
                split_image1 = base64.b64decode(image1_data)
                image_array1 = np.frombuffer(split_image1, np.uint8)
                image_array1 = cv2.imdecode(image_array1, cv2.COLOR_BGR2RGB)
                sbLogWrite("OK",str(data_log)+ "==(Shape1) "+ str(type_data) + "================("+str(image_array1.shape)+")===============")

                split_image2 = base64.b64decode(image2_data)
                image_array2 = np.frombuffer(split_image2, np.uint8)
                image_array2 = cv2.imdecode(image_array2, cv2.COLOR_BGR2RGB)
                sbLogWrite("OK",str(data_log)+ "==(Shape2) "+ str(type_data) + "================("+str(image_array2.shape)+")===============")

            elif type_data == "S" or type_data == "I":

                split_image1 = base64.b64decode(image1_data)
                image_array1 = np.frombuffer(split_image1, np.uint8)
                image_array1 = cv2.imdecode(image_array1, cv2.COLOR_BGR2RGB)
                sbLogWrite("OK",str(data_log)+ "==(Shape1) "+ str(type_data) + "================("+str(image_array1.shape)+")===============")

                split_image2 = base64.b64decode(image1_data)
                image_array2 = np.frombuffer(split_image2, np.uint8)
                image_array2 = cv2.imdecode(image_array2, cv2.COLOR_BGR2RGB)
                #sbLogWrite("OK",str(data_log)+ "==(Shape2)================("+str(image_array2.shape)+")===============")
            elif type_data == "D":
                image_array1= []
                image_array2 = []

            

            thread = threading.Thread(target=worker_thread, args=(type_data,info_criteria_data,image_array1, image_array2,str(data_log)))
            thread.start()
            thread.join()


            if thread_result is not None:
                   
                data = thread_result
                sbLogWrite("data",str(data_log) +"=(result)="+ data)
                sbLogWrite("OK",str(data_log)+ "==(OK)================(End)===============")
                return data, 200
            else:
                return jsonify({"error": "Không có kết quả"}), 400

        else:
            sbLogWrite("Error","==Error===============================")
            return jsonify({"error": "Dữ liệu image1 hoặc image2 không tồn tại trong yêu cầu."}), 400

    except Exception as e:
        sbLogWrite("Error",str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    try:
        app.run(host=HOST, port=PORT, debug=False)
    except Exception as e:
        sbLogWrite("Error",str(e))

