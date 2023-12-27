import torch
import clip
from PIL import Image
import torch.nn as nn
import time

def map_value(value, from_min, from_max, to_min, to_max):
    
    try:# Ánh xạ giá trị từ phạm vi ban đầu sang phạm vi mới
        if(value <= from_min):
            return 0
        elif(value >= from_max):
            return 100

        from_range = from_max - from_min
        to_range = to_max - to_min
        scaled_value = (value - from_min) / from_range
        new_value = to_min + (scaled_value * to_range)
        return round(new_value,1)
    except:
        return 0
# Load and preprocess the image using TensorFlow
def Check_CosineSimilarity(image_path_1,image_path_2,device_torch,model_torch,preprocess_torch):
    try:
        start_time = time.time()
        # device_torch = "cuda" if torch.cuda.is_available() else "cpu"
        # model_torch, preprocess_torch = clip.load("ViT-B/32", device=device_torch)

        image1 = image_path_1
        image2= image_path_2
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Time load model: {elapsed_time:.2f} seconds')
        print('===============Check_CosineSimilarity Torch CLIP==================')
        cos = torch.nn.CosineSimilarity(dim=0)
        with torch.no_grad():
            image1_preprocess = preprocess_torch(Image.open(image1)).unsqueeze(0).to(device_torch)
            image1_features = model_torch.encode_image( image1_preprocess)

            image2_preprocess = preprocess_torch(Image.open(image2)).unsqueeze(0).to(device_torch)
            image2_features = model_torch.encode_image( image2_preprocess)

            similarity = cos(image1_features[0],image2_features[0]).item()
            similarity = (similarity+1)/2
            print("Image similarity", similarity*1000)
            result_1 = map_value(similarity*1000, 970, 996, 87, 100)
            print(result_1,"%")

            _similarity = torch.nn.functional.cosine_similarity(image1_features, image2_features)
            print('Similarity score:', _similarity.item()*1000)
            result_2 = map_value(_similarity.item()*1000, 950, 996, 87, 100)
            print(result_2,"%")

            result = (result_1 + result_2)/2
            result = result_1
        return round(result,0)
        '''similarity_score = 0
        print(round(_similarity.item(),2))
        
        if round(_similarity.item(),2) <= 0.94 :
            similarity_score = 0
        elif round(similarity,3) >= 0.990 and round(_similarity.item(),3) >= 0.990 :
            similarity_score = similarity*100
            return round(similarity_score,0)
        elif round(similarity,3) >= 0.990 and round(_similarity.item(),3) >= 0.980 :
            similarity_score = _similarity*100
        elif round(similarity,3) >= 0.980 and round(_similarity.item(),3) >= 0.980 :
            similarity_score = _similarity*100 - 1
        else:
            similarity_score = round(similarity,3)*100 - 12
    

        print('similarity_score:', round(similarity_score,0))
        print('------ % ------', round(result_1,0))

        print('===============Check_CosineSimilarity Torch CLIP==================')
        #return round(similarity_score,0)
        '''
    except:
        return 0
if __name__ == '__main__':
    try:
        device_torch = "cuda" if torch.cuda.is_available() else "cpu"
        model_torch, preprocess_torch = clip.load("ViT-B/32", device=device_torch)
        similarity_score = Check_CosineSimilarity("test/005.png","test/005.png",device_torch,model_torch,preprocess_torch)
        print(similarity_score)
    except:
        pass
