import cv2
import numpy as np
import os

def create_or_clear_directory(directory_path):
    try:
        # Kiểm tra nếu thư mục đã tồn tại
        if os.path.exists(directory_path):
            # Xóa hết dữ liệu trong thư mục
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                except Exception as e:
                    print(f"Không thể xóa {file_path}: {e}")
        else:
            # Nếu thư mục chưa tồn tại, tạo mới nó
            try:
                os.makedirs(directory_path)
            except Exception as e:
                print(f"Không thể tạo thư mục {directory_path}: {e}")
        return True
    except Exception as e:
        print(f"Không thể làm gì thư mục {directory_path}: {e}")
        return False

def image_transparent(image):
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
    except:
        return image

def create_criteria_images(image_path,image):
    try:
        image = cv2.resize(image, (224, 224))
        transparent_image = image_transparent(image)
        size = transparent_image.shape[0]
        #path = "criteria_images"
        create_or_clear_directory(image_path)
        # Xoay từ 0 đến 359 độ với bước xoay 1 độ
        angle = 0
        while angle <= 360:
            rotation_matrix = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1)
            # Thực hiện xoay ảnh
            rotated_image = cv2.warpAffine(transparent_image, rotation_matrix, (size, size))
            filename = f'{image_path}/image_{angle}.jpg'
            cv2.imwrite(filename, rotated_image)
            angle += 4
        return True
    except:
        pass
    return False
