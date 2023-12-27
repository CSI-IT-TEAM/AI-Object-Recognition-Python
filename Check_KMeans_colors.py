import cv2
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances
from collections import Counter
import matplotlib.pyplot as plt 


def KMeans_colors(image1,image2,path):
    try:
        def RGB2HEX(color):
            return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

        def extract_colors(image, num_colors):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixels = image.reshape(-1, 3)

            
            kmeans = KMeans(n_clusters=num_colors,random_state = 1,n_init="auto")
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)

            # Đếm số điểm ảnh trong mỗi cluster
            _, counts = np.unique(kmeans.labels_, return_counts=True)

            total_pixels = len(pixels)
            percentages = (counts / total_pixels) * 100

            return colors,percentages,kmeans.labels_

        num_colors = 5  # Điều chỉnh số màu bạn muốn trích xuất
        colors1,percentages1,labels1 = extract_colors(image1, num_colors)
        colors2,percentages2,labels2 = extract_colors(image2, num_colors)


        


        # Tạo biểu đồ màu (histogram) cho hai hình ảnh
        hist1, _ = np.histogram(labels1, bins=range(0, 6))
        hist2, _ = np.histogram(labels2, bins=range(0, 6))

        # Tính tương tự giữa hai biểu đồ màu (có thể sử dụng các phương pháp khác)
        similarity = 1 - pairwise_distances([hist1], [hist2], metric="cosine")
        val = round(similarity[0][0]*100,2)
        print("=====================Similarity cosine:=========================")
        print("Similarity cosine:", val)
        print("=====================Similarity cosine:=========================")

        from colormath.color_objects import LabColor
        from colormath.color_diff import delta_e_cie1976
        def delta_e(color1, color2):
            lab_color1 = LabColor(color1[0], color1[1], color1[2])
            lab_color2 = LabColor(color2[0], color2[1], color2[2])
            delta_e = delta_e_cie1976(lab_color1, lab_color2)
            return delta_e

        delta_e_values = []

        for i  in colors1:
            for j in colors2:
                de = delta_e(i, j)
                delta_e_values.append(de)

        print("Delta E values for corresponding colors:")
        #print(delta_e_values)
        print("MIN",min(delta_e_values))
        print("MAX",max(delta_e_values))
        print("AVG",np.mean(delta_e_values))
        print("Trung vị Delta E:", np.median(delta_e_values))
       
        print("Do lech chuan Delta E:", np.std(delta_e_values))
        print("MAX - MIN  Delta E:", np.max(delta_e_values) - np.min(delta_e_values))


        # counts = Counter(labels2)
        # counts = dict(sorted(counts.items(), key=lambda item: item[1]))
        # hex_colors = [RGB2HEX(colors1[i]) for i in counts.keys()]
        #plt.suptitle('Colors Detection Object ($n='+str(num_colors)+'$)' , fontsize=20)
        #plt.title("AVG DETAE : " + str(round(np.mean(delta_e_values),2)) + " -- MAX :" + str(round(max(delta_e_values),2)), fontsize=10)
        #plt.pie(counts.values(), labels=hex_colors, colors=hex_colors,autopct='%.0f%%')
        #plt.savefig(path + 'Kmeans_Object.png')

        return delta_e_values
    except:
        return []

def KMeans_1colors(image1,path):
    try:
        def RGB2HEX(color):
            return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

        def extract_colors(image, num_colors):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixels = image.reshape(-1, 3)

            
            kmeans = KMeans(n_clusters=num_colors,random_state = 1,n_init="auto")
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)

            # Đếm số điểm ảnh trong mỗi cluster
            _, counts = np.unique(kmeans.labels_, return_counts=True)

            total_pixels = len(pixels)
            percentages = (counts / total_pixels) * 100

            return colors,percentages,kmeans.labels_

        num_colors = 2  # Điều chỉnh số màu bạn muốn trích xuất
        colors1,percentages1,labels = extract_colors(image1, num_colors)

        counts = Counter(labels)
        counts = dict(sorted(counts.items(), key=lambda item: item[1]))
        

        print("====== 1 IMAGE ===========",colors1)
        from colormath.color_objects import LabColor
        from colormath.color_diff import delta_e_cie1976
        def delta_e(color1, color2):
            lab_color1 = LabColor(color1[0], color1[1], color1[2])
            lab_color2 = LabColor(color2[0], color2[1], color2[2])
            delta_e = delta_e_cie1976(lab_color1, lab_color2)
            return delta_e

        delta_e_values = []

        for i  in colors1:
            for j in colors1:
                try:
                    de = delta_e(i, j)
                    delta_e_values.append(de)
                except:
                    print("eee")

        print("====== 0000 IMAGE ===========")
        #print(delta_e_values)
        print("MIN",min(delta_e_values))
        print("MAX",max(delta_e_values))
        print("AVG",np.mean(delta_e_values))
        print("Trung vị Delta E:", np.median(delta_e_values))
        
        print("Do lech chuan Delta E:", np.std(delta_e_values))
        print("MAX - MIN  Delta E:", np.max(delta_e_values) - np.min(delta_e_values))

        print("====== 1 IMAGE ===========",path)

        check_color = 2 # = 1 (1 mau) // = 2(nhieu mau)
        if max(delta_e_values) <= 15 and np.mean(delta_e_values) <= 15:
            check_color = 1

        # hex_colors = [RGB2HEX(colors1[i]) for i in counts.keys()]
        # plt.suptitle('Colors Detection Criteria' , fontsize=20)
        # plt.title("AVG DETAE : " + str(round(np.mean(delta_e_values),2)) + " -- MAX :" + str(round(max(delta_e_values),2)), fontsize=10)
        # plt.pie(counts.values(), labels=hex_colors, colors=hex_colors,autopct='%.0f%%')
        # plt.savefig(path + 'Kmeans_Criteria.jpg')

        return delta_e_values,check_color
    except:
        return [],2


if __name__ == '__main__':
    try:
        image1 = cv2.imread('criteria_images/0/Criteria/1.png')
        # image2 = cv2.imread('criteria_images/0/Criteria/2.png')
        # delta_e_values = KMeans_colors(image1,image2,'')
        # print(delta_e_values)
        delta_e_values,_ = KMeans_1colors(image1,'')
        print(delta_e_values)
    except:
        pass

