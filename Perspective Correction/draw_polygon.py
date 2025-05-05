import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import xml.etree.ElementTree as ET
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 从XML文件中提取位置坐标和面积信息
def extract_object_info_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    objects_info = {}

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in objects_info:
            objects_info[name] = []

        polygon = obj.find('polygon')
        coords = []
        for i in range(1, len(polygon) // 2 + 1):
            x = float(polygon.find(f'x{i}').text)
            y = float(polygon.find(f'y{i}').text)
            coords.append((x, y))

        objects_info[name].append({
            'coords': coords,
        })

    return objects_info


def create_object_mask(image, objects_info, object_name):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    if object_name in objects_info:
        for obj in objects_info[object_name]:
            coords = obj['coords']
            points = np.array(coords, np.int32)
            points = points.reshape((-1, 1, 2))

            cv2.fillPoly(mask, [points], 255)

    return mask


def extract_dominant_color_kmeans(image, mask, n_clusters=2):
    object_area = cv2.bitwise_and(image, image, mask=mask)
    object_area_hsv = cv2.cvtColor(object_area, cv2.COLOR_BGR2HSV)
    object_area_reshaped = object_area_hsv.reshape((-1, 3))
    object_area_reshaped = object_area_reshaped[np.all(object_area_reshaped != [0, 0, 0], axis=1)]

    sse = []
    for k in range(1, 4):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(object_area_reshaped)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 4), sse, marker='o', color='b')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    # plt.show()

    diffs = np.diff(sse)
    optimal_k = np.argmin(diffs) + 2
    print(f"肘部法则推荐的最优聚类数：{optimal_k}")
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans.fit(object_area_reshaped)
    dominant_colors_hsv = kmeans.cluster_centers_

    labels = kmeans.labels_
    label_counts = np.bincount(labels)
    most_frequent_label = np.argmax(label_counts)

    most_frequent_color_hsv = dominant_colors_hsv[most_frequent_label]
    most_frequent_color_rgb = cv2.cvtColor(np.uint8([[most_frequent_color_hsv]]), cv2.COLOR_HSV2RGB).reshape(-1, 3)
    return most_frequent_color_rgb

# 提取区域并使用 K-means 提取主体颜色，保存并显示图像
def extract_object_color_kmeans(image, mask, object_name,n_clusters=5):
    # 提取区域的主体颜色
    dominant_colors = extract_dominant_color_kmeans(image, mask, n_clusters)

    # 获取最主要的颜色（最占比的颜色）
    print(f"提取到的主色调：{dominant_colors}")
    dominant_color = dominant_colors[0]

    # 通过掩码提取区域
    object_area = cv2.bitwise_and(image, image, mask=mask)
    output_folder="./draw_polygon/extracted_object_area"
    output_image_path = os.path.join(output_folder, f"{object_name}_extracted_object_area.jpg")
    # 保存提取的区域图像
    cv2.imwrite(output_image_path, object_area)
    print(f"提取的区域已保存到: {output_image_path}")

    # 调整显示尺寸
    img_resized = cv2.resize(object_area, (800,800), interpolation=cv2.INTER_AREA)

    # 显示缩放后的图像
    cv2.imshow("Extracted Object", img_resized)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭窗口

    return dominant_color

# 显示图像并保存
def show_and_save_image(image, save_path='./draw_polygon/output_image2.jpg'):
    print(save_path)
    cv2.imwrite(save_path, image)
    img_resized = cv2.resize(image, (800,800), interpolation=cv2.INTER_AREA)

    # 显示缩放后的图像
    cv2.imshow("Saved Image", img_resized)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭窗口

# 在原图基础上填充区域并用主色调填充
def fill_objects_with_color(image, objects_info, object_name, dominant_color):
    # 将主色调转换为 BGR 格式，并确保是整数
    dominant_bgr_color = tuple(int(c) for c in (dominant_color[2], dominant_color[1], dominant_color[0]))

    # 遍历所有区域，使用主色调填充
    if object_name in objects_info:
        for obj in objects_info[object_name]:
            coords = obj['coords']
            points = np.array(coords, np.int32)
            points = points.reshape((-1, 1, 2))

            # 使用主色调填充区域
            cv2.fillPoly(image, [points], dominant_bgr_color)

    return image

# 主函数，执行所有操作
def main(image_path, image_path_col, xml_path, save_path='./draw_polygon/output_image/output_image1.jpg'):
    # 提取位置信息
    objects_info = extract_object_info_from_xml(xml_path)

    # 读取图像
    image = cv2.imread(image_path)
    image_col = cv2.imread(image_path_col)

    # 遍历所有类别
    for object_name in objects_info:
        print(f"正在处理类别: {object_name}")
        if object_name =='window' or object_name =='door' or object_name =='car':
        # 创建掩码
            mask = create_object_mask(image, objects_info, object_name)
        else:
            mask = create_object_mask(image, objects_info, object_name)
            # 对于每个类别，清除与当前类别无关的部分
            for other_object_name in objects_info:
                if other_object_name != object_name:
                    # 获取其他类别的掩码
                    other_mask = create_object_mask(image, objects_info, other_object_name)
                    # 使用位运算将其他类别部分清除（保留当前类别的部分）
                    mask = cv2.bitwise_and(mask, cv2.bitwise_not(other_mask))

        # 提取区域颜色并保存
        dominant_color = extract_object_color_kmeans(image_col, mask,object_name)

        # 绘制区域并填充主体颜色
        image_with_objects = fill_objects_with_color(image, objects_info, object_name, dominant_color)

        # 显示图像并保存
        show_and_save_image(image_with_objects, save_path)

# 调用主函数，传入图像路径和XML路径
image_path = r'D:\Python\buidding_marking\draw_polygon\origin\20250118100841.png'  # 修改为你的图像路径
image_path_col = r'D:\Python\buidding_marking\draw_polygon\ori\IMG_4371.HEIC.jpg'  # 修改为你的彩色图像路径
xml_path = r'D:\Python\buidding_marking\draw_polygon\position\IMG_4371-HEIC_JPG.rf.f8cc5c916e05355a573680ddf4cbe2e8.xml'  # 修改为你的XML路径

main(image_path, image_path_col, xml_path)