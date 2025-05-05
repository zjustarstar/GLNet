import os
from PIL import Image

# 设置文件夹路径
folder1_path = r"D:\project\RTFP-improve\data\classall\translated_data_test_hznu_zhengmian_V2\color_mask"  # 第一个文件夹路径
folder2_path = r"D:\project\RTFP-improve\demo\ours"  # 第二个文件夹路径

# 获取第一个文件夹中的所有PNG文件
folder1_images = [f for f in os.listdir(folder1_path) if f.endswith('.png')]

# 遍历文件夹1中的图像文件
for image_name in folder1_images:
    # 获取对应的第二个文件夹中的图像名称
    corresponding_image_name = image_name.replace(".png", "_segmentation1.png")

    # 构建文件的完整路径
    image1_path = os.path.join(folder1_path, image_name)
    image2_path = os.path.join(folder2_path, corresponding_image_name)

    # 如果第二个文件夹中的图像存在
    if os.path.exists(image2_path):
        # 打开第一个文件夹中的图像和第二个文件夹中的图像
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # 调整第二个文件夹中的图像大小以匹配第一个文件夹中的图像
        resized_image2 = image2.resize(image1.size)

        # 保存调整大小后的图像
        resized_image2.save(image2_path)
        print(f"已调整 {corresponding_image_name} 的大小")
    else:
        print(f"未找到对应图像: {corresponding_image_name}")
