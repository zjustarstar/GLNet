import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 图像和结果的保存路径
image_folder = r'E:\xmsb\GLNet\Perspective Correction\duogelimian'
output_folder = r'E:\xmsb\GLNet\Perspective Correction\result'
matrix_folder = r'E:\xmsb\GLNet\Perspective Correction\Test\example_point\matrices'
points_folder = r'E:\xmsb\GLNet\Perspective Correction\Test\example_point\point'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(matrix_folder):
    os.makedirs(matrix_folder)

# 计数器
total_images = 0
saved_images = 0
skipped_images = 0

# 处理每张图片
for filename in sorted(os.listdir(image_folder)):
    # 不管是什么后缀的文件都处理
    if filename:  # 仅当文件名非空时
        total_images += 1  # 增加总图像计数
        file_path = os.path.join(image_folder, filename)
        image = cv2.imread(file_path)

        if image is None:  # 确保图像加载成功
            print(f"无法加载图像：{filename}，跳过")
            skipped_images += 1
            continue

        orig_image = image.copy()
        height, width = image.shape[:2]

        # 提取图像扩展名
        base_filename, ext = os.path.splitext(filename)

        # 动态构建角点文件路径
        points_file_path = os.path.join(points_folder, f"{base_filename}.txt")
        print(f"检查角点文件路径：{points_file_path}")

        if not os.path.exists(points_file_path):
            print(f"角点文件不存在：{points_file_path}，跳过图像 {filename}")
            skipped_images += 1  # 增加跳过计数
            continue

        points = np.loadtxt(points_file_path, dtype=np.float32)
        print(f"角点文件内容：{points}")
        if points.shape != (4, 2):
            print(f"角点文件格式错误：{points_file_path}，需要4个点，跳过图像 {filename}")
            skipped_images += 1  # 增加跳过计数
            continue

        # 计算目标点
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        target_width = int(max_x - min_x)
        target_height = int(max_y - min_y)
        pts2 = np.float32([
            [min_x, min_y],
            [min_x, min_y + target_height],
            [min_x + target_width, min_y + target_height],
            [min_x + target_width, min_y]
        ])

        print(f"目标矩形 pts2: {pts2}")

        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(points, pts2)
        result = cv2.warpPerspective(orig_image, matrix, (width, height))

        # 保存变换矩阵到文件夹
        matrix_file_path = os.path.join(matrix_folder, f"{base_filename}.txt")
        np.savetxt(matrix_file_path, matrix, fmt='%f')
        print(f"保存了矩阵：{matrix_file_path}")

        # 保存变换后的图像
        result_file_path = os.path.join(output_folder, filename)
        plt.imsave(result_file_path, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        print(f"处理并保存了图像：{filename}")
        saved_image = cv2.imread(result_file_path)
        saved_image = cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 以正确显示颜色

        plt.figure(figsize=(8, 6))  # 设置显示窗口大小
        plt.imshow(saved_image)
        plt.axis("off")  # 隐藏坐标轴
        plt.title(f"Saved Image: {filename}")
        plt.show()

        saved_images += 1  # 增加保存计数

print(f"总图像数: {total_images}")
print(f"保存的图像数: {saved_images}")
print(f"跳过的图像数: {skipped_images}")
print("所有图像处理完成。")
