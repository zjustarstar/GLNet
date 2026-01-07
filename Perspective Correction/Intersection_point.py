import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from PIL import Image, ImageOps
from sklearn.linear_model import LinearRegression
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# --------------------------- 计算工具 ---------------------------

# 计算线段长度
def calculate_line_length(line):
    x1, y1, x2, y2 = line
    return np.hypot(x2 - x1, y2 - y1)  # 计算两点之间的欧几里得距离


# 计算单线角度，角度范围0到180度。
def calculate_line_angle(x1, y1, x2, y2):
    """
    计算直线的角度，并将角度规范到0到180度的范围内。

    :param x1: 直线起点的x坐标
    :param y1: 直线起点的y坐标
    :param x2: 直线终点的x坐标
    :param y2: 直线终点的y坐标
    :return: 规范后的角度（整数）
    """
    # 计算直线的斜率
    dx = x2 - x1
    dy = y1 - y2

    # 使用atan2计算角度（以弧度为单位）
    angle_rad = np.arctan2(dy, dx)

    # 将角度转换为度数
    angle_deg = np.degrees(angle_rad)

    # 将角度规范到0到180度的范围内
    if angle_deg < 0:
        angle_deg += 180
    elif angle_deg > 180:
        angle_deg = 360 - angle_deg

    # 保留角度的整数部分
    angle_int = int(angle_deg)

    return angle_int


# 计算单线的角度，并根据角度范围计算其x截距或y截距，保留整数部分
def calculate_intercepts(line):
    """
    计算给定直线的角度，并根据角度范围计算其x截距或y截距，并保留整数部分。

    :param line: 一个包含四个浮点数的数组 [x1, y1, x2, y2]，表示直线的两个端点。
    :return: 如果角度在45°到135°之间，返回x截距；否则返回y截距。
    """
    x1, y1, x2, y2 = line

    # 计算直线的角度
    angle = calculate_line_angle(x1, y1, x2, y2)

    # 计算斜率
    dx = x2 - x1
    dy = y2 - y1

    # 避免除零错误
    if dx == 0:
        # 垂直线
        return (None, int(y1))  # 返回y截距
    elif dy == 0:
        # 水平线
        return (int(x1), None)  # 返回x截距

    # 计算斜率
    m = dy / dx

    # 计算y轴截距
    b = y1 - m * x1

    # 根据角度范围计算x截距或y截距
    if 45 <= angle <= 135:
        # 计算x截距
        x_intercept = -b / m
        return (int(x_intercept), None)
    else:
        # 计算y截距
        y_intercept = b
        return (None, int(y_intercept))


def draw_lines(img, lines, colors, output_path, kd=3, display_size=(800, 600)):
    img_copy = img.copy()  # 创建图像副本以便绘制
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = map(int, line)  # 获取线段起点和终点的整数坐标
        color = colors[i]  # 获取对应颜色
        cv2.line(img_copy, (x1, y1), (x2, y2), color, kd, cv2.LINE_AA)  # 在线段绘制

    cv2.imwrite(output_path, img_copy)  # 保存图像

    # 调整显示尺寸
    img_resized = cv2.resize(img_copy, display_size, interpolation=cv2.INTER_AREA)

    # cv2.imshow("Result", img_resized)  # 展示缩放后的图像
    # cv2.waitKey(0)  # 等待按键
    # cv2.destroyAllWindows()  # 关闭窗口


# 生成唯一颜色
def generate_unique_colors(n):
    used_colors = set()
    colors = []
    while len(colors) < n:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        if color not in used_colors:
            colors.append(color)
            used_colors.add(color)  # 添加到已使用的颜色集合中
    return colors


# --------------------------- 直线检测部分 ---------------------------

def detect_and_draw_lines(img_path, output_path1, output_path2, length_threshold):
    """
    检测并绘制图像中的直线段。
    :param img_path: 输入图像路径
    :param output_folder: 输出文件夹路径
    :param length_threshold: 直线段长度阈值，低于此长度的直线段将被排除
    :return: img, lines
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片：{img_path}")
        return None, None

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
    # 创建LSD检测器，自定义参数
    lsd = cv2.createLineSegmentDetector(scale=0.8, sigma_scale=0.6, quant=4.0,
                                        ang_th=22.5, log_eps=0.0, density_th=0.7)
    lines = lsd.detect(img_gray)[0]  # 检测图像中的所有直线段
    print(f"初次检测到的直线数: {len(lines)}")
    if lines is not None:
        lines = lines[:, 0]  # 提取每条直线的坐标 (x1, y1, x2, y2)
        # 过滤掉长度小于阈值的直线段
        lines = [line for line in lines if calculate_line_length(line) >= length_threshold]
        # print(f"长度过滤后检测到的直线数: {len(lines)}")

        # 绘制所有直线段，使用单一颜色（绿色）
        single_color = [(0, 255, 0)] * len(lines)

        draw_lines(img, lines, single_color, output_path1)
        # print(f"检测结果已保存至：{output_path1}")
        # 绘制所有直线段，使用唯一颜色
        unique_colors = generate_unique_colors(len(lines))

        draw_lines(img, lines, unique_colors, output_path2)

        # print(f"检测结果已保存至：{output_path2}")
    else:
        print(f"未检测到任何直线：{img_path}")
        lines = []

    return img, lines


def classify_lines(lines, angle_threshold=0.5, intercept_threshold=4):
    """
    对直线进行分类，基于它们的角度和截距。

    :param lines: 直线列表
    :param angle_threshold: 角度阈值
    :param intercept_threshold: 截距阈值
    :return: 分类后的直线列表
    """
    categories = []
    for line in lines:
        x1, y1, x2, y2 = line
        angle = calculate_line_angle(x1, y1, x2, y2)
        x_intercept, y_intercept = calculate_intercepts(line)

        found_category = False
        for category in categories:
            ref_angle, ref_x, ref_y = category['reference']

            if 45 <= ref_angle <= 135:  # 接近垂直，用 X 截距
                if (abs(ref_angle - angle) <= angle_threshold and
                        ref_x is not None and x_intercept is not None and
                        abs(ref_x - x_intercept) <= intercept_threshold):
                    category['lines'].append(line)
                    found_category = True
                    break
            else:  # 接近水平，用 Y 截距
                if (abs(ref_angle - angle) <= angle_threshold and
                        ref_y is not None and y_intercept is not None and
                        abs(ref_y - y_intercept) <= intercept_threshold):
                    category['lines'].append(line)
                    found_category = True
                    break

        if not found_category:
            categories.append({
                'reference': (angle, x_intercept, y_intercept),
                'lines': [line]
            })
    return categories


def draw_and_save_categories(img, categories, output_path):
    """
    绘制分类后的直线，并保存为图像文件。同类直线使用同一种颜色，不同类直线使用不同颜色。
    :param img: 原始图像
    :param categories: 分类后的直线列表
    :param output_path: 输出图像文件路径
    """
    img_copy = img.copy()  # 创建图像副本以便绘制
    colors = generate_unique_colors(len(categories))  # 生成与类别数相等的不同颜色

    for i, category in enumerate(categories):
        color = colors[i]
        for line in category['lines']:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(img_copy, (x1, y1), (x2, y2), color, 5, cv2.LINE_AA)  # 在线段绘制

    cv2.imwrite(output_path, img_copy)  # 保存绘制后的图像


def process_categories(categories, percentage=0.4):
    # 分桶
    bucket_1 = []  # 45°到135°之间的直线
    bucket_2 = []  # 其他角度的直线

    for category in categories:
        angle, _, _ = category['reference']
        if 45 <= angle <= 135:
            bucket_1.append(category)
        else:
            bucket_2.append(category)

    def sum_line_lengths(bucket):
        # 计算每个类别的总长度并存储
        total_lengths = []
        for cat in bucket:
            total_length = sum(calculate_line_length(line) for line in cat['lines'])
            total_lengths.append((cat, total_length))
        return total_lengths

    # 对于桶一
    bucket_1_total_lengths = sum_line_lengths(bucket_1)
    bucket_1_sorted = sorted(bucket_1_total_lengths, key=lambda x: x[1], reverse=True)

    # 保留前40%
    num_to_keep_bucket_1 = max(1, int(len(bucket_1) * percentage))
    top_categories_bucket_1 = [item[0] for item in bucket_1_sorted[:num_to_keep_bucket_1]]

    # 对于桶二
    bucket_2_total_lengths = sum_line_lengths(bucket_2)
    bucket_2_sorted = sorted(bucket_2_total_lengths, key=lambda x: x[1], reverse=True)

    # 保留前40%
    num_to_keep_bucket_2 = max(1, int(len(bucket_2) * percentage))
    top_categories_bucket_2 = [item[0] for item in bucket_2_sorted[:num_to_keep_bucket_2]]

    # 合并两个桶的结果
    processed_categories = top_categories_bucket_1 + top_categories_bucket_2

    # 返回结果
    return processed_categories


# --------------------------- 直线拟合部分 ---------------------------
# 根据样本点进行直线拟合
def fit_line_from_points(points):
    """
    使用最小二乘法拟合直线，返回拟合后的直线的斜率和截距。

    :param points: 一组样本点 [(x1, y1), (x2, y2), ..., (xn, yn)]
    :return: 拟合后的直线的斜率 m 和截距 b
    """
    points = np.array(points)
    X = points[:, 0].reshape(-1, 1)  # X 为 x 值
    y = points[:, 1]  # y 为 y 值

    model = LinearRegression()
    model.fit(X, y)  # 拟合直线
    m = model.coef_[0]  # 斜率
    b = model.intercept_  # 截距

    return m, b


# 计算投影点的坐标
def project_point_on_line(x, y, line):
    """
    计算给定点 (x, y) 在给定直线上的投影点。
    :param x, y: 需要投影的点的坐标
    :param line: 直线的两点 (x1, y1, x2, y2)
    :return: 投影点的坐标 (px, py)
    """
    x1, y1, x2, y2 = line

    # 计算直线的参数
    dx = x2 - x1
    dy = y2 - y1

    # 计算点 (x, y) 到直线的垂直投影
    t = ((x - x1) * dx + (y - y1) * dy) / (dx ** 2 + dy ** 2)
    px = x1 + t * dx
    py = y1 + t * dy

    return px, py


# 获取每类直线的拟合直线
def fit_lines_for_categories(categories):
    """
    对每类直线进行拟合，并返回拟合的结果。

    :param categories: 分类后的直线列表
    :return: 拟合直线的端点 [(x1, y1), (x2, y2)]
    """
    fitted_lines = []

    for category in categories:
        category_lines = category['lines']
        sample_points = []

        # 遍历该类别中的所有直线
        for line in category_lines:
            x1, y1, x2, y2 = line

            # 从每条直线选取 0/5, 1/5, 2/5, 3/5, 4/5, 5/5 的点
            for i in range(6):
                t = i / 5.0
                sample_x = x1 + t * (x2 - x1)
                sample_y = y1 + t * (y2 - y1)
                sample_points.append((sample_x, sample_y))

        # 对选取的点进行直线拟合
        m, b = fit_line_from_points(sample_points)

        # 根据拟合直线的角度和其他信息来选择端点
        # 假设拟合的直线是 y = mx + b, 根据角度选择端点
        angle = calculate_line_angle(x1, y1, x2, y2)

        if 45 <= angle <= 135:  # 选择y值最大和最小的投影点
            # 查找样本点中，投影后y值最大和最小的点
            projected_points = [project_point_on_line(x, y, (x1, y1, x2, y2)) for x, y in sample_points]
            max_y_point = max(projected_points, key=lambda p: p[1])
            min_y_point = min(projected_points, key=lambda p: p[1])
            fitted_lines.append((max_y_point, min_y_point))
        else:  # 选择x值最大和最小的投影点
            # 查找样本点中，投影后x值最大和最小的点
            projected_points = [project_point_on_line(x, y, (x1, y1, x2, y2)) for x, y in sample_points]
            max_x_point = max(projected_points, key=lambda p: p[0])
            min_x_point = min(projected_points, key=lambda p: p[0])
            fitted_lines.append((max_x_point, min_x_point))

    return fitted_lines


# 主要逻辑函数
def fit_and_draw_fitted_lines(img, categories, output_path):
    """
    对每类直线进行拟合，绘制拟合后的直线，并保存图像。

    :param img: 输入图像
    :param categories: 分类后的直线列表
    :param output_path: 输出路径
    """
    fitted_lines = fit_lines_for_categories(categories)
    return fitted_lines


def convert_fitted_lines_to_lines(fitted_lines):
    """
    将拟合的直线端点列表转换为与原始直线格式相同的 [x1, y1, x2, y2] 格式。

    :param fitted_lines: 拟合后的直线端点列表 [((x1, y1), (x2, y2)), ...]
    :return: 转换后的直线列表 [[x1, y1, x2, y2], ...]
    """
    converted_lines = []
    for (x1, y1), (x2, y2) in fitted_lines:
        converted_lines.append([x1, y1, x2, y2])

    return converted_lines


# --------------------------- 直线过滤部分 ---------------------------


# --------------------------- 直线过滤(四区域)部分 ---------------------------
# 将直线分配到角度桶并按长度排序，保留前40%
def filter_lines_by_length(angle_buckets, threshold):
    filtered_buckets = {}

    for angle, lines in angle_buckets.items():
        # 计算每条直线的长度
        lines_with_length = [(line, calculate_line_length(line)) for line in lines]

        # 按照长度从大到小排序
        lines_with_length.sort(key=lambda x: x[1], reverse=True)

        # 保留前40%的直线
        num_lines_to_keep = int(len(lines_with_length) * threshold)
        filtered_buckets[angle] = [line for line, _ in lines_with_length[:num_lines_to_keep]]

    return filtered_buckets


# 将直线分配到角度桶中
def assign_lines_to_angle_buckets(fitted_lines):
    angle_buckets = defaultdict(list)  # 创建一个字典，键为角度，值为直线列表
    for line in fitted_lines:
        x1, y1, x2, y2 = line
        angle = calculate_line_angle(x1, y1, x2, y2)
        angle_buckets[angle].append(line)  # 根据角度将直线添加到相应桶中
    return angle_buckets


# 处理桶一：45°到135°之间的角度
def process_bucket_one(angle_buckets):
    lines = []
    for angle in range(45, 136):
        if angle in angle_buckets:
            lines.extend(angle_buckets[angle])

    # 对桶一中的直线按 x 截距从大到小排序
    lines_with_x_intercepts = []
    for line in lines:
        x_intercept, _ = calculate_intercepts(line)
        if x_intercept is not None:
            lines_with_x_intercepts.append((line, x_intercept))
    lines_with_x_intercepts.sort(key=lambda x: x[1], reverse=True)

    # 找到排序后的第一条和最后一条直线
    first_line = lines_with_x_intercepts[0][0]
    last_line = lines_with_x_intercepts[-1][0]

    # 计算每条直线的 y 坐标
    min_y = float('inf')  # 初始化为正无穷大
    max_y = float('-inf')  # 初始化为负无穷大
    # 遍历直线并计算最小和最大y坐标
    for line in lines:
        x1, y1, x2, y2 = line
        min_y = min(min_y, y1, y2)  # 更新最小y坐标
        max_y = max(max_y, y1, y2)  # 更新最大y坐标

    return min_y, max_y, first_line, last_line  # 返回第一条和最后一条直线


# 处理桶二：0°到45°和135°到180°之间的角度
def process_bucket_two(angle_buckets):
    lines = []
    for angle in range(0, 45):
        if angle in angle_buckets:
            lines.extend(angle_buckets[angle])
    for angle in range(135, 180):
        if angle in angle_buckets:
            lines.extend(angle_buckets[angle])

    # 对桶二中的直线按 y 截距从大到小排序
    lines_with_y_intercepts = []
    for line in lines:
        _, y_intercept = calculate_intercepts(line)
        if y_intercept is not None:
            lines_with_y_intercepts.append((line, y_intercept))
    lines_with_y_intercepts.sort(key=lambda x: x[1], reverse=True)

    # 找到排序后的第一条和最后一条直线
    first_line = lines_with_y_intercepts[0][0]
    last_line = lines_with_y_intercepts[-1][0]

    # 计算每条直线的 x 坐标
    min_x = float('inf')  # 初始化为正无穷大
    max_x = float('-inf')  # 初始化为负无穷大

    # 遍历直线并计算最小和最大x坐标
    for line in lines:
        x1, y1, x2, y2 = line
        min_x = min(min_x, x1, x2)  # 更新最小x坐标
        max_x = max(max_x, x1, x2)  # 更新最大x坐标

    return min_x, max_x, first_line, last_line  # 返回第一条和最后一条直线


# 计算垂直距离的中点并绘制
def calculate_and_draw_vertical_midpoint(x1, x2, img, output_path):
    x3 = (x1 + x2) / 2
    height = img.shape[0]
    cv2.line(img, (int(x3), 0), (int(x3), height), (0, 0, 255), 5)  # 红色
    cv2.line(img, (int(x1), 0), (int(x1), height), (0, 0, 255), 5)  # 红色
    cv2.line(img, (int(x2), 0), (int(x2), height), (0, 0, 255), 5)  # 红色
    cv2.imwrite(output_path, img)
    return x3


# 计算水平距离的中点并绘制
def calculate_and_draw_horizontal_midpoint(y1, y2, img, output_path):
    y3 = (y1 + y2) / 2
    width = img.shape[1]
    cv2.line(img, (0, int(y3)), (width, int(y3)), (0, 0, 255), 10)  # 红色
    cv2.line(img, (0, int(y1)), (width, int(y1)), (0, 0, 255), 10)  # 红色
    cv2.line(img, (0, int(y2)), (width, int(y2)), (0, 0, 255), 10)  # 红色
    cv2.imwrite(output_path, img)
    return y3


def convert_filtered_buckets_to_fitted_lines(filtered_buckets):
    fitted_lines = []

    # 遍历 filtered_buckets 中的每个桶
    for angle, lines in filtered_buckets.items():
        # 将每个桶中的直线提取并添加到 fitted_lines 中
        fitted_lines.extend(lines)

    return fitted_lines


def si_qu_yu(img_path, fitted_lines, output_path):
    img = cv2.imread(img_path)  # 读取图像

    # 1. 按角度分配到桶中
    angle_buckets = assign_lines_to_angle_buckets(fitted_lines)

    # filtered_buckets = filter_lines_by_length(angle_buckets,0.3)

    # 2. 处理桶一
    y1, y2, first_line_1, last_line_1 = process_bucket_one(angle_buckets)

    # 3. 处理桶二
    x1, x2, first_line_2, last_line_2 = process_bucket_two(angle_buckets)

    fitted_lines = convert_filtered_buckets_to_fitted_lines(angle_buckets)

    # 4. 创建颜色列表
    colors = []
    for line in fitted_lines:
        # 桶一处理：45°到135°之间
        if line == first_line_1 or line == last_line_1:
            colors.append((255, 0, 0))  # 蓝色
        # 桶二处理：0°到45°和135°到180°
        elif line == first_line_2 or line == last_line_2:
            colors.append((255, 0, 0))  # 蓝色
        else:
            colors.append((0, 255, 0))  # 其他直线为绿色

    # 4. 绘制垂直线
    x3 = calculate_and_draw_vertical_midpoint(x1, x2, img, output_path)

    # 5. 绘制水平线
    y3 = calculate_and_draw_horizontal_midpoint(y1, y2, img, output_path)

    # 7. 绘制带颜色的直线
    draw_lines(img, fitted_lines, colors, output_path)

    return x3, y3


def classify_lines_in_buckets(angle_buckets, y3, x3):
    # 定义结果桶
    upper_bucket = []
    lower_bucket = []
    left_bucket = []
    right_bucket = []

    # 处理桶一：45°到135°之间的角度
    for angle in range(0, 45):
        if angle in angle_buckets:
            for line in angle_buckets[angle]:
                x1, y1, x2, y2 = line
                # 如果两个端点的 y 坐标都不超过 y3，归为上桶
                if y1 <= y3 and y2 <= y3:
                    upper_bucket.append(line)
                else:
                    lower_bucket.append(line)

    # 处理桶一：45°到135°之间的角度
    for angle in range(135, 180):
        if angle in angle_buckets:
            for line in angle_buckets[angle]:
                x1, y1, x2, y2 = line
                # 如果两个端点的 y 坐标都不超过 y3，归为上桶
                if y1 <= y3 and y2 <= y3:
                    upper_bucket.append(line)
                else:
                    lower_bucket.append(line)

    # 处理桶二：0°到45°和135°到180°之间的角度
    for angle in range(45, 136):
        if angle in angle_buckets:
            for line in angle_buckets[angle]:
                x1, y1, x2, y2 = line
                # 如果两个端点的 x 坐标都不超过 x3，归为左桶
                if x1 <= x3 and x2 <= x3:
                    left_bucket.append(line)
                else:
                    right_bucket.append(line)

    return upper_bucket, lower_bucket, left_bucket, right_bucket


# 长度过滤函数
def length_filter(lines, percentage=50):
    lines_sorted_by_length = sorted(lines, key=calculate_line_length, reverse=True)
    count_to_keep = int(len(lines_sorted_by_length) * (percentage / 100))
    return lines_sorted_by_length[:count_to_keep]


# 角度过滤函数
def angle_filter(lines, percentage=50):
    angles = [calculate_line_angle(line[0], line[1], line[2], line[3]) for line in lines]
    angle_freq = {}
    for angle in angles:
        angle_freq[angle] = angle_freq.get(angle, 0) + 1
    sorted_angles = sorted(angle_freq.items(), key=lambda x: x[1], reverse=True)
    angles_to_keep = [angle for angle, _ in sorted_angles[:int(len(sorted_angles) * (percentage / 100))]]
    filtered_lines = [line for line in lines if
                      calculate_line_angle(line[0], line[1], line[2], line[3]) in angles_to_keep]
    return filtered_lines


from collections import Counter


def filter_lines_cd(lines):
    if not lines:
        return []

    # 计算所有直线的长度
    lengths = [calculate_line_length(line) for line in lines]
    avg_length = np.mean(lengths)

    # 过滤：根据长度进行过滤
    filtered_lines = []
    for line in lines:
        length = calculate_line_length(line)

        # 判断条件：长度大于等于平均长度
        if length >= avg_length:
            filtered_lines.append(line)

    # 如果过滤后没有直线，则保留长度最大的直线
    if not filtered_lines:
        max_length = max(lengths)  # 获取最大长度
        # 保留所有最大长度的直线
        filtered_lines = [line for line in lines if calculate_line_length(line) == max_length]

    return filtered_lines


def filter_lines_jd_nm(lines):
    if not lines:
        return []

    # 计算所有直线的角度
    angles = [calculate_line_angle(x1, y1, x2, y2) for x1, y1, x2, y2 in lines]
    for i in range(len(angles)):
        if 135 <= angles[i] <= 180:
            angles[i] = abs(angles[i] - 180)  # 将135°到180°的角度转化到-45°到0°之间

    # 计算平均角度
    avg_angle = np.mean(angles)

    # 按平均角度过滤
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        angle = calculate_line_angle(x1, y1, x2, y2)
        if 135 <= angle <= 180:
            angle = abs(angle - 180)  # 转化角度范围
        if abs(angle - avg_angle) <= 2:  # 判断角度差异
            filtered_lines.append(line)

    # 如果按平均角度过滤后没有直线，则保留某个角度组最多的直线
    if not filtered_lines:
        # 统计每个角度的出现次数
        angle_counts = Counter(angles)
        most_common_angle, _ = angle_counts.most_common(1)[0]  # 获取出现次数最多的角度

        # 保留与最多角度相同的直线
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = calculate_line_angle(x1, y1, x2, y2)
            if 135 <= angle <= 180:
                angle = abs(angle - 180)  # 转化角度范围
            if angle == most_common_angle:
                filtered_lines.append(line)

    return filtered_lines


def filter_lines_jd(lines):
    if not lines:
        return []

    # 计算所有直线的角度
    angles = [calculate_line_angle(x1, y1, x2, y2) for x1, y1, x2, y2 in lines]

    avg_angle = np.mean(angles)

    # 按平均角度过滤
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        angle = calculate_line_angle(x1, y1, x2, y2)
        # 判断条件：角度与平均角度差异小于等于5°
        if abs(angle - avg_angle) <= 2:
            filtered_lines.append(line)

    # 如果按平均角度过滤后没有直线，则保留最多角度组的直线
    if not filtered_lines:
        # 统计每个角度的出现次数
        angle_counts = Counter(angles)
        most_common_angle, _ = angle_counts.most_common(1)[0]  # 获取出现次数最多的角度

        # 保留与最多角度相同的直线
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = calculate_line_angle(x1, y1, x2, y2)
            if angle == most_common_angle:
                filtered_lines.append(line)

    return filtered_lines


def process_bucket_sx(sx):
    # 对桶一中的直线按 x 截距从大到小排序
    lines_with_y_intercepts = []
    for line in sx:
        _, y_intercept = calculate_intercepts(line)
        if y_intercept is not None:
            lines_with_y_intercepts.append((line, y_intercept))
    lines_with_y_intercepts.sort(key=lambda x: x[1], reverse=True)

    # 找到排序后的第一条和最后一条直线
    first_line = lines_with_y_intercepts[0][0]
    last_line = lines_with_y_intercepts[-1][0]

    return first_line, last_line


def process_bucket_zy(zy):
    lines_with_x_intercepts = []
    for line in zy:
        x_intercept, _ = calculate_intercepts(line)
        if x_intercept is not None:
            lines_with_x_intercepts.append((line, x_intercept))
    lines_with_x_intercepts.sort(key=lambda x: x[1], reverse=True)

    # 找到排序后的第一条和最后一条直线
    first_line = lines_with_x_intercepts[0][0]
    last_line = lines_with_x_intercepts[-1][0]

    return first_line, last_line


def calculate_slope(x1, y1, x2, y2):
    if x2 == x1:  # 垂直线，斜率为无穷大
        return float('inf')
    return (y2 - y1) / (x2 - x1)


# 计算截距
def calculate_intercept(x1, y1, slope):
    return y1 - slope * x1


# 计算交点
def find_intersection(sx, zy):
    x1, y1, x2, y2 = sx
    x3, y3, x4, y4 = zy

    # 计算斜率
    m1 = calculate_slope(x1, y1, x2, y2)
    m2 = calculate_slope(x3, y3, x4, y4)

    # 如果两条直线平行或重合（斜率相同），则返回 None
    if m1 == m2:
        return None

    # 计算截距
    b1 = calculate_intercept(x1, y1, m1)
    b2 = calculate_intercept(x3, y3, m2)

    # 计算交点的 x 坐标
    x = (b2 - b1) / (m1 - m2)

    # 代入直线方程计算 y 坐标
    y = m1 * x + b1

    return (x, y)


# 计算 sx 中每一条直线与 zy 中每一条直线的交点
def calculate_all_intersections(sx_lines, zy_lines):
    intersections = []
    for sx in sx_lines:
        for zy in zy_lines:
            intersection = find_intersection(sx, zy)
            if intersection:  # 如果有交点，添加到结果列表
                intersections.append(intersection)
    return intersections


# 绘制直线的函数
def draw_lines_dt(img, lines, colors, thickness):
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = map(int, line)  # 获取线段起点和终点的整数坐标
        color = colors[i]  # 获取对应颜色
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)  # 绘制线段


# 绘制交点的函数
def draw_points(img, points, color, radius):
    for point in points:
        x, y = map(int, point)  # 获取点的坐标
        cv2.circle(img, (x, y), radius, color, -1)  # 绘制圆形标记交点


def process_bucket_sxay(sx):
    if not sx:  # 检查桶是否为空
        return None

    lines_with_y_intercepts = []
    for line in sx:
        _, y_intercept = calculate_intercepts(line)
        if y_intercept is not None:
            lines_with_y_intercepts.append((line, y_intercept))

    if not lines_with_y_intercepts:
        return None

    lines_with_y_intercepts.sort(key=lambda x: x[1], reverse=True)
    selected_line = lines_with_y_intercepts[0][0]
    sx.remove(selected_line)
    return selected_line


def process_bucket_sxiy(sx):
    if not sx:
        return None

    lines_with_y_intercepts = []
    for line in sx:
        _, y_intercept = calculate_intercepts(line)
        if y_intercept is not None:
            lines_with_y_intercepts.append((line, y_intercept))

    if not lines_with_y_intercepts:
        return None

    lines_with_y_intercepts.sort(key=lambda x: x[1], reverse=True)
    selected_line = lines_with_y_intercepts[-1][0]
    sx.remove(selected_line)
    return selected_line


def process_bucket_zyax(zy):
    if not zy:
        return None

    lines_with_x_intercepts = []
    for line in zy:
        x_intercept, _ = calculate_intercepts(line)
        if x_intercept is not None:
            lines_with_x_intercepts.append((line, x_intercept))

    if not lines_with_x_intercepts:
        return None

    lines_with_x_intercepts.sort(key=lambda x: x[1], reverse=True)
    selected_line = lines_with_x_intercepts[0][0]
    zy.remove(selected_line)
    return selected_line


def process_bucket_zyix(zy):
    if not zy:
        return None

    lines_with_x_intercepts = []
    for line in zy:
        x_intercept, _ = calculate_intercepts(line)
        if x_intercept is not None:
            lines_with_x_intercepts.append((line, x_intercept))

    if not lines_with_x_intercepts:
        return None

    lines_with_x_intercepts.sort(key=lambda x: x[1], reverse=True)
    selected_line = lines_with_x_intercepts[-1][0]
    zy.remove(selected_line)
    return selected_line


# 计算交点
def calculate_intersection(line1, line2):
    """计算两条直线的交点"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # 直线方程 Ax + By = C
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    # 计算交点
    determinant = A1 * B2 - A2 * B1
    if determinant == 0:
        return None  # 平行线没有交点

    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant
    return (x, y)


# 计算直线是否在图像内
def is_point_inside_image(x, y, width, height):
    """判断点是否在图像内"""
    return 0 <= x <= width and 0 <= y <= height


def update_lines(upper_bucket, lower_bucket, left_bucket, right_bucket, image_width, image_height):
    up = process_bucket_sxiy(upper_bucket)
    lower = process_bucket_sxay(lower_bucket)
    left = process_bucket_zyix(left_bucket)
    right = process_bucket_zyax(right_bucket)

    up_tf = lower_tf = left_tf = right_tf = True

    ur = calculate_intersection(up, right)
    rl = calculate_intersection(right, lower)
    ll = calculate_intersection(lower, left)
    lu = calculate_intersection(left, up)

    ur_in = ur and is_point_inside_image(ur[0], ur[1], image_width, image_height)
    rl_in = rl and is_point_inside_image(rl[0], rl[1], image_width, image_height)
    ll_in = ll and is_point_inside_image(ll[0], ll[1], image_width, image_height)
    lu_in = lu and is_point_inside_image(lu[0], lu[1], image_width, image_height)

    if not ur_in:
        up_tf = right_tf = False
    if not rl_in:
        right_tf = lower_tf = False
    if not ll_in:
        lower_tf = left_tf = False
    if not lu_in:
        left_tf = up_tf = False

    while not (up_tf and lower_tf and right_tf and left_tf):
        if not up_tf:
            up1 = up
            up = process_bucket_sxiy(upper_bucket)
            if up is None:
                up = up1
                break
            ur = calculate_intersection(up, right)
            lu = calculate_intersection(left, up)
            ur_in = ur and is_point_inside_image(ur[0], ur[1], image_width, image_height)
            lu_in = lu and is_point_inside_image(lu[0], lu[1], image_width, image_height)
            if ur_in and lu_in:
                up_tf = True

        if not lower_tf:
            lower1 = lower
            lower = process_bucket_sxay(lower_bucket)
            if lower is None:
                lower = lower1
                break
            rl = calculate_intersection(right, lower)
            ll = calculate_intersection(lower, left)
            rl_in = rl and is_point_inside_image(rl[0], rl[1], image_width, image_height)
            ll_in = ll and is_point_inside_image(ll[0], ll[1], image_width, image_height)
            if rl_in and ll_in:
                lower_tf = True

        if not left_tf:
            left1 = left
            left = process_bucket_zyix(left_bucket)
            if left is None:
                left = left1
                break
            ll = calculate_intersection(lower, left)
            lu = calculate_intersection(left, up)
            ll_in = ll and is_point_inside_image(ll[0], ll[1], image_width, image_height)
            lu_in = lu and is_point_inside_image(lu[0], lu[1], image_width, image_height)
            if ll_in and lu_in:
                left_tf = True

        if not right_tf:
            right1 = right
            right = process_bucket_zyax(right_bucket)
            if right is None:
                right = right1
                break
            lu = calculate_intersection(left, up)
            ur = calculate_intersection(up, right)
            lu_in = lu and is_point_inside_image(lu[0], lu[1], image_width, image_height)
            ur_in = ur and is_point_inside_image(ur[0], ur[1], image_width, image_height)
            if lu_in and ur_in:
                right_tf = True

    return up, lower, left, right, ur, rl, ll, lu


def move_lines_to_fit_image(up, lower, left, right, image_width, image_height):
    """
    根据交点是否在图像范围内，平移相关的直线直到交点在图像内。
    最终返回更新后的直线和它们计算出的交点。
    """
    # 计算交点
    ur = calculate_intersection(up, right)
    rl = calculate_intersection(right, lower)
    ll = calculate_intersection(lower, left)
    lu = calculate_intersection(left, up)

    # 检查交点是否在图像范围内
    def is_in_image_range(x, y):
        return 0 <= x <= image_width and 0 <= y <= image_height

    # 处理右上角交点（`ur`）
    if ur:
        ur_x, ur_y = ur
        if not is_in_image_range(ur_x, ur_y):
            # 平移 right 直线使得 x 坐标在范围内
            if ur_x < 0 or ur_x > image_width:
                right = move_right_line_to_fit_x(ur_x, right, image_width)
            # 平移 up 直线使得 y 坐标在范围内
            if ur_y < 0 or ur_y > image_height:
                up = move_up_line_to_fit_y(ur_y, up, image_height)

    # 处理右下角交点（`rl`）
    if rl:
        rl_x, rl_y = rl
        if not is_in_image_range(rl_x, rl_y):
            # 平移 right 直线使得 x 坐标在范围内
            if rl_x < 0 or rl_x > image_width:
                right = move_right_line_to_fit_x(rl_x, right, image_width)
            # 平移 lower 直线使得 y 坐标在范围内
            if rl_y < 0 or rl_y > image_height:
                lower = move_lower_line_to_fit_y(rl_y, lower, image_height)

    # 处理左下角交点（`ll`）
    if ll:
        ll_x, ll_y = ll
        if not is_in_image_range(ll_x, ll_y):
            # 平移 left 直线使得 x 坐标在范围内
            if ll_x < 0 or ll_x > image_width:
                left = move_left_line_to_fit_x(ll_x, left, image_width)
            # 平移 lower 直线使得 y 坐标在范围内
            if ll_y < 0 or ll_y > image_height:
                lower = move_lower_line_to_fit_y(ll_y, lower, image_height)

    # 处理左上角交点（`lu`）
    if lu:
        lu_x, lu_y = lu
        if not is_in_image_range(lu_x, lu_y):
            # 平移 left 直线使得 x 坐标在范围内
            if lu_x < 0 or lu_x > image_width:
                left = move_left_line_to_fit_x(lu_x, left, image_width)
            # 平移 up 直线使得 y 坐标在范围内
            if lu_y < 0 or lu_y > image_height:
                up = move_up_line_to_fit_y(lu_y, up, image_height)

    # 计算更新后的交点
    ur_updated = calculate_intersection(up, right)
    rl_updated = calculate_intersection(right, lower)
    ll_updated = calculate_intersection(lower, left)
    lu_updated = calculate_intersection(left, up)

    # 返回更新后的直线和交点
    return up, lower, left, right, ur_updated, rl_updated, ll_updated, lu_updated


def move_right_line_to_fit_x(x, right, image_width):
    """
    将 `right` 直线平移到 x 坐标在图像范围内。
    """
    # 假设 x 是交点的 x 坐标，计算需要移动的距离
    move_distance = image_width - x if x > image_width else -x
    # 平移 right 直线
    return [coord + move_distance if i % 2 == 0 else coord for i, coord in enumerate(right)]


def move_up_line_to_fit_y(y, up, image_height):
    """
    将 `up` 直线平移到 y 坐标在图像范围内。
    """
    # 假设 y 是交点的 y 坐标，计算需要移动的距离
    move_distance = image_height - y if y > image_height else -y
    # 平移 up 直线
    return [coord + move_distance if i % 2 == 1 else coord for i, coord in enumerate(up)]


def move_lower_line_to_fit_y(y, lower, image_height):
    """
    将 `lower` 直线平移到 y 坐标在图像范围内。
    """
    # 假设 y 是交点的 y 坐标，计算需要移动的距离
    move_distance = image_height - y if y > image_height else -y
    # 平移 lower 直线
    return [coord + move_distance if i % 2 == 1 else coord for i, coord in enumerate(lower)]


def move_left_line_to_fit_x(x, left, image_width):
    """
    将 `left` 直线平移到 x 坐标在图像范围内。
    """
    # 假设 x 是交点的 x 坐标，计算需要移动的距离
    move_distance = image_width - x if x > image_width else -x
    # 平移 left 直线
    return [coord + move_distance if i % 2 == 0 else coord for i, coord in enumerate(left)]


def not_four(si_qu_yu_l_filter):
    # 1. 按角度分配到桶中
    angle_buckets = assign_lines_to_angle_buckets(si_qu_yu_l_filter)

    # 2. 处理桶一
    y1, y2, first_line_1, last_line_1 = process_bucket_one(angle_buckets)

    # 3. 处理桶二
    x1, x2, first_line_2, last_line_2 = process_bucket_two(angle_buckets)

    up, lower, left, right = last_line_2, first_line_2, first_line_1, last_line_1

    ur = calculate_intersection(up, right)
    rl = calculate_intersection(right, lower)
    ll = calculate_intersection(lower, left)
    lu = calculate_intersection(left, up)

    return up, lower, left, right, ur, rl, ll, lu


def group_lines_by_angle(fitted_lines, bucket_size=10):
    # 创建一个字典，键为角度范围的桶，值为直线列表
    angle_buckets = defaultdict(list)

    for line in fitted_lines:
        x1, y1, x2, y2 = line
        angle = calculate_line_angle(x1, y1, x2, y2)

        # 计算桶的索引，根据自定义的bucket_size
        bucket_index = int(angle // bucket_size)

        # 防止角度超出最大值（180°）
        if bucket_index >= 180 // bucket_size:
            bucket_index = (180 // bucket_size) - 1

        # 将直线添加到对应的桶中
        angle_buckets[bucket_index].append(line)

    return angle_buckets


def plot_angle_histogram_from_buckets(angle_buckets, bucket_size=10):
    # 获取桶的数量（即桶的索引）
    bucket_indices = list(angle_buckets.keys())

    # 获取每个桶中的直线数量
    bucket_counts = [len(angle_buckets[index]) for index in bucket_indices]

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.bar(bucket_indices, bucket_counts, width=0.8, align='center', color='skyblue', edgecolor='black')

    # 设置X轴标签为桶的角度范围
    x_ticks = [f'{i * bucket_size}°-{(i + 1) * bucket_size}°' for i in bucket_indices]
    plt.xticks(bucket_indices, x_ticks, rotation=45)

    # 设置图表的标题和标签
    plt.xlabel('Angle Buckets')
    plt.ylabel('Number of Lines')
    plt.title(f'Line Distribution Across {bucket_size}° Angle Buckets')

    # 显示图表
    plt.tight_layout()
    plt.show()


def get_lines_from_top_bucket(angle_buckets, bucket_size=10):
    # 计算每个桶的频数，并按频数从高到低排序
    bucket_counts = {bucket: len(lines) for bucket, lines in angle_buckets.items()}
    sorted_buckets = sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True)

    # 获取频数最高的第一个桶
    top_bucket = sorted_buckets[0][0]  # 获取频数最高的桶的角度范围

    # 提取频数最高桶中的所有直线
    lines_from_top_bucket = angle_buckets[top_bucket]

    # print(f"Retrieved lines from the top bucket with the highest frequency: Bucket {top_bucket}.")

    return lines_from_top_bucket


def process_lines_and_get_top_half(fitted_lines, bucket_size=10):
    # 1. 将直线分配到角度桶中
    angle_buckets = group_lines_by_angle(fitted_lines, bucket_size)

    # 2. 绘制角度直方图
    # plot_angle_histogram_from_buckets(angle_buckets, bucket_size)

    # 3. 获取频数前50%高的桶中的直线
    top_half_lines = get_lines_from_top_bucket(angle_buckets, bucket_size)

    # angle_buckets = group_lines_by_angle(top_half_lines, bucket_size)

    # 2. 绘制角度直方图
    # plot_angle_histogram_from_buckets(angle_buckets, bucket_size)

    return top_half_lines


# ------------------三倍方差------------------------
def filter_lines_by_angle_variance(sx):
    """
    处理给定的直线坐标列表，计算每条直线的角度，过滤异常值。

    :param sx: 一个二维列表，包含直线的端点坐标。例如：[ [x1, y1, x2, y2], [x1, y1, x2, y2], ... ]
    :return: 返回去除异常值后的直线坐标列表。
    """
    # 计算所有直线的角度
    angles = []
    for line in sx:
        x1, y1, x2, y2 = line
        angle = calculate_line_angle(x1, y1, x2, y2)

        # 处理角度范围，135度到180度的角度减去180度
        if 135 <= angle <= 180:
            angle -= 180
        angles.append(angle)

    # 计算角度的均值和标准方差
    mean_angle = np.mean(angles)
    std_dev = np.std(angles)

    # 计算阈值：3倍标准方差
    threshold = 2 * std_dev

    # 过滤掉角度差异超过3倍标准方差的直线
    filtered_lines = []
    for i, line in enumerate(sx):
        angle = angles[i]
        if abs(angle - mean_angle) <= threshold:
            filtered_lines.append(line)

    return filtered_lines


def main():
    # input_folder = r"E:\xmsb\GLNet\Perspective Correction\duogelimian"
    # input_folder = r"E:\Experiments\GLNet\Perspective Correction\angle_adjust_train_ori"
    input_folder = r"./test/Construction/origin"

    output_folder1 = "./test/Construction/example_point/lines_same"
    output_folder2 = "./test/Construction/example_point/lines_diff"
    output_folder4 = "./test/Construction/example_point/lines_class"
    output_folder5 = "./test/Construction/example_point/len_filter1"
    output_folder6 = "./test/Construction/example_point/lines_fit"
    output_folder8 = "./test/Construction/example_point/regions"
    output_folder9 = "./test/Construction/example_point/ang_filter1+fit_regions"
    output_folder10 = "./test/Construction/example_point/len_filter2"
    output_folder11 = "./test/Construction/example_point/ang_filter2"
    output_folder12 = "./test/Construction/example_point/bl"
    output_folder13 = "./test/Construction/example_point/four_pts"
    output_folder14 = "./test/Construction/example_point/point"

    # 输出文件夹
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    os.makedirs(output_folder4, exist_ok=True)
    os.makedirs(output_folder5, exist_ok=True)
    os.makedirs(output_folder6, exist_ok=True)
    os.makedirs(output_folder8, exist_ok=True)
    os.makedirs(output_folder9, exist_ok=True)
    os.makedirs(output_folder10, exist_ok=True)
    os.makedirs(output_folder11, exist_ok=True)
    os.makedirs(output_folder12, exist_ok=True)
    os.makedirs(output_folder13, exist_ok=True)
    os.makedirs(output_folder14, exist_ok=True)

    # 获取所有图像路径
    # image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))
    if not image_paths:
        print("未找到任何图像文件。")
        return

    # 用于记录无法处理的图像路径
    failed_images = []

    for img_path in image_paths:
        try:
            # 获取文件名以生成对应的输出文件名
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            # 各个阶段图片输出路径
            output_path1 = os.path.join(output_folder1, f"{file_name}_lines_same.jpg")
            output_path2 = os.path.join(output_folder2, f"{file_name}_lines_diff.jpg")
            output_path4 = os.path.join(output_folder4, f"{file_name}_lines_class.jpg")
            output_path5 = os.path.join(output_folder5, f"{file_name}_len_filter1.jpg")
            output_path6 = os.path.join(output_folder6, f"{file_name}_lines_fit.jpg")
            output_path8 = os.path.join(output_folder8, f"{file_name}_regions.jpg")
            output_path9 = os.path.join(output_folder9, f"{file_name}_fit_regions.jpg")
            output_path10 = os.path.join(output_folder10, f"{file_name}_len_filter2.jpg")
            output_path11 = os.path.join(output_folder11, f"{file_name}_ang_filter2.jpg")
            # output_path12 = os.path.join(output_folder12, f"{file_name}bl_lines.jpg")
            output_path13 = os.path.join(output_folder13, f"{file_name}four_pts.jpg")
            output_path14 = os.path.join(output_folder14, f"{file_name}.txt")

            print(f"处理图像: {img_path}")
            # LSD检测直线
            length_threshold = 40  # 过滤掉长度小于 40 像素的直线段
            img, lines = detect_and_draw_lines(img_path, output_path1, output_path2, length_threshold)
            if img is None or lines is None:
                print(f"检测或过滤直线失败: {img_path}")
                continue

            # 直线分类：角度+截距
            categories = classify_lines(lines)
            print(f"分类后共有 {len(categories)} 类直线。")
            draw_and_save_categories(img, categories, output_path4)
            # 过滤
            sum_lengths_categories = process_categories(categories, percentage=0.2)
            draw_and_save_categories(img, sum_lengths_categories, output_path5)
            # print(f"长度之和过滤后共有 {len(sum_lengths_categories)} 类直线。")

            # 直线拟合：最小二乘法
            fitted_lines = fit_and_draw_fitted_lines(img, sum_lengths_categories, output_path6)
            fitted_lines = convert_fitted_lines_to_lines(fitted_lines)
            draw_lines(img, fitted_lines, generate_unique_colors(len(fitted_lines)), output_path6)

            # 划分四区域
            x, y = si_qu_yu(img_path, fitted_lines, output_path8)
            angle_buckets = assign_lines_to_angle_buckets(fitted_lines)
            upper_bucket, lower_bucket, left_bucket, right_bucket = classify_lines_in_buckets(angle_buckets, y, x)

            sx = upper_bucket + lower_bucket
            zy = left_bucket + right_bucket
            # print("--------------")
            # print(len(sx + zy))
            sx = filter_lines_by_angle_variance(sx)
            zy = filter_lines_by_angle_variance(zy)
            # print("--------------")
            # print(len(sx + zy))
            fitted_lines = sx + zy
            angle_buckets = assign_lines_to_angle_buckets(fitted_lines)

            upper_bucket, lower_bucket, left_bucket, right_bucket = classify_lines_in_buckets(angle_buckets, y, x)
            si_qu_yu_l_filter = upper_bucket + lower_bucket + left_bucket + right_bucket
            si_qu_yu_l = upper_bucket + lower_bucket + left_bucket + right_bucket
            upper_lines_color = [(0, 0, 255)] * len(upper_bucket)
            lower_lines_color = [(255, 0, 0)] * len(lower_bucket)
            left_lines_color = [(0, 255, 0)] * len(left_bucket)
            right_lines_color = [(0, 255, 255)] * len(right_bucket)
            all_colors = upper_lines_color + lower_lines_color + left_lines_color + right_lines_color
            draw_lines(img, si_qu_yu_l, all_colors, output_path9)

            # 寻找边界直线，计算交点
            if not (len(upper_bucket) and len(lower_bucket) and len(left_bucket) and len(right_bucket)):
                print(f"初次检测到的直线数: {len(lines)}")
                up, lower, left, right, ur_updated, rl_updated, ll_updated, lu_updated = not_four(si_qu_yu_l_filter)
            else:
                upper_bucket = filter_lines_cd(upper_bucket)
                lower_bucket = filter_lines_cd(lower_bucket)
                left_bucket = filter_lines_cd(left_bucket)
                right_bucket = filter_lines_cd(right_bucket)
                upper_lines_color = [(0, 0, 255)] * len(upper_bucket)
                lower_lines_color = [(255, 0, 0)] * len(lower_bucket)
                left_lines_color = [(0, 255, 0)] * len(left_bucket)
                right_lines_color = [(0, 255, 255)] * len(right_bucket)
                si_qu_yu_l_filter = upper_bucket + lower_bucket + left_bucket + right_bucket
                all_colors = upper_lines_color + lower_lines_color + left_lines_color + right_lines_color
                # draw_lines(img, si_qu_yu_l_filter, all_colors, output_path10)

                upper_bucket = filter_lines_jd_nm(upper_bucket)
                lower_bucket = filter_lines_jd_nm(lower_bucket)
                left_bucket = filter_lines_jd(left_bucket)
                right_bucket = filter_lines_jd(right_bucket)
                upper_lines_color = [(0, 0, 255)] * len(upper_bucket)
                lower_lines_color = [(255, 0, 0)] * len(lower_bucket)
                left_lines_color = [(0, 255, 0)] * len(left_bucket)
                right_lines_color = [(0, 255, 255)] * len(right_bucket)
                si_qu_yu_l_filter = upper_bucket + lower_bucket + left_bucket + right_bucket
                print(f"最终的直线数: {len(si_qu_yu_l_filter)}")
                all_colors = upper_lines_color + lower_lines_color + left_lines_color + right_lines_color
                draw_lines(img, si_qu_yu_l_filter, all_colors, output_path11)

                # 只获取图像的尺寸，不加载图像内容
                image = Image.open(img_path)
                image = ImageOps.exif_transpose(image)
                image_width, image_height = image.size

                up, lower, left, right, ur, rl, ll, lu = update_lines(upper_bucket, lower_bucket, left_bucket,
                                                                      right_bucket,
                                                                      image_width, image_height)

                up, lower, left, right, ur_updated, rl_updated, ll_updated, lu_updated = move_lines_to_fit_image(up,
                                                                                                                 lower,
                                                                                                                 left,
                                                                                                                 right,
                                                                                                                 image_width,
                                                                                                                 image_height)

            combined_lines = [up, lower, left, right]
            # combined_intersections = [ur_updated, rl_updated, ll_updated, lu_updated]
            combined_intersections = [lu_updated, ll_updated, rl_updated, ur_updated]
            # print(combined_intersections)
            # print(combined_lines)

            # 绘制 zy 直线（蓝色，粗10px）
            zy_colors = [(0, 0, 255)] * len(combined_lines)  # 蓝色
            draw_lines_dt(img, combined_lines, zy_colors, thickness=20)

            # 绘制 si_qu_yu_l_filter 直线（绿色，粗5px）
            si_qu_yu_l_filter_colors = [(0, 255, 0)] * len(si_qu_yu_l_filter)  # 绿色
            draw_lines_dt(img, si_qu_yu_l_filter, si_qu_yu_l_filter_colors, thickness=3)

            # 绘制 intersections 交点（黄色，粗10px）
            draw_points(img, combined_intersections, color=(0, 255, 255), radius=40)

            cv2.imwrite(output_path13, img)  # 保存绘制后的图像

            img_resized = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
            # cv2.imshow("Result", img_resized)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            np.savetxt(output_path14, combined_intersections, fmt='%f %f')

        except Exception as e:
            print(f"处理图像 {img_path} 时发生错误: {e}")
            failed_images.append(img_path)  # 记录无法处理的图像
            continue  # 发生异常时跳过当前图像，继续处理下一个图像

    if failed_images:
        print("\n以下图像无法处理：")
        for failed_img in failed_images:
            print(failed_img)


if __name__ == "__main__":
    main()
