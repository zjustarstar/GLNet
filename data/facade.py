import torch
from PIL import Image
import numpy as np
from utils.base_tools import get_name
import json
from sklearn.model_selection import train_test_split
import cv2


ignore_label = 255
num_classes = 7
colors = {
    0: [0, 0, 0],      # 背景
    1: [70, 70, 70],    # building
    2: [0, 0, 142],    # car
    3: [153, 153, 153],  # door
    4: [70, 130, 180],    # sky
    5: [107, 142, 35],  # tree
    6: [250, 170, 30]   # window
}


class PolygonTrans():
    def __init__(self):
        # 更新类别名称及对应的编号
        self.binary = {
            "building": 1,
            "car": 2,
            "door": 3,
            "sky": 4,
            "tree": 5,  # 原来的 tree
            "window": 6
        }
        # 覆盖顺序
        self.overlap_order = ["sky", "building", "door", "window", "tree", "car"]

    def polygon2mask(self, img_size, polygons, rectangles):
        mask = np.zeros(img_size, dtype=np.uint8)
        for cat in self.overlap_order:
            polygon = polygons[cat]
            cv2.fillPoly(mask, polygon, color=self.binary[cat])
            rectangle = rectangles[cat]
            for ret in rectangle:
                x1, y1 = ret[0]
                x2, y2 = ret[1]
                mask[y1:y2, x1:x2] = self.binary[cat]
        return mask

    # translate label_id to color img
    def id2trainId(self, label, select=None):
        w, h = label.shape
        label_copy = np.zeros((w, h, 3), dtype=np.uint8)
        for index, color in colors.items():
            if select is not None:
                if index == select:
                    label_copy[label == index] = color
                else:
                    continue
            else:
                label_copy[label == index] = color
        return label_copy.astype(np.uint8)


def read_json(file_name):
    # 按新类别调整
    record = {"building": [], "car": [], "door": [], "sky": [], "tree": [], "window": []}
    record_rectangle = {"building": [], "car": [], "door": [], "sky": [], "tree": [], "window": []}

    with open(file_name, "r") as load_polygon:
        data = json.load(load_polygon)

    data = data["shapes"]
    for item in data:
        label = item["label"]
        points = item["points"]
        shape = item["shape_type"]
        if label not in record:
            continue

        if shape == "rectangle":
            record_rectangle[label].append(np.array(points, dtype=np.int32))
        else:
            record[label].append(np.array(points, dtype=np.int32))

    return record, record_rectangle


def prepare_facade_data(args):
    roots = args.root + "classall/translated_data_train_hznu_zhengmian_V2/"
    # roots_extra = args.root + "classall/translated_data_c6extra/"
    root_test = args.root + "classall/translated_data_test_hznu_zhengmian_V2/"

    items = get_name(roots + "images", mode_folder=False)
    # items_extra = get_name(roots_extra + "images", mode_folder=False)
    items_test = get_name(root_test + "images", mode_folder=False)

    record = []
    for item in items:
        record.append([roots + "images/" + item, roots + "binary_mask/" + item])
    # record_extra = []
    # for items_extra1 in items_extra:
    #     record_extra.append([roots_extra + "images/" + items_extra1, roots_extra + "binary_mask/" + items_extra1])
    record_test = []
    for item1 in items_test:
        record_test.append([root_test + "images/" + item1, root_test + "binary_mask/" + item1])

    # 直接划分训练集和验证集
    train, val = train_test_split(record, train_size=0.9, random_state=1)
    # train.extend(record_extra)
    # val, test = train_test_split(other, train_size=0.95, random_state=1)
    test = []
    test = record_test

    # 提取图像文件名
    image_names = [item[0].split("/")[-1] for item in test]
    image_names_train = [item1[0].split("/")[-1] for item1 in train]
    image_names_val = [item2[0].split("/")[-1] for item2 in val]

    # 使用 join 方法将所有文件名拼接为一个字符串，并用换行符分隔
    output_file = "test_image_names_c6.txt"
    output_file_train = "train_image_names_c6_extra.txt"
    output_file_val = "val_image_names_c6_extra.txt"

    # with open(output_file, "w") as f:
    #     f.write("\n".join(image_names))
    # with open(output_file_train, "w") as f:
    #     f.write("\n".join(image_names_train))
    # with open(output_file_val, "w") as f:
    #     f.write("\n".join(image_names_val))

    # print(f"图像文件名已保存到 {output_file}")
    # print(f"图像文件名已保存到 {output_file_train}")
    # print(f"图像文件名已保存到 {output_file_val}")

    return {"train": train, "val": val, "test": test}


class Facade(torch.utils.data.Dataset):
    def __init__(self, args, mode, joint_transform=None, standard_transform=None):
        self.args = args
        self.imgs = prepare_facade_data(args)[mode]
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.joint_transform = joint_transform
        self.standard_transform = standard_transform
        if self.args.use_ignore:
            self.id_to_trainid = {6: 255, 7: 255, 8: 255, 9: 255}

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        if self.args.use_ignore:
            for k, v in self.id_to_trainid.items():
                mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.standard_transform is not None:
            img = self.standard_transform(img)

        return {"images": img, "masks": torch.from_numpy(np.array(mask, dtype=np.int32)).long()}

    def __len__(self):
        return len(self.imgs)