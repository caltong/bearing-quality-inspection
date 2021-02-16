import cv2
import json
import numpy as np
from PIL import Image
from utils import get_radius_center, add_black_background
from time import time
import copy
import random
import os
import math


def generate_new_data_v2(image_path, labels_path):
    name_ori = os.path.split(image_path)[-1]
    image_ori = cv2.imread(os.path.join("test", name_ori))
    label_ori = get_label(name_ori.split('.')[0] + '.json')
    # Image.fromarray(image_ori).show()
    # print(label_ori)
    # background = get_background(image_ori, label_ori)
    # background.show()
    # Image.fromarray(image_ori).show()
    background = get_random_good_as_background()
    # background.show()
    # 损伤区域小图
    target_list = []
    for label in label_ori:
        target = get_target(image_ori, label)
        target = random_shape_target(target)
        target_list.append(target)
        # target.show()
    # print(target_list)
    image_new = copy.deepcopy(background)
    for target in target_list:
        image_new = background_add_target(image_new, target)

    # image_new.show()
    return image_new


def get_image(image_name):
    root_path = os.path.join('data_all', 'side', 'train', 'bad')
    images = os.listdir(root_path)
    if image_name in images:
        return cv2.imread(os.path.join(root_path, image_name))
    else:
        return None


def get_label(label_name):
    root_path = 'test'
    labels = os.listdir(root_path)
    label_path = os.path.join(root_path, label_name)
    if label_name in labels:
        with open(label_path) as f:  # json load labels
            json_labels = json.load(f)
            # 解析labels
            label = []  # 损伤位置组成的list [np.array,...]
            for i in json_labels['shapes']:
                label.append(np.array(i['points'], np.int32))
        return label
    else:
        return None


def get_target(image, label):
    # 获取单个损伤区域 返回PIL Image
    mask = np.ones(image.shape, np.uint8)
    mask.fill(255)
    roi_coner = label
    cv2.fillPoly(mask, [roi_coner], 0)
    masked_image = cv2.bitwise_or(image, mask)
    # target_0 = Image.fromarray(masked_image)
    max = np.max(label, axis=0)
    min = np.min(label, axis=0)
    target = masked_image[min[1]:max[1], min[0]:max[0]]  # opencv 先y坐标后x坐标
    return Image.fromarray(target)


def get_background(image, label):
    # 返回裁切后的正方形工件背景
    # 获取工件中心坐标和半径
    center, radius = get_radius_center(Image.fromarray(image))
    left_up_corner = np.array(center) - radius

    # 获取原图裁切目标图 带有损伤信息 cv2 crop [y:y+h,x:x+h]
    origin_crop_image = image[center[1] - radius:center[1] + radius,
                        center[0] - radius:center[0] + radius]

    # 根据labels创建mask遮罩
    mask = np.zeros((origin_crop_image.shape[0], origin_crop_image.shape[1]), np.uint8)
    # 根据圆心坐标移动labels坐标 labels [np.array,...]
    for i in range(len(label)):
        label[i] = label[i] - np.repeat([left_up_corner], len(label[i]), axis=0)
        cv2.fillPoly(mask, [label[i]], 255)
    # inpaint
    # Image.fromarray(mask).show()
    background = cv2.inpaint(origin_crop_image, mask, 3, cv2.INPAINT_NS)
    return Image.fromarray(background)


def get_random_good_as_background():
    root_path = os.path.join('data_all', 'side', 'train', 'good')
    images = os.listdir(root_path)
    random_image = random.choice(images)
    random_image = cv2.imread(os.path.join(root_path, random_image))
    # 获取工件中心坐标和半径
    center, radius = get_radius_center(Image.fromarray(random_image))
    # 获取原图裁切目标图 带有损伤信息 cv2 crop [y:y+h,x:x+h]
    random_image = random_image[center[1] - radius:center[1] + radius,
                   center[0] - radius:center[0] + radius]
    return Image.fromarray(random_image)


def random_shape_target(target):
    target = target.resize((int((random.random() + 0.5) * target.size[0]),
                            int((random.random() + 0.5) * target.size[1])))
    if random.random() > 0.5:
        target = target.transpose(Image.FLIP_LEFT_RIGHT)
    target = target.rotate(random.random() * 360, fillcolor=(255, 255, 255))
    return target


def background_add_target(backgroud, target):
    backgroud = cv2.cvtColor(np.asarray(backgroud), cv2.COLOR_RGB2BGR)
    target = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)
    # 二值化
    ret, mask = cv2.threshold(target, 150, 255, cv2.THRESH_BINARY)
    # 反向
    mask_inv = cv2.bitwise_not(mask)
    # 随机角度
    random_theta = random.random() * 360
    # 随机中心距
    random_r = random.randrange(300, 400)
    # padding后边长
    side = backgroud.shape[0]
    # 通过padding 移动损伤target
    # 保证padding后和background尺寸一致
    padding = ((int(side / 2 - random_r * math.sin(random_theta / 180 * math.pi) - mask_inv.shape[0] / 2),
                side - mask_inv.shape[0] - int(
                    side / 2 - random_r * math.sin(random_theta / 180 * math.pi) - mask_inv.shape[0] / 2)),
               (int(side / 2 + random_r * math.cos(random_theta / 180 * math.pi) - mask_inv.shape[1] / 2),
                side - mask_inv.shape[1] - int(
                    side / 2 + random_r * math.cos(random_theta / 180 * math.pi) - mask_inv.shape[1] / 2)))
    # 出现负数的情况
    if padding[0][0] < 0:
        padding = ((0, side - mask_inv.shape[0]), (padding[1][0], padding[1][1]))
    if padding[0][1] < 0:
        padding = ((side - mask_inv.shape[0], 0), (padding[1][0], padding[1][1]))
    if padding[1][0] < 0:
        padding = ((padding[0][0], padding[0][1]), (0, side - mask_inv.shape[1]))
    if padding[1][1] < 0:
        padding = ((padding[0][0], padding[0][1]), (side - mask_inv.shape[1], 0))
    padding_3d = (padding[0], padding[1], (0, 0))
    mask_inv = np.pad(mask_inv, padding_3d, "constant")
    target = np.pad(target, padding_3d, "constant", constant_values=255)
    backgroud = cv2.bitwise_and(backgroud, backgroud, mask=cv2.bitwise_not(mask_inv[:, :, 0]))
    target = cv2.bitwise_and(target, target, mask=mask_inv[:, :, 0])
    backgroud = cv2.add(backgroud, target)

    return Image.fromarray(backgroud)


if __name__ == '__main__':
    image_new = generate_new_data_v2("test/01130_124.jpg", "test/01130_1.json")
    image_new.show()
    # add_black_background(image_new).show()
    # print(get_image("01130_1.jpg"))
    # print(get_label("01130_1.json"))
