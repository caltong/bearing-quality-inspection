import cv2
import json
import numpy as np
from PIL import Image
from utils import get_radius_center
from time import time
import copy
import random


def generate_new_data(image_path, labels_path):
    image_cv2 = cv2.imread(image_path)  # cv2 load image BGR image has 3 channels
    with open(labels_path) as f:  # json load labels
        json_labels = json.load(f)
    # 解析labels
    labels = []  # 损伤位置组成的list [np.array,...]
    for i in json_labels['shapes']:
        labels.append(np.array(i['points'], np.int32))

    # 获取工件中心坐标和半径
    center, radius = get_radius_center(Image.fromarray(image_cv2))
    left_up_corner = np.array(center) - radius

    # 获取原图裁切目标图 带有损伤信息 cv2 crop [y:y+h,x:x+h]
    origin_crop_image = image_cv2[center[1] - radius:center[1] + radius,
                        center[0] - radius:center[0] + radius]

    # 根据labels创建mask遮罩
    mask = np.zeros((origin_crop_image.shape[0], origin_crop_image.shape[1]), np.uint8)
    # 根据圆心坐标移动labels坐标 labels [np.array,...]
    for i in range(len(labels)):
        labels[i] = labels[i] - np.repeat([left_up_corner], len(labels[i]), axis=0)
        cv2.fillPoly(mask, [labels[i]], 255)
    # inpaint
    background = cv2.inpaint(origin_crop_image, mask, 3, cv2.INPAINT_NS)

    # 分别调整每个损伤
    new_image = copy.deepcopy(background)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    origin_copy = copy.deepcopy(origin_crop_image)
    origin_copy = cv2.cvtColor(origin_copy, cv2.COLOR_BGR2GRAY)
    for i in labels:
        # 只要损伤区域不要其他区域
        single_mask = np.zeros(new_image.shape, np.uint8)
        single_mask = cv2.fillPoly(single_mask, [i], 1)
        single_mask = np.multiply(single_mask, origin_copy)

        # 获取随机移动值
        xc = radius
        yc = radius
        x0 = (np.min(i[:, 0]) + np.max(i[:, 0])) / 2
        y0 = (np.min(i[:, 1]) + np.max(i[:, 1])) / 2
        k = (yc - y0) / (xc - x0)
        b = yc - k * xc
        delta_x = int((random.random() - 0.5) * 50)
        delta_y = int(k * (x0 + delta_x) + b - y0)
        if random.randint(0, 1):
            # 移动损伤位置
            (mx, my) = np.meshgrid(np.arange(single_mask.shape[1]), np.arange(single_mask.shape[0]))
            ox = (mx - delta_x).astype(np.float32)
            oy = (my - delta_y).astype(np.float32)
            single_mask = cv2.remap(single_mask, ox, oy, cv2.INTER_LINEAR)
        else:
            delta_x, delta_y = 0, 0
        # 对应的改变背景留黑位置
        adjust_i = i + np.array([delta_x, delta_y])

        # 只要其他区域不要损伤区域
        background_mask = np.ones(new_image.shape, np.uint8)
        background_mask = cv2.fillPoly(background_mask, [adjust_i], 0)
        background_mask = np.multiply(background_mask, new_image)
        new_image = np.add(single_mask, background_mask)

    new_image = Image.fromarray(new_image)
    return new_image


if __name__ == "__main__":
    tic = time()
    new_image = generate_new_data("inpaint-sample/12060_942.bmp", "inpaint-sample/12060_942.json")
    toc = time()
    print(toc - tic)
    new_image.show()
