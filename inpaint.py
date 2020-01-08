import numpy as np
from matplotlib import pyplot as plt
import cv2
import json
from utils import get_radius_center
from PIL import Image

img_id = 'inpaint-sample/12060_942'
img = cv2.imread(img_id + '.bmp')
with open(img_id + '.json') as f:
    json_labels = json.load(f)


def get_background(img, label):
    # img: open_cv image
    # label raw json file
    labels = []
    for i in json_labels['shapes']:
        labels.append(i['points'])
    # 创建mask
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in labels:
        cv2.fillPoly(mask, [np.array(i, np.int32)], (255, 0, 0))
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)  # 生成background

    # 裁切损伤位置小图
    targets = []
    for i in labels:
        mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        cv2.fillPoly(mask, [np.array(i, np.int32)], 1)
        target = np.stack((mask,) * 3, -1) * img
        np_label = np.array(i)
        top = np.min(np_label[:, 1])
        right = np.max(np_label[:, 0])
        down = np.max(np_label[:, 1])
        left = np.min(np_label[:, 0])
        targets.append(target[int(top):int(down), int(left):int(right), :])
    return result, targets


result, targets = get_background(img, json_labels)
result = Image.fromarray(result)
center, radius = get_radius_center(result)
center_x = center[0]
center_y = center[1]
center_r = radius
half_width = int(center_r * 1.01)  # height = width
result = result.crop((int(center_x - half_width), int(center_y - half_width),
                      int(center_x + half_width), int(center_y + half_width)))

result.show()
print(result.size)
print(center, radius)
