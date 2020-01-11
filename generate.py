import cv2
import json
import numpy as np
from PIL import Image
from utils import get_radius_center


def generate_new_data(image_path, labels_path):
    image_cv2 = cv2.imread(image_path)  # cv2 load image
    with open(labels_path) as f:  # json load labels
        json_labels = json.load(f)

    # 解析labels
    labels = []  # 损伤位置组成的list
    for i in json_labels['shapes']:
        labels.append(i['points'])

    # 根据labels创建mask遮罩
    mask = np.zeros((image_cv2.shape[0], image_cv2.shape[1]), np.uint8)
    for i in labels:
        cv2.fillPoly(mask, [np.array(i, np.int32)], (255, 255, 255))
    # inpaint
    background = cv2.inpaint(image_cv2, mask, 3, cv2.INPAINT_NS)

    # 获取工件中心坐标和半径
    center, radius = get_radius_center(Image.fromarray(background))
    # 获取损伤目标小图
    for i in labels:
        np_label = np.array(i, np.int32)
        np_label
    new_image = Image.fromarray(background)
    return new_image


if __name__ == "__main__":
    new_image = generate_new_data("inpaint-sample/12060_942.bmp", "inpaint-sample/12060_942.json")
    new_image.show()
