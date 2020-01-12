import cv2
import json
import numpy as np
from PIL import Image
from utils import get_radius_center
from time import time


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
    left_up_corner = np.array(center) - radius
    # 获取损伤目标小图
    new_labels = []
    for i in labels:
        a_np_label = np.array(i, np.int32)
        plus_len = a_np_label.shape[0]
        plus = np.repeat([left_up_corner], plus_len, axis=0)
        a_np_label = a_np_label - plus
        new_labels.append(a_np_label)

    # 获取原图裁切目标图 带有损伤信息
    origin_crop_image = image_cv2[center[1] - radius:center[1] + radius,
                        center[0] - radius:center[0] + radius]



    # crop (left, top, right, bottom)
    new_image = Image.fromarray(background).crop((center[0] - radius,
                                                  center[1] - radius,
                                                  center[0] + radius,
                                                  center[1] + radius))
    return new_image


if __name__ == "__main__":
    tic = time()
    new_image = generate_new_data("inpaint-sample/12060_942.bmp", "inpaint-sample/12060_942.json")
    toc = time()
    print(toc - tic)
    new_image.show()
