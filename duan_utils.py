# -*- coding=utf-8 -*-
'''
#@ filename:  object-crop
#@ author:    cjr
#@ date:      2021-4-8
#@ brief:     object-crop method for the other two parts
'''

import cv2 as cv
import numpy as np

from PIL import Image

import os

def get_c_r_naive_duan(image):
    '''
    object crop for duan
    Args:
        image: pil format

    Returns:
        center coordinates corresponding to original image
        radius of selected circle
    '''
    img = np.asarray(image)
    img_shape = img.shape

    # convert bgr format to gray
    if len(img_shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(img, (9, 9), 0)
    (_, thresh) = cv.threshold(blurred, 90, 255, cv.THRESH_BINARY)

    # 形态学
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # 腐蚀膨胀
    closed = cv.erode(closed, None, iterations=1)
    closed = cv.dilate(closed, None, iterations=4)

    # 检测轮廓
    (_, cnts, _) = cv.findContours(
        # 参数一： 二值化图像
        closed.copy(),
        # 参数二：轮廓类型
        cv.RETR_EXTERNAL,  # 表示只检测外轮廓
        # cv2.RETR_CCOMP,                #建立两个等级的轮廓,上一层是边界
        # cv2.RETR_LIST,                 #检测的轮廓不建立等级关系
        # cv2.RETR_TREE,                 #建立一个等级树结构的轮廓
        # cv2.CHAIN_APPROX_NONE,         #存储所有的轮廓点，相邻的两个点的像素位置差不超过1
        # 参数三：处理近似方法
        cv.CHAIN_APPROX_SIMPLE,  # 例如一个矩形轮廓只需4个点来保存轮廓信息
        # cv2.CHAIN_APPROX_TC89_L1,
        # cv2.CHAIN_APPROX_TC89_KCOS
    )

    center = []
    radius = []
    for i in cnts:
        x, y, w, h = cv.boundingRect(i)

        # a simple yet effective method for contour selecting
        if 1030 <= h <= 1110 and 1030 <= w <= 1110:
            print(w, h, x, y)
            c = (int(x + w / 2), int(y + h / 2))
            r = int(w // 2)
            center.append(c)
            radius.append(r)

    return center, radius

class GenerateDuan:
    '''
    input image: PIL format
    '''
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        w, h = image.size
        center, radius = get_c_r_naive_duan(image)
        center, radius = center[0], radius[0]

        #  in case the boundary is exceeded
        if radius > center[1]:
            left_up_corner = (center[0] - radius, 0)
        else:
            left_up_corner = (center[0] - radius, center[1] - radius)
        if center[1] + radius > h:
            right_bottom_corner = (center[0] + radius, h)
        else:
            right_bottom_corner = (center[0] + radius, center[1] + radius)
        image = image.crop((left_up_corner[0], left_up_corner[1], right_bottom_corner[0], right_bottom_corner[1]))

        image = image.resize((1024, 1024))

        return {'image': image, 'label': label}