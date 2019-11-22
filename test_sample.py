from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from time import time


def get_radius_center(image):
    # image = Image.open('data/upper/train/bad/06031_233.bmp')
    img = np.array(image)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img  # 已经是灰度图
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # 提取梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # 去噪 二值化
    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    # 形态学
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # 腐蚀膨胀
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # 检测轮廓
    (_, cnts, _) = cv2.findContours(
        # 参数一： 二值化图像
        closed.copy(),
        # 参数二：轮廓类型
        cv2.RETR_EXTERNAL,  # 表示只检测外轮廓
        # cv2.RETR_CCOMP,                #建立两个等级的轮廓,上一层是边界
        # cv2.RETR_LIST,                 #检测的轮廓不建立等级关系
        # cv2.RETR_TREE,                 #建立一个等级树结构的轮廓
        # cv2.CHAIN_APPROX_NONE,         #存储所有的轮廓点，相邻的两个点的像素位置差不超过1
        # 参数三：处理近似方法
        cv2.CHAIN_APPROX_SIMPLE,  # 例如一个矩形轮廓只需4个点来保存轮廓信息
        # cv2.CHAIN_APPROX_TC89_L1,
        # cv2.CHAIN_APPROX_TC89_KCOS
    )

    # 获得面积最大的两个轮廓
    big_cnts = []  # 较大的cnts
    for i in cnts:
        _, _, w, h = cv2.boundingRect(i)
        if (w > 200) or (h > 200):
            big_cnts.append(i)
    cnt = big_cnts[0]
    for i in big_cnts[1:]:
        cnt = np.vstack((cnt, i))
    # 求得轮廓的最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius


# chamfer r =470
# side r = 460
# upper r =445
wrong = []
path = 'data/side/train/good'
tic = time()
for i in os.listdir(path):
    center, radius = get_radius_center(Image.open(os.path.join(path, i)))
    if (radius < 455) or (radius > 465):
        print(center, radius, i)
        wrong.append(os.path.join(path, i))
toc = time()
print('{}s per image'.format((tic - toc) / len(os.listdir(path))))
