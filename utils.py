import torch
from torch.utils.data import Dataset
import os
from skimage import io
import skimage
import numpy as np
import cv2
import numbers
import torchvision
from PIL import Image, ImageDraw, ImageOps
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random


def add_black_center(image):
    # 添加黑色中心
    size = image.size
    circle = np.zeros(size, dtype='uint8')

    # circle = np.stack((circle,) * 3, -1)
    if len(np.array(image).shape) == 2:
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
    else:
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # 黑白照片其实不需要RGB2BGR
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=200, maxRadius=300)
    if circles is not None:
        cv2.circle(circle, (int(circles[0][0][0]), int(circles[0][0][1])), int(circles[0][0][2]), 255, -1)
    else:
        cv2.circle(circle, (size[0] // 2, size[1] // 2), int((size[0] // 2) * 0.5), 255, -1)
    img_np = cv2.bitwise_and(img_np, img_np, mask=cv2.bitwise_not(circle))
    img = Image.fromarray(np.array(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)))

    return img


def add_black_background(image):
    # 添加黑色背景 适配side model
    size = image.size
    circle = np.zeros(size, dtype='uint8')
    cv2.circle(circle, (size[0] // 2, size[1] // 2), int((size[0] // 2) * 0.95), 1, -1)
    circle = np.stack((circle,) * 3, -1)
    if len(np.array(image).shape) == 2:
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
    else:
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # 黑白照片其实不需要RGB2BGR
    img_np = img_np * circle  # 0去除1留存
    img = Image.fromarray(np.array(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)))

    return img


def get_radius_center(image):
    # image = Image.open('data/upper/train/bad/06031_233.bmp')
    img = np.array(image)
    img_shape = img.shape
    if len(img_shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # 已经是灰度图
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # 提取梯度
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)
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

    # 获得长度大于200的轮廓
    big_cnts = []
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


class CircleToRectangle(object):
    def __call__(self, img):
        center, radius = get_radius_center(img)
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        rec = []
        r1 = 220
        r2 = radius
        rotation = random.random() * 2 * math.pi
        for i in np.linspace(rotation, 2 * math.pi + rotation, 448):
            col = []
            for j in np.linspace(r1, r2, 448):
                x = int(j * math.cos(i) + center[0])
                y = int(j * math.sin(i) + center[1])
                col.append(img_np[y][x])
            rec.append(col)
        rec = np.array(rec)
        target_rec = cv2.cvtColor(rec, cv2.COLOR_GRAY2RGB)
        target_rec_pillow = Image.fromarray(target_rec)

        return target_rec_pillow


class TargetCenterCrop(object):
    # fit well on all three models
    def __call__(self, img):
        center, radius = get_radius_center(img)
        center_x = center[0]
        center_y = center[1]
        center_r = radius
        half_width = int(center_r * 1.01)  # height = width
        img = img.crop((int(center_x - half_width), int(center_y - half_width),
                        int(center_x + half_width), int(center_y + half_width)))
        # 生成mask覆盖非检测区域
        size = img.size
        circle = np.zeros(size, dtype='uint8')  # 黑色背景
        cv2.circle(circle, (size[0] // 2, size[1] // 2), size[0] // 2, 1, -1)  # 中心圆内不改变
        circle = np.stack((circle,) * 3, -1)  # 扩展维度 gray to rgb
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # 黑白照片其实不需要RGB2BGR
        img_np = img_np * circle  # 0去除1留存
        img = Image.fromarray(np.array(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)))  # 黑白照片其实不需要RGB2BGR

        return img


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

    # def focal_loss_alt(self, x, y):
    #     '''Focal loss alternative.
    #     Args:
    #       x: (tensor) sized [N,D].
    #       y: (tensor) sized [N,].
    #     Return:
    #       (tensor) focal loss.
    #     '''
    #     alpha = 0.25
    #
    #     t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)
    #     t = t[:, 1:]
    #     t = Variable(t).cuda()
    #
    #     xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
    #     pt = (2 * xt + 1).sigmoid()
    #
    #     w = alpha * t + (1 - alpha) * (1 - t)
    #     loss = -w * pt.log() / 2
    #     return loss.sum()
    #
    # def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
    #     '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
    #     Args:
    #       loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
    #       loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
    #       cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
    #       cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
    #     loss:
    #       (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
    #     '''
    #     batch_size, num_boxes = cls_targets.size()
    #     pos = cls_targets > 0  # [N,#anchors]
    #     num_pos = pos.data.long().sum()
    #
    #     ################################################################
    #     # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
    #     ################################################################
    #     mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
    #     masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
    #     masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
    #     loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
    #
    #     ################################################################
    #     # cls_loss = FocalLoss(loc_preds, loc_targets)
    #     ################################################################
    #     pos_neg = cls_targets > -1  # exclude ignored anchors
    #     mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
    #     masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
    #     cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])
    #
    #     print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0] / num_pos, cls_loss.data[0] / num_pos), end=' | ')
    #     loss = (loc_loss + cls_loss) / num_pos
    #     return loss


class UpperAndLowerCenterCrop(object):
    def __call__(self, img):
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 30, 100)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=33, minRadius=420,
                                   maxRadius=470)
        if circles is None:
            raise AssertionError('OpenCV find no circle.')
        circle = circles[0][0]
        center_x = circle[0]
        center_y = circle[1]
        center_r = 470
        half_width = center_r * 1.05  # height = width
        img = img.crop((int(center_x - half_width), int(center_y - half_width),
                        int(center_x + half_width), int(center_y + half_width)))
        # 生成mask覆盖非检测区域
        size = img.size
        circle = np.zeros(size, dtype='uint8')  # 黑色背景
        cv2.circle(circle, (size[0] // 2, size[1] // 2), size[0] // 2, 1, -1)  # 中心圆内不改变
        circle = np.stack((circle,) * 3, -1)  # 扩展维度 gray to rgb
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # 黑白照片其实不需要RGB2BGR
        img_np = img_np * circle  # 0去除1留存
        img = Image.fromarray(np.array(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)))  # 黑白照片其实不需要RGB2BGR

        return img


class SideCenterCrop(object):
    def __call__(self, img):
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 30, 100)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=33, minRadius=420,
                                   maxRadius=470)
        if circles is None:
            raise AssertionError('OpenCV find no circle.')
        circle = circles[0][0]
        center_x = circle[0]
        center_y = circle[1]
        center_r = 470
        half_width = center_r * 1.05  # height = width
        img = img.crop((int(center_x - half_width), int(center_y - half_width),
                        int(center_x + half_width), int(center_y + half_width)))
        # 生成mask覆盖非检测区域
        size = img.size
        circle = np.zeros(size, dtype='uint8')  # 黑色背景
        cv2.circle(circle, (size[0] // 2, size[1] // 2), size[0] // 2, 1, -1)  # 中心圆内不改变
        circle = np.stack((circle,) * 3, -1)  # 扩展维度 gray to rgb
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # 黑白照片其实不需要RGB2BGR
        img_np = img_np * circle  # 0去除1留存
        img = Image.fromarray(np.array(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)))  # 黑白照片其实不需要RGB2BGR

        return img


class ChamferCenterCrop(object):
    def __call__(self, img):
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 30, 100)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=33, minRadius=420,
                                   maxRadius=470)
        if circles is None:
            raise AssertionError('OpenCV find no circle.')
        circle = circles[0][0]
        center_x = circle[0]
        center_y = circle[1]
        center_r = 470
        half_width = center_r * 1.05  # height = width
        img = img.crop((int(center_x - half_width), int(center_y - half_width),
                        int(center_x + half_width), int(center_y + half_width)))
        # 生成mask覆盖非检测区域
        size = img.size
        circle = np.zeros(size, dtype='uint8')  # 黑色背景
        cv2.circle(circle, (size[0] // 2, size[1] // 2), size[0] // 2, 1, -1)  # 中心圆内不改变
        circle = np.stack((circle,) * 3, -1)  # 扩展维度 gray to rgb
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # 黑白照片其实不需要RGB2BGR
        img_np = img_np * circle  # 0去除1留存
        img = Image.fromarray(np.array(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)))  # 黑白照片其实不需要RGB2BGR

        return img


def show_img(img):
    img = (np.array(img) * 255).astype(np.uint8)
    img = np.moveaxis(img, 0, -1)
    img = Image.fromarray(img)
    return img
# class UpperAndLowerFacesData(Dataset):
#     def __init__(self, transform=None):
#         self.root_dir = os.path.join('data', '端面')
#         self.good_dir = os.path.join(self.root_dir, 'good')
#         self.bad_dir = os.path.join(self.root_dir, 'bad')
#         self.all_img_path = []
#         for name in os.listdir(self.good_dir):
#             name = os.path.join(self.good_dir, name)
#             self.all_img_path.append(name)
#         for name in os.listdir(self.bad_dir):
#             name = os.path.join(self.bad_dir, name)
#             self.all_img_path.append(name)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.all_img_path)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         img_name = self.all_img_path[idx]
#         image = io.imread(img_name)
#         is_good_or_bad = os.path.split(os.path.split(img_name)[0])[-1]
#         if is_good_or_bad == 'good':
#             label = 1
#         else:
#             label = 0
#         sample = {'image': image, 'label': label}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample
#
#
# class Rescale(object):
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         w, h = 1920, 1200
#         image = image[0:h, int(w / 2 - h / 2):int(w / 2 + h / 2)]
#         image = skimage.transform.resize(image, (self.output_size, self.output_size))
#
#         return {'image': image, 'label': label}
#
#
# class RandomCrop(object):
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
#
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#
#         h, w = image.shape[:2]
#         new_h, new_w = self.output_size
#
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)
#
#         image = image[top: top + new_h, left: left + new_w]
#
#         return {'image': image, 'label': label}
#
#
# class ToTensor(object):
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         image = torch.tensor(image, dtype=torch.float)
#         image = image.unsqueeze(0)  # 添加1channel (batch_size,1,224,224)
#         label = torch.tensor(label, dtype=torch.float)
#         return {'image': image, 'label': label}

# test sample

# upper_dataset = UpperAndLowerFacesData(transform=torchvision.transforms.Compose([Rescale(256),
#                                                                                  RandomCrop(224),
#                                                                                  ToTensor()]))
# data_loader = torch.utils.data.DataLoader(upper_dataset, batch_size=4, shuffle=True)
#
# for i in range(5):
#     sample = upper_dataset[i]
#     print(i, sample['image'].size(), sample['label'].size())
#
# for i_batch, sample_batched in enumerate(data_loader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['label'].size())
