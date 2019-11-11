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
