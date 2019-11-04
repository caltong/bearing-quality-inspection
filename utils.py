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
        center_r = 455
        half_width = center_r * 1.05  # height = width
        img = img.crop((int(center_x - half_width), int(center_y - half_width),
                        int(center_x + half_width), int(center_y + half_width)))
        # size = img.size
        # mask = Image.new('L', size, 0)
        # draw = ImageDraw.Draw(mask)
        # draw.ellipse((0, 0) + size, fill=255)
        # output = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
        # output.putalpha(mask)
        # back = Image.new('L', size, 0)
        # back.paste(output, (0, 0), output)

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
        center_r = 455
        half_width = center_r * 1.05  # height = width
        img = img.crop((int(center_x - half_width), int(center_y - half_width),
                        int(center_x + half_width), int(center_y + half_width)))
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
