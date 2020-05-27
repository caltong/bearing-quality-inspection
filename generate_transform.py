from generate import generate_new_data
from torchvision import transforms
from generate_dataset import GenerateDataset
import torch
import numpy as np
from utils import get_radius_center
from torchvision.transforms.functional import crop
from utils import add_black_background
from PIL import Image, ImageEnhance
import random


class Generate(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, flag=True):
        self.flag = flag

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # print(sample['json_path'])
        if self.flag:
            if sample['json_path'][-4:] != 'None':
                image = generate_new_data(sample['image_path'], sample['json_path'])
            else:
                center, radius = get_radius_center(image)
                left_up_corner = np.array(center) - radius
                image = crop(image, left_up_corner[1], left_up_corner[0], radius * 2, radius * 2)
        else:
            center, radius = get_radius_center(image)
            left_up_corner = np.array(center) - radius
            image = crop(image, left_up_corner[1], left_up_corner[0], radius * 2, radius * 2)
        image = image.resize((1024, 1024))
        return {'image': image, 'label': label}


class ColorJitter(object):
    def __init__(self, p=0.5, color=1.0, contrast=1.0, brightness=1.0, sharpness=1.0):
        self.p = p
        self.color = color
        self.contrast = contrast
        self.brightness = brightness
        self.sharpness = sharpness

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > self.p:
            return {'image': image, 'label': label}
        # image = Image.fromarray(image, mode='RGB')
        image = ImageEnhance.Color(image).enhance((random.random() + 0.5) * self.color)
        image = ImageEnhance.Contrast(image).enhance((random.random() + 0.5) * self.contrast)
        image = ImageEnhance.Brightness(image).enhance((random.random() + 0.5) * self.brightness)
        image = ImageEnhance.Sharpness(image).enhance((random.random() + 0.5) * self.sharpness)
        # image = np.array(image)
        return {'image': image, 'label': label}


class RandomRotation(object):
    def __init__(self, degrees):
        if degrees < 0 or degrees > 180:
            # 顺逆旋转 degrees最大值为180 最小值为0 0为不旋转
            raise Exception('角度值不合理')
        self.degrees = degrees

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.rotate(random.uniform(-self.degrees, self.degrees))
        return {'image': image, 'label': label}


class AddBlackBackground(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = add_black_background(image)
        return {'image': image, 'label': label}


class Resize(object):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        else:
            if size[0] != size[1]:
                raise Exception('尺寸必须为正方形')
        self.size = (size[0], size[1])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.resize(self.size)
        return {'image': image, 'label': label}


class Flip(object):
    def __init__(self, p):
        # p 为翻转概率
        if p < 0 or p > 1:
            raise Exception('概率p值不合理')
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.array(image)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        return {'image': torch.from_numpy(image).float(),
                'label': int(label)}


if __name__ == '__main__':
    transforms_dataset = GenerateDataset(csv_file='./train.csv', root_dir='./',
                                         transform=transforms.Compose([Generate(True),
                                                                       ColorJitter(0.5, 1.0, 1.0, 1.0, 1.0),
                                                                       AddBlackBackground(),
                                                                       RandomRotation(180),
                                                                       Flip(0.5),
                                                                       Resize(512)]))
    for i in range(len(transforms_dataset)):
        sample = transforms_dataset[i]
        if i == 5:
            break
        sample['image'].show()
        print(i, sample['image'], sample['label'])
    # train_data_loader = torch.utils.data.DataLoader(transforms_dataset, batch_size=16, shuffle=True)
    #
    # for i in train_data_loader:
    #     print(i['image'].shape)
