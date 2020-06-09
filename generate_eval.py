import torch
from PIL import Image
from utils import get_radius_center
from torchvision.transforms.functional import crop
import numpy as np
from utils import add_black_background
import os
import time

model_path = 'resnet152.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path)
model.to(device)
model.eval()


def img2tensor(image_path):
    # 读取图片
    image_ori = Image.open(image_path)
    # 获取目标位置
    center, radius = get_radius_center(image_ori)
    left_up_corner = np.array(center) - radius
    image = crop(image_ori, left_up_corner[1], left_up_corner[0], radius * 2, radius * 2)
    # 添加黑色背景
    image = add_black_background(image)
    # 缩放
    image = image.resize((512, 512))
    # to tensor
    image = np.array(image)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    image = image[np.newaxis, ...]
    image = torch.from_numpy(image).float()
    return image, image_ori


def eval_in_dir(path):
    images = os.listdir(path)
    zero = 0
    one = 0
    total = len(images)
    for image in images:
        dir = os.path.join(path, image)
        time0 = time.time()
        img_tensor, img = img2tensor(dir)
        time1 = time.time()
        img_tensor.to(device)
        time2 = time.time()
        output = model(img_tensor.cuda())
        time3 = time.time()
        # print(output)
        # print(torch.max(output, 1))
        _, pred = torch.max(output, 1)
        if pred == 0:
            zero += 1
            if mode == 'good':
                print(image)
        else:
            one += 1
            if mode == 'bad':
                print(image)
        # print(pred)
        # print(time1 - time0, time2 - time1, time3 - time2)
    return zero, one, total


if __name__ == '__main__':
    mode = 'bad'
    tic = time.time()
    zero, one, total = eval_in_dir(os.path.join('data', 'side', 'val', 'bad'))
    toc = time.time()
    print('Eval on {} images, zero: {}, one: {}.'.format(total, zero, one))
    print('Use {}s per image.'.format((toc - tic) / total))
