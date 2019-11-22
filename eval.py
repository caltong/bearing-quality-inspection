import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, center_crop, resize
import os
import time
import argparse
from utils import get_radius_center

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help='data path')
parser.add_argument('model_path', type=str, help='model_path')
parser.add_argument('mode', type=str, help='bad or good')
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
mode = args.mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = torchvision.models.resnet50()
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 2)
# model.load_state_dict(torch.load('upper_and_lower_model_use_restnet50_crop_and_crop.pth'))
model = torch.load(model_path)
model.to(device)
model.eval()


# img = Image.open('data/端面/val/good/10251_69.bmp')

def img2tensor(path):
    img = Image.open(path)
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
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)  # 黑白照片其实不需要RGB2BGR
    img_np = img_np * circle  # 0去除1留存
    img = Image.fromarray(np.array(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)))  # 黑白照片其实不需要RGB2BGR
    img = resize(img, 224)
    # img_np = np.array(img)
    # img_np = np.moveaxis(img_np, -1, 0)
    # img_np = img_np[np.newaxis, ...]
    # img_tensor = torch.tensor(img_np, dtype=torch.float32)
    img_tensor = to_tensor(img)
    img_tensor = img_tensor.numpy()
    img_tensor = img_tensor[np.newaxis, ...]
    img_tensor = torch.from_numpy(img_tensor)
    return img_tensor, img


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


zero, one, total = eval_in_dir(data_path)
print('Eval on {} images, zero: {}, one: {}.'.format(total, zero, one))
