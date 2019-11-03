import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import time
import copy

from utils import SideCenterCrop

epochs = 12
batch_size = 32
lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(1200),
        SideCenterCrop(),
        transforms.Resize(224),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(1200),
        SideCenterCrop(),
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
}

data_dir = os.path.join('data', '侧面')
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
