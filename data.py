import torch
from torch.utils.data import Dataset
import os
from skimage import io
import skimage
import torchvision


class UpperAndLowerFacesData(Dataset):
    def __init__(self, transform=None):
        self.root_dir = os.path.join('data', '端面')
        self.good_dir = os.path.join(self.root_dir, 'good')
        self.bad_dir = os.path.join(self.root_dir, 'bad')
        self.all_img_path = []
        for name in os.listdir(self.good_dir):
            name = os.path.join(self.good_dir, name)
            self.all_img_path.append(name)
        for name in os.listdir(self.bad_dir):
            name = os.path.join(self.bad_dir, name)
            self.all_img_path.append(name)
        self.transform = transform

    def __len__(self):
        return len(self.all_img_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.all_img_path[idx]
        image = io.imread(img_name)
        is_good_or_bad = os.path.split(os.path.split(img_name)[0])[-1]
        if is_good_or_bad == 'good':
            label = 1
        else:
            label = 0
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        w, h = 1920, 1200
        image = image[0:h, int(w / 2 - h / 2):int(w / 2 + h / 2)]
        image = skimage.transform.resize(image, (224, 224))

        return {'image': image, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0)  # 添加1channel (batch_size,1,224,224)
        label = torch.tensor(label)
        return {'image': image, 'label': label}

# test sample

# upper_dataset = UpperAndLowerFacesData(transform=torchvision.transforms.Compose([Rescale(224),
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
