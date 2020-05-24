from torch.utils.data import Dataset
import torch
from skimage import io
import numpy as np
import os
import pandas as pd
from PIL import Image


class GenerateDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0])
        image = Image.fromarray(io.imread(img_name))
        label = self.csv_data.iloc[idx, 1]
        label = np.array([label])
        label = label.astype('float')
        json_path = os.path.join(self.root_dir, self.csv_data.iloc[idx, 2])
        sample = {'image': image, 'label': label, 'image_path': img_name, 'json_path': json_path}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    dataset = GenerateDataset(csv_file='test/label.csv', root_dir='test')

    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample['image'].size, sample['label'].shape, sample['image_path'], sample['json_path'])
