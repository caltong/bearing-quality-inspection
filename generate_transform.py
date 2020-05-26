from generate import generate_new_data
from torchvision import transforms
from generate_dataset import GenerateDataset
import torch
import numpy as np
from utils import get_radius_center
from torchvision.transforms.functional import crop
from utils import add_black_background


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
        image = image.resize((512, 512))
        image = add_black_background(image)
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
                                         transform=transforms.Compose([Generate(True)]))
    for i in range(len(transforms_dataset)):
        sample = transforms_dataset[i]
        if i == 10:
            sample['image'].show()
        print(i, sample['image'], sample['label'])
    # train_data_loader = torch.utils.data.DataLoader(transforms_dataset, batch_size=16, shuffle=True)
    #
    # for i in train_data_loader:
    #     print(i['image'].shape)
