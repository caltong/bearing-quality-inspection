import os
from PIL import Image

root_dir = os.path.join('data', 'side')
train_or_val = ['train', 'val']
good_or_bad = ['good', 'bad']

for i in train_or_val:
    for j in good_or_bad:
        path = os.path.join(root_dir, i, j)
        images = os.listdir(path)
        for image in images:
            if Image.open(os.path.join(root_dir, i, j, image)).size != (1920, 1200):
                print(os.path.join(root_dir, i, j, image))
