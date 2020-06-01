import os
from PIL import Image

root_dir = os.path.join('data', 'side')
train_or_val = ['train', 'val']
good_or_bad = ['good', 'bad']


def check_size():
    for i in train_or_val:
        for j in good_or_bad:
            path = os.path.join(root_dir, i, j)
            images = os.listdir(path)
            for image in images:
                if Image.open(os.path.join(root_dir, i, j, image)).size != (1920, 1200):
                    print(os.path.join(root_dir, i, j, image))


def check_duplicate():
    good = []
    bad = []
    for i in train_or_val:
        path = os.path.join(root_dir, i, 'good')
        images = os.listdir(path)
        for image in images:
            good.append(image)
    for i in train_or_val:
        path = os.path.join(root_dir, i, 'bad')
        images = os.listdir(path)
        for image in images:
            bad.append(image)

    set_a = set(good) & set(bad)
    print(set_a)
    print(len(set_a))
    for i in train_or_val:
        path = os.path.join(root_dir, i, 'bad')
        images = os.listdir(path)
        for duplicate in set_a:
            if duplicate in images:
                os.remove(os.path.join(root_dir, i, 'bad', duplicate))


if __name__ == '__main__':
    check_duplicate()
