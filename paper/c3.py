import cv2
import matplotlib
import os

train_bad = "../data_all/side/train/bad"
train_good = "../data_all/side/train/good"
val_bad = "../data_all/side/val/bad"
val_good = "../data_all/side/val/good"

paths = [train_bad, train_good, val_bad, val_good]

gray = 0
rgb = 0
for path in paths:
    images = os.listdir(path)
    for image in images:
        image = os.path.join(path, image)
        image = cv2.imread(image)
        if image.shape[2] == 3:
            rgb += 1
        else:
            gray += 1
print("gray: " + str(gray))
print("grb: " + str(rgb))


