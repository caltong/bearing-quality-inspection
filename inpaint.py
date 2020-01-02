import numpy as np
from matplotlib import pyplot as plt
import cv2
import json

img_id = 'inpaint-sample/12060_394'
img = cv2.imread(img_id + '.bmp')
with open(img_id + '.json') as f:
    json_labels = json.load(f)

labels = []
for i in json_labels['shapes']:
    labels.append(i['points'])

# 创建mask
mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
for i in labels:
    cv2.fillPoly(mask, [np.array(i, np.int32)], (255, 0, 0))

result = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
plt.imshow(result)
plt.show()
