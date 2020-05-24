import pandas as pd
import os

# 获取所有文件名
files = os.listdir("test")
# 创建list
json_list = []
image_list = []
for file in files:
    if file.endswith(".json"):
        json_list.append(file)
    elif not file.endswith(".csv"):
        image_list.append(file)

# 写入csv的数据
# [文件名, 正负样本, json名]
data = []
for image in image_list:
    one_data = [image, 0]
    json = image.split(".")[0] + ".json"
    if json in json_list:
        one_data.append(json)
    else:
        one_data.append("None")
    data.append(one_data)

data_frame = pd.DataFrame(data)
data_frame.to_csv("./test/label.csv", header=False, index=False)
print(data_frame)
print(data)
