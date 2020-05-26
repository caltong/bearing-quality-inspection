import pandas as pd
import os

root_dir = os.path.join('data', 'side')
train_or_val = ['train', 'val']
good_or_bad = ['good', 'bad']
# json文件临时位置
json_names = os.listdir('test')
# 训练或者评估
for train_val in train_or_val:
    # 正负样本
    data = []
    for good_bad in good_or_bad:
        path = os.path.join(root_dir, train_val, good_bad)
        image_names = os.listdir(path)
        # 路径下所有文件

        for image_name in image_names:
            one_data = [os.path.join(path, image_name)]
            if good_bad == 'good':
                one_data.append(1)
                one_data.append('None')
            else:
                one_data.append(0)
                json = image_name.split('.')[0] + '.json'
                if json in json_names:
                    one_data.append(os.path.join('test', json))
                else:
                    one_data.append('None')
            data.append(one_data)
    data_frame = pd.DataFrame(data)
    data_frame.to_csv("./" + train_val + ".csv", header=False, index=False)
    print(data_frame.info())
