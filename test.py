# -*- codeing = utf-8 -*-
# @Time : 2022/7/14 23:32
# @Author : wzh
# @File : train.py
# @Softwafe : PyCharm
from ast import Mod
import time
import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter

from Datasets import TestData, id_to_label
# from classify_model import ResNet as Model   # 
from classify_model import ResNeXt as Model   # 

from torch.utils.data import DataLoader

import tqdm

# 定义训练设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("训练设备：{}".format(device))

# 构建训练集和验证集
test_root_dir = '../dataset/test_images'
test_dataset = TestData(test_root_dir)

batch_size = 128
# dataloader加载数据
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# 导入模型resnet50_true

weights_path = "runs/train/exp30/best.pt"
ckpt = torch.load(weights_path, map_location=device)
resnet50 = Model().to(device)
resnet50.load_state_dict(ckpt['model'].state_dict())

start_time = time.time()


# 测试步骤开始
results = {}
resnet50.eval()
with torch.no_grad():
    for data in tqdm.tqdm(test_dataloader):
        imgs, imgs_name = data
        imgs = imgs.to(device)
        outputs = resnet50(imgs)
        output_classes_id = outputs.argmax(1).tolist()
        
        for img_name, cls_id in zip(imgs_name, output_classes_id):
            class_name = id_to_label[cls_id]

            results[img_name] = class_name

lines = ['image_id,label\n']
for k,v in results.items():
    line = k + ',' + v + '\n'

    lines.append(line)

file_name = 'submission.csv'
with open(file_name, 'w') as f:
    f.writelines(lines)