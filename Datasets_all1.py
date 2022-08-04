# -*- codeing = utf-8 -*-
# @Time : 2022/7/15 10:49
# @Author : wzh
# @File : dataset.py
# @Softwafe : PyCharm
import os

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader

label_to_label = {'bacterial_leaf_blight': 0, 'bacterial_leaf_streak': 1,
                  'bacterial_panicle_blight': 2, 'blast': 3,
                  'brown_spot': 4, 'dead_heart': 5,
                  'downy_mildew': 6, 'hispa': 7,
                  'normal': 8, 'tungro': 9
                  }

id_to_label = {0:'bacterial_leaf_blight', 1:'bacterial_leaf_streak',
                  2:'bacterial_panicle_blight', 3:'blast',
                  4:'brown_spot', 5:'dead_heart',
                  6:'downy_mildew', 7:'hispa',
                  8:'normal', 9:'tungro'
                  }


# 构建训练集
class TrainData(Dataset):
    # 根据图像数据集路径（由根目录名‘root_dir’和数据集名‘label_dir’组成）生成Dataset
    def __init__(self, root_dir, label_dir, img_size=(320,240)):
        self.root_dir = root_dir
        self.label_dir = label_dir
        # 获取数据集路径
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 将数据集路径下单个图像的名称存储在列表中
        self.img_path = os.listdir(self.path)

        # 图像转化成tensor型,并调整成统一格式
        trans_resize = torchvision.transforms.Resize(img_size) 
        # 注意，如果是PIL图像，.ToTensor()基本上都会缩放到[0.0,1.0]之间，详细看官方教程
        trans_tensor = torchvision.transforms.ToTensor()           # 转化为tensor类型
        trans_normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))       # 按照预训练模型参数进行归一化
        
        self.trans_compose = torchvision.transforms.Compose([trans_resize, trans_tensor, trans_normalize])

    # 获取单个样本图像相关信息
    def __getitem__(self, idx):
        # 获取单个样本图像名
        img_name = self.img_path[idx]
        # 获取单个样本图像路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # 根据图像路径获取图像
        img = Image.open(img_item_path)
        
        # img [channel, H, W]
        img = self.trans_compose(img)

        # 获取图像标签（此处图像label为其文件夹名称）
        label = self.label_dir
        label = label_to_label[label]
        label = torch.tensor(label)
        # 返回图像与标签

        return img, label

    # 获取样本长度
    def __len__(self):
        return len(self.img_path)


# 构建训练集
class TestData(Dataset):
    # 根据图像数据集路径（由根目录名‘root_dir’和数据集名‘label_dir’组成）生成Dataset
    def __init__(self, root_dir, img_size=(320,240)):
        self.root_dir = root_dir
        
        # 将数据集路径下单个图像的名称存储在列表中
        self.img_path = os.listdir(self.root_dir)

        # 图像转化成tensor型,并调整成统一格式
        trans_resize = torchvision.transforms.Resize(img_size) 
        # 注意，如果是PIL图像，.ToTensor()基本上都会缩放到[0.0,1.0]之间，详细看官方教程
        trans_tensor = torchvision.transforms.ToTensor()           # 转化为tensor类型
        trans_normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))       # 按照预训练模型参数进行归一化
        
        self.trans_compose = torchvision.transforms.Compose([trans_resize, trans_tensor, trans_normalize])
        

    # 获取单个样本图像相关信息
    def __getitem__(self, idx):
        # 获取单个样本图像名
        img_name = self.img_path[idx]
        # 获取单个样本图像路径
        img_item_path = os.path.join(self.root_dir, img_name)
        # 根据图像路径获取图像
        img = Image.open(img_item_path)
        
        img = self.trans_compose(img)

        # 返回图像与标签
        return img, img_name

    # 获取样本长度
    def __len__(self):
        return len(self.img_path)


def dataset_fn(root_dir, img_size, mode='train'):
    bacterial_leaf_blight_label_dir = 'bacterial_leaf_blight'
    bacterial_leaf_streak_label_dir = 'bacterial_leaf_streak'
    bacterial_panicle_blight_label_dir = 'bacterial_panicle_blight'
    blast_label_dir = 'blast'
    brown_spot_label_dir = 'brown_spot'
    dead_heart_label_dir = 'dead_heart'
    downy_mildew_label_dir = 'downy_mildew'
    hispa_label_dir = 'hispa'
    normal_label_dir = 'normal'
    tungro_label_dir = 'tungro'

    bacterial_leaf_blight_dataset = TrainData(root_dir, bacterial_leaf_blight_label_dir, img_size=img_size)
    bacterial_leaf_streak_dataset = TrainData(root_dir, bacterial_leaf_streak_label_dir, img_size=img_size)
    bacterial_panicle_blight_dataset = TrainData(root_dir, bacterial_panicle_blight_label_dir, img_size=img_size)
    blast_dataset = TrainData(root_dir, blast_label_dir, img_size=img_size)
    brown_spot_dataset = TrainData(root_dir, brown_spot_label_dir, img_size=img_size)
    dead_heart_dataset = TrainData(root_dir, dead_heart_label_dir, img_size=img_size)
    downy_mildew_dataset = TrainData(root_dir, downy_mildew_label_dir, img_size=img_size)
    hispa_dataset = TrainData(root_dir, hispa_label_dir, img_size=img_size)
    normal_dataset = TrainData(root_dir, normal_label_dir, img_size=img_size)
    tungro_dataset = TrainData(root_dir, tungro_label_dir, img_size=img_size)

    # 数据集的拼接
    train_dataset = bacterial_leaf_blight_dataset + bacterial_leaf_streak_dataset + \
                    bacterial_panicle_blight_dataset + blast_dataset + \
                    brown_spot_dataset + dead_heart_dataset + \
                    downy_mildew_dataset + hispa_dataset + \
                    normal_dataset + tungro_dataset
    if mode == 'train':
        root_dir = '../dataset/val_images'
        bacterial_leaf_blight_dataset = TrainData(root_dir, bacterial_leaf_blight_label_dir, img_size=img_size)
        bacterial_leaf_streak_dataset = TrainData(root_dir, bacterial_leaf_streak_label_dir, img_size=img_size)
        bacterial_panicle_blight_dataset = TrainData(root_dir, bacterial_panicle_blight_label_dir, img_size=img_size)
        blast_dataset = TrainData(root_dir, blast_label_dir, img_size=img_size)
        brown_spot_dataset = TrainData(root_dir, brown_spot_label_dir, img_size=img_size)
        dead_heart_dataset = TrainData(root_dir, dead_heart_label_dir, img_size=img_size)
        downy_mildew_dataset = TrainData(root_dir, downy_mildew_label_dir, img_size=img_size)
        hispa_dataset = TrainData(root_dir, hispa_label_dir, img_size=img_size)
        normal_dataset = TrainData(root_dir, normal_label_dir, img_size=img_size)
        tungro_dataset = TrainData(root_dir, tungro_label_dir, img_size=img_size)

        train_dataset += bacterial_leaf_blight_dataset + bacterial_leaf_streak_dataset + \
                    bacterial_panicle_blight_dataset + blast_dataset + \
                    brown_spot_dataset + dead_heart_dataset + \
                    downy_mildew_dataset + hispa_dataset + \
                    normal_dataset + tungro_dataset
    
    return train_dataset


def create_dataloader(data_path, img_size, shuffle=False, batch_size=1, num_workers=8, mode='train'):

    dataset = dataset_fn(data_path, img_size, mode=mode)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    return dataloader, dataset


'''
# 测试
import torchvision
from torch import nn
resnet50_true = torchvision.models.resnet50(pretrained=True)
resnet50_true.fc = nn.Linear(2048, 10)

verification_root_dir = 'verification_images'
verification_dataset = dataset_fn(verification_root_dir)
verification_dataloader = DataLoader(verification_dataset, batch_size=64)
for data in verification_dataloader:
    imgs, targets = data
    # imgs = imgs.to(device)
    # targets = targets.to(device)
    outputs = resnet50_true(imgs)
    # print(outputs)
    # test_loss = loss_fn(outputs, targets)
    outputs = outputs.argmax(1)
    print('真实值：{}'.format(targets))
    print('输出值：{}'.format(outputs))
'''

'''
# 验证数据集的构建
train_root_dir = 'train_images'
verification_root_dir = 'verification_images'
train_dataset = dataset_fn(train_root_dir)
verification_dataset = dataset_fn(verification_root_dir)
# print('训练集大小:{},验证集大小:{}.'.format(len(train_dataset), len(verification_dataset)))

imgs, targets = verification_dataset[100]
print(targets)
print(targets.type)
'''

'''
# 构建测试集
class TestData(Dataset):
    # 根据图像数据集路径（由根目录名‘root_dir’）生成Dataset
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # 获取数据集路径
        self.path = self.root_dir
        # 将数据集路径下单个图像的名称存储在列表中
        self.img_path = os.listdir(self.path)

    # 获取单个样本图像相关信息
    def __getitem__(self, idx):
        # 获取单个样本图像名
        img_name = self.img_path[idx]
        # 获取单个样本图像路径
        img_item_path = os.path.join(self.root_dir, img_name)
        # 根据图像路径获取图像
        img = Image.open(img_item_path)
        # 返回图像
        return img

    # 获取样本长度
    def __len__(self):
        return len(self.img_path)

test_root_dir = 'verification_images'
test_dataset = TestData(test_root_dir)
print(len(test_dataset))
img = test_dataset[0]
img.show()
'''
