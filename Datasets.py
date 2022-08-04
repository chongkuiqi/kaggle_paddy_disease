# -*- codeing = utf-8 -*-
# @Time : 2022/7/15 10:49
# @Author : wzh
# @File : dataset.py
# @Softwafe : PyCharm
from email.mime import image
import os

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
# import numpy as np
import cv2

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


train_transforms = A.Compose([
    A.OneOf([
        A.Rotate(30, p=0.5),
        A.HorizontalFlip(p=0.5),
        # A.CenterCrop(height=480,width=480,p=1),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1)
    ], p=0.8),
    A.Resize(320, 240),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(320, 240),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 构建训练集
class TrainData(Dataset):
    # 根据图像数据集路径（由根目录名‘root_dir’和数据集名‘label_dir’组成）生成Dataset
    def __init__(self, root_dir, label_dir, img_size=(320,240), mode='train'):
        self.root_dir = root_dir
        self.label_dir = label_dir
        # 获取数据集路径
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 将数据集路径下单个图像的名称存储在列表中
        self.img_path = os.listdir(self.path)
        
        assert mode in ('train', 'val')
        if mode == 'train':
            self.trans_compose = train_transforms
        elif mode=='val':
            self.trans_compose = val_transforms
        else:
            raise NotImplementedError

    # 获取单个样本图像相关信息
    def __getitem__(self, idx):
        # 获取单个样本图像名
        img_name = self.img_path[idx]
        # 获取单个样本图像路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # 根据图像路径获取图像
        img = cv2.imread(img_item_path)[:, :, ::-1]  # convert it to rgb
        img = img.astype('float32')
        # img [channel, H, W]
        img = self.trans_compose(image=img)['image']

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

        
        self.trans_compose = val_transforms

    # 获取单个样本图像相关信息
    def __getitem__(self, idx):
        # 获取单个样本图像名
        img_name = self.img_path[idx]
        # 获取单个样本图像路径
        img_item_path = os.path.join(self.root_dir, img_name)
        # 根据图像路径获取图像
        img = cv2.imread(img_item_path)[:, :, ::-1]  # convert it to rgb
        img = img.astype('float32')
        
        img = self.trans_compose(image=img)['image']

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

    bacterial_leaf_blight_dataset = TrainData(root_dir, bacterial_leaf_blight_label_dir, img_size=img_size, mode=mode)
    bacterial_leaf_streak_dataset = TrainData(root_dir, bacterial_leaf_streak_label_dir, img_size=img_size, mode=mode)
    bacterial_panicle_blight_dataset = TrainData(root_dir, bacterial_panicle_blight_label_dir, img_size=img_size, mode=mode)
    blast_dataset = TrainData(root_dir, blast_label_dir, img_size=img_size, mode=mode)
    brown_spot_dataset = TrainData(root_dir, brown_spot_label_dir, img_size=img_size, mode=mode)
    dead_heart_dataset = TrainData(root_dir, dead_heart_label_dir, img_size=img_size, mode=mode)
    downy_mildew_dataset = TrainData(root_dir, downy_mildew_label_dir, img_size=img_size, mode=mode)
    hispa_dataset = TrainData(root_dir, hispa_label_dir, img_size=img_size, mode=mode)
    normal_dataset = TrainData(root_dir, normal_label_dir, img_size=img_size, mode=mode)
    tungro_dataset = TrainData(root_dir, tungro_label_dir, img_size=img_size, mode=mode)

    # 数据集的拼接
    train_dataset = bacterial_leaf_blight_dataset + bacterial_leaf_streak_dataset + \
                    bacterial_panicle_blight_dataset + blast_dataset + \
                    brown_spot_dataset + dead_heart_dataset + \
                    downy_mildew_dataset + hispa_dataset + \
                    normal_dataset + tungro_dataset
    
    if mode == 'train':
        root_dir = '../dataset/val_images'
        bacterial_leaf_blight_dataset = TrainData(root_dir, bacterial_leaf_blight_label_dir, img_size=img_size, mode=mode)
        bacterial_leaf_streak_dataset = TrainData(root_dir, bacterial_leaf_streak_label_dir, img_size=img_size, mode=mode)
        bacterial_panicle_blight_dataset = TrainData(root_dir, bacterial_panicle_blight_label_dir, img_size=img_size, mode=mode)
        blast_dataset = TrainData(root_dir, blast_label_dir, img_size=img_size, mode=mode)
        brown_spot_dataset = TrainData(root_dir, brown_spot_label_dir, img_size=img_size, mode=mode)
        dead_heart_dataset = TrainData(root_dir, dead_heart_label_dir, img_size=img_size, mode=mode)
        downy_mildew_dataset = TrainData(root_dir, downy_mildew_label_dir, img_size=img_size, mode=mode)
        hispa_dataset = TrainData(root_dir, hispa_label_dir, img_size=img_size, mode=mode)
        normal_dataset = TrainData(root_dir, normal_label_dir, img_size=img_size, mode=mode)
        tungro_dataset = TrainData(root_dir, tungro_label_dir, img_size=img_size, mode=mode)

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

