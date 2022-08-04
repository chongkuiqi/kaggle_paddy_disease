# -*- codeing = utf-8 -*-
import torch
import torch.nn as nn
import torchvision
# import timm

CLASSES_NAME = ('bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 
                'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa',
                'normal', 'tungro')

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.classes_name = CLASSES_NAME
        
        self.num_classes = len(self.classes_name)

        # net = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        # self.new_fc = nn.Linear(2048, self.num_classes)
        net = torchvision.models.resnet34(weights='IMAGENET1K_V1')
        self.new_fc = nn.Linear(512, self.num_classes)

        self.net = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
            net.avgpool,
        )

        del net

    
    def forward(self, x):
        
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.new_fc(x)

        return x


class ResNeXt(nn.Module):
    def __init__(self):
        super().__init__()

        self.classes_name = CLASSES_NAME
        
        self.num_classes = len(self.classes_name)
        
        net = torchvision.models.resnext50_32x4d(weights='IMAGENET1K_V2')
        self.new_fc = nn.Linear(2048, self.num_classes)
        self.net = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
            net.avgpool,
        )

        del net

    def forward(self, x):        
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.new_fc(x)

        return x

class ConvNext(nn.Module):
    def __init__(self):
        super().__init__()

        self.classes_name = CLASSES_NAME
        
        self.num_classes = len(self.classes_name)
        
        # net = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        # self.new_fc = nn.Linear(1024, self.num_classes)
        # net = torchvision.models.convnext_small(weights='IMAGENET1K_V1')
        # self.new_fc = nn.Linear(768, self.num_classes)
        net = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        self.new_fc = nn.Linear(768, self.num_classes)
        
        self.net = nn.Sequential(
            net.features,
            net.avgpool,
            net.classifier[:-1]
        )
        del net


    def forward(self, x):

        return self.new_fc(self.net(x))


class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.classes_name = CLASSES_NAME
        
        self.num_classes = len(self.classes_name)
        
        # net = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        # self.new_fc = nn.Linear(1280, self.num_classes)
        # net = torchvision.models.efficientnet_b4(weights='IMAGENET1K_V1')
        # self.new_fc = nn.Linear(1792, self.num_classes)
        net = torchvision.models.efficientnet_b3(weights='IMAGENET1K_V1')
        self.new_fc = nn.Linear(1536, self.num_classes)
        
        self.classifier = net.classifier[:-1]

        self.net = nn.Sequential(
            net.features,
            net.avgpool,
        )
        del net


    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.new_fc(x)
        return x
