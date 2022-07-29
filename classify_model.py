# -*- codeing = utf-8 -*-
import torch
import torch.nn as nn
import torchvision

CLASSES_NAME = ('bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 
                'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa',
                'normal', 'tungro')

# 用于resent50/101/152模型的block
class BottleNeck(nn.Module):
    # 扩展，表示该残差块的输出通道数与输入通道数不同
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        '''
        inplanes:当前特征图的通道数，也即输入该残差块的特征图的通道数
        plane：该残差块的第一个卷积的输出通道数，注意不是该残差块的输出通道数，expansion*plane才是该残差块的输出通道数
        '''
        super(BottleNeck, self).__init__()
        # 第一个卷积层，减少特征图的通道数，或者保持不变
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个卷积层，减小特征图的尺寸，或者保持不变
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, # change
                    padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 第3个卷积层，扩展通道数
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 输入量即为残差
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 是否需要下采样，在每一个layer的第一个残差块中都是需要下采样的，
        # 该layer的其他残差块不需要下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.classes_name = CLASSES_NAME
        
        self.num_classes = len(self.classes_name)
        
        self.net = torchvision.models.resnet50(pretrained=True)
        self.new_fc = nn.Linear(1000, self.num_classes)

    
    def forward(self, x):
        
        return self.new_fc(self.net(x))



class ResNeXt(nn.Module):
    def __init__(self):
        super().__init__()

        self.classes_name = CLASSES_NAME
        
        self.num_classes = len(self.classes_name)
        
        # self.net = torchvision.models.resnext50_32x4d(pretrained=True)
        self.net = torchvision.models.resnext101_32x8d(pretrained=True)
        self.new_fc = nn.Linear(1000, self.num_classes)

    
    def forward(self, x):
        
        return self.new_fc(self.net(x))


class WideResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.classes_name = CLASSES_NAME
        
        self.num_classes = len(self.classes_name)
        
        self.net = torchvision.models.wide_resnet50_2(pretrained=True)
        self.new_fc = nn.Linear(1000, self.num_classes)

    
    def forward(self, x):
        
        return self.new_fc(self.net(x))


# 具有6x下采样的ResNet网络
class ResNet_C6(nn.Module):
    def __init__(self):
        super(ResNet_C6, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.fc = resnet50.fc
        self.new_fc = nn.Linear(1000, 10)

        # 添加几层，用于生成C6特征图
        self.inplanes = 2048
        layer5 = self._make_layer(BottleNeck, 512, 2, stride=2)


        self.resnet = nn.Sequential(
            nn.Sequential(resnet50.conv1, resnet50.bn1,resnet50.relu),  # C1
            nn.Sequential(resnet50.maxpool, resnet50.layer1),  # C2  
            resnet50.layer2,  # C3
            resnet50.layer3,  # C4
            resnet50.layer4,  # C5
            layer5,
            resnet50.avgpool,
        )

        del resnet50


    def _make_layer(self, block, plane, num_blocks, stride=1):
        '''
        plane: 该layer的残差块的第一个卷积的输出通道数
        num_blocks: 该layer的残差块的个数
        stride：该layer是否需要缩小尺寸
        '''
        downsample = None
        # 如果需要进行下采样，或者当前的特征图的通道数不等于该残差块的输出通道数，无法进行add拼接
        if stride != 1 or self.inplanes != plane * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, plane * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(plane * block.expansion),
            )

        layers = []
        # 该layer的第一个残差块，
        # 对layer1来讲，第一个残差块的输入特征图与输出特征图之间，没有尺寸的变化，只有通道数的变化
        # 而对layer2-4来讲，第一个残差块既有尺寸的变化，也有通道数的变化
        layers.append(block(self.inplanes, plane, stride, downsample))
        
        # 当前的特征图的通道数发生变化
        self.inplanes = plane * block.expansion
        # 该layer后续的残差块，输入特征图与输出特征图不需要调整尺寸和通道数
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, plane))

        return nn.Sequential(*layers)


    def forward(self, x):
        
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.new_fc(self.fc(x))

        return x


# 具有6x下采样的ResNet网络
class ResNet_C6_2(nn.Module):
    def __init__(self):
        super(ResNet_C6_2, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.fc = resnet50.fc
        self.new_fc = nn.Linear(1000, 10)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.resnet = nn.Sequential(
            nn.Sequential(resnet50.conv1, resnet50.bn1,resnet50.relu),  # C1
            nn.Sequential(resnet50.maxpool, resnet50.layer1),  # C2  
            resnet50.layer2,  # C3
            resnet50.layer3,  # C4
            resnet50.layer4,  # C5
            maxpool,
            resnet50.avgpool,
        )

        del resnet50


    def forward(self, x):
        
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.new_fc(self.fc(x))

        return x
