'''
Author: whj
Date: 2022-02-15 12:36:42
LastEditors: whj
LastEditTime: 2022-02-28 13:25:01
Description: file content
'''
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torchvision.models as models

from .act_helper import act_helper



class conv3dBlock(nn.Module):
    def __init__(self, _in, _out, ksize=3, stride=1, padding='same', dila=1, groups=1, bias=True, bn=False, act='relu'):
        super(conv3dBlock, self).__init__()
        self.bn = bn
        if padding == 'same':
            padding = (ksize - 1) // 2
        self.conv = nn.Conv3d(in_channels=_in, 
                            out_channels=_out, 
                            kernel_size=ksize, 
                            stride=stride,
                            padding=padding,
                            dilation=dila,
                            groups=groups,
                            bias=bias)
                            
        if self.bn:
            self.bn = nn.BatchNorm3d(_out)

        self.act = act_helper(act)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        return self.act(x)


class conv3d_4x4_Block(nn.Module):
    def __init__(self, _in, _out, ksize=4, stride=1, padding='same', dila=1, groups=1, bias=True, bn=False, act='relu'):
        super(conv3d_4x4_Block, self).__init__()
        self.bn = bn
        if padding == 'same':
            padding = (ksize - 1) // 2
        # pytorch do not have same padding
        # a even number convolution with same padding 
        # need to pad 1 more element
        # at left, _, top, _, front, _
        self.pad = nn.ConstantPad3d((1, 0, 1, 0, 1, 0), 0)
        self.conv = nn.Conv3d(in_channels=_in, 
                            out_channels=_out, 
                            kernel_size=ksize, 
                            stride=stride,
                            padding=padding,
                            dilation=dila,
                            groups=groups,
                            bias=bias)
                            
        if self.bn:
            self.bn = nn.BatchNorm3d(_out)

        self.act = act_helper(act)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        return self.act(x)


class FeatureNetBackbone(nn.Module):
    def __init__(self, in_channels=1):
        super(FeatureNetBackbone, self).__init__()
        self.block_1 = conv3dBlock(in_channels, 32, ksize=7, stride=2, dila=1, padding='same', groups=1, bias=True, bn=False, act='relu')
        self.block_2 = conv3dBlock(32, 32, ksize=5, stride=1, dila=1, groups=1, padding='same', bias=True, bn=False, act='relu')
        # special 4x4 convolution layer
        self.block_3 = conv3d_4x4_Block(32, 64)
        # pytorch do not have same padding
        # not use a 4x4 convolution layer
        # self.block_3 = conv3dBlock(32, 64, ksize=5, stride=1, dila=1, groups=1, padding='same', bias=True, bn=False, act='relu')
        self.block_4 = conv3dBlock(64, 64, ksize=3, stride=1, dila=1, groups=1, padding='same', bias=True, bn=False, act='relu')
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.pool(x)
        
        return x


class FeatureNet(nn.Module):
    '''
        paper: FeatureNet: Machining feature recognition based on 3D Convolution Neural Network
    '''
    def __init__(self, num_classes=24, input_shape=(64, 64, 64)):
        super(FeatureNet, self).__init__()
        self.backbone = FeatureNetBackbone(in_channels=1)
        # downsample 1/4 & out channel 64
        inshape = 64 * (input_shape[0] // 4) ** 3
        self.fc1 = nn.Linear(inshape, 128, bias=True)
        self.fc2 = nn.Linear(128, num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x