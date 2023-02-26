'''
Author: whj
Date: 2022-02-28 13:23:59
LastEditors: whj
LastEditTime: 2022-02-28 13:25:11
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


class BaselineNetBackbone(nn.Module):
    '''
        paper: Identifying manufacturability and machining processes using deep 3D convolutional networks
    '''
    def __init__(self, in_channels=1):
        super(BaselineNetBackbone, self).__init__()
        self.block_1 = conv3dBlock(in_channels, 32, ksize=7, stride=2, dila=1, padding='same', groups=1, bias=True, bn=False, act='relu')
        self.block_2 = conv3dBlock(32, 32, ksize=5, stride=1, dila=1, groups=1, padding='same', bias=True, bn=False, act='relu')
        self.block_3 = conv3dBlock(32, 64, ksize=3, stride=1, dila=1, groups=1, padding='same', bias=True, bn=False, act='relu')
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.pool(x)
        
        return x


class BaselineNet(nn.Module):
    '''
        paper: Identifying manufacturability and machining processes using deep 3D convolutional networks
    '''
    def __init__(self, num_classes=24, input_shape=(64, 64, 64)):
        super(BaselineNet, self).__init__()
        self.backbone = BaselineNetBackbone(in_channels=1)
        self.drop = nn.Dropout(p=0.2)
        # downsample 1/4 & out channel 64
        inshape = 64 * (input_shape[0] // 4) ** 3
        self.fc1 = nn.Linear(inshape, 128, bias=True)
        self.fc2 = nn.Linear(128, num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.fc2(x)
        return x


class BaselineNet2Backbone(nn.Module):
    '''
        paper: Part machining feature recognition based on a deep learning method
    '''
    def __init__(self, in_channels=1):
        super(BaselineNet2Backbone, self).__init__()
        self.block_1 = conv3dBlock(in_channels, 32, ksize=3, stride=1, dila=1, padding='same', groups=1, bias=True, bn=False, act='relu')
        self.block_2 = conv3dBlock(32, 32, ksize=3, stride=1, dila=1, groups=1, padding='same', bias=True, bn=False, act='relu')
        self.block_3 = conv3dBlock(32, 32, ksize=3, stride=2, dila=1, groups=1, padding='same', bias=True, bn=False, act='relu')
        self.block_4 = conv3dBlock(32, 64, ksize=3, stride=1, dila=1, groups=1, padding='same', bias=True, bn=False, act='relu')
        self.block_5 = conv3dBlock(64, 64, ksize=3, stride=2, dila=1, groups=1, padding='same', bias=True, bn=False, act='relu')

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        
        return x


class BaselineNet2(nn.Module):
    '''
        paper: Part machining feature recognition based on a deep learning method
    '''
    def __init__(self, num_classes=24, input_shape=(64, 64, 64)):
        super(BaselineNet2, self).__init__()
        self.backbone = BaselineNet2Backbone(in_channels=1)
        # downsample 1/4 & out channel 64
        inshape = 64 * (input_shape[0] // 4) ** 3
        self.fc1 = nn.Linear(inshape, 124, bias=True)
        self.fc2 = nn.Linear(124, 124, bias=True)
        self.fc3 = nn.Linear(124, num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x


class VoxNet(torch.nn.Module):
    def __init__(self, num_classes, input_shape=(64, 64, 64)):
                 #weights_path=None,
                 #load_body_weights=True,
                 #load_head_weights=True):
        """
        VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition.
        Modified in order to accept different input shapes.
        Parameters
        ----------
        num_classes: int, optional
            Default: 10
        input_shape: (x, y, z) tuple, optional
            Default: (32, 32, 32)
        weights_path: str or None, optional
            Default: None
        load_body_weights: bool, optional
            Default: True
        load_head_weights: bool, optional
            Default: True
        Notes
        -----
        Weights available at: url to be added
        If you want to finetune with custom classes, set load_head_weights to False.
        Default head weights are pretrained with ModelNet10.
        """
        super(VoxNet, self).__init__()
        self.body = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv3d(in_channels=1,
                                      out_channels=32, kernel_size=5, stride=2)),
            ('lkrelu1', torch.nn.LeakyReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('lkrelu2', torch.nn.LeakyReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))

        # Trick to accept different input shapes
        x = self.body(torch.autograd.Variable(
            torch.rand((1, 1) + input_shape)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n

        self.head = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(first_fc_in_features, 128)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(128, num_classes))
        ]))

        #if weights_path is not None:
        #    weights = torch.load(weights_path)
        #    if load_body_weights:
        #        self.body.load_state_dict(weights["body"])
        #    elif load_head_weights:
        #        self.head.load_state_dict(weights["head"])

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x