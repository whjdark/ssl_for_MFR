'''
Author: whj
Date: 2022-02-15 12:36:42
LastEditors: whj
LastEditTime: 2022-02-16 11:38:53
Description: file content
'''
import torch
import torch.nn as nn



def act_helper(act_type):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    elif act_type == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif act_type == 'gelu':
        return nn.GELU(inplace=True)
    else:
        raise ValueError('Unsupported activation type: ' + act_type)