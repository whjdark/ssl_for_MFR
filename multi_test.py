'''
Author: whj
Date: 2022-02-15 15:48:15
LastEditors: whj
LastEditTime: 2022-02-17 00:31:34
Description: file content
'''
import os
import sys

import torch

import numpy as np

# append root path of the project to the sys path
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from engine import multi as mlt

torch.backends.cudnn.benchmark = True
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


group_idx = 3

p,r = mlt.test_msvnet(group_idx)

print('Precision for the MsvNet on group ', group_idx, ': ',p)
print('Recall for the MsvNet on group ', group_idx, ': ',r)

p,r = mlt.test_featurenet(group_idx)


print('Precision for the FeatureNet on group ', group_idx, ': ',p)
print('Recall for the FeatureNet on group ', group_idx, ': ',r)

msvnet_ssl_num_1