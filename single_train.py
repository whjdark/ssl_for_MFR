'''
Author: whj
Date: 2021-10-27 13:43:16
LastEditors: whj
LastEditTime: 2022-05-31 21:19:32
Description: file content
'''
from core.engine.trainer import (
    train_eval_model, eval_model, 
    infer_time_test, 
    draw_TSNE, draw_ROC_CM
    )
import argparse
import os
import sys
import random
import warnings

import torch

import numpy as np

# close warnings
warnings.filterwarnings('ignore')

# append root path of the project to the sys path
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)


# set flags / seeds / env
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# random.seed(42)
# os.environ['PYTHONHASHSEED'] = str(42)
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)

parser = argparse.ArgumentParser(description='Feature Recognition Training')
parser.add_argument('--data_path', dest='data_path',
                    default='data', type=str, help='path to the data')
parser.add_argument('--resolution', dest='resolution',
                    default=64, type=int, help='model resolution: 16, 32, 64')
parser.add_argument('--num_of_class', dest='num_of_class',
                    default=24, type=int, help='number of classes')
parser.add_argument('--num_train', dest='num_train', default=2,
                    type=int, help='number of training examples per class')
parser.add_argument('--num_val_test', dest='num_val_test', default=600,
                    type=int, help='number of val/test examples per class')
parser.add_argument('--arch', dest='arch', default='FeatureNet',
                    type=str, help='network arch: FeatureNet, FeatureNetLite, MsvNet, MsvNetLite, BaselineNet, BaselineNet2, VoxNet')
parser.add_argument('--base_lr', dest='base_lr', default=0.001,
                    type=float, help='base learning rate')
parser.add_argument('--train_epochs', dest='train_epochs', default=100,
                    type=int, help='num of epochs at surpvised training')
parser.add_argument('--train_batchsize', dest='train_batchsize',
                    default=64, type=int, help='train batch size')
parser.add_argument('--val_batchsize', dest='val_batchsize',
                    default=64, type=int, help='valid batch size')
parser.add_argument('--weight_decay', dest='weight_decay',
                    default=0.0, type=float, help='weight decay')
parser.add_argument('--warmup_epochs', dest='warmup_epochs',
                    default=0, type=int, help='warmup epochs')
parser.add_argument('--lr_sch', dest='lr_sch', default='constant',
                    type=str, help='learning rate scheduler type: constant, exp, cos, multistep, step')
parser.add_argument('--optim', dest='optim',
                    default='adam', type=str, help='optimizer type: adam, sgdm, rmsprop, adamw')
parser.add_argument('--data_aug', dest='data_aug', action='store_true',
                    help='whether to use data augmentation')
parser.add_argument('--num_cuts', dest='num_cuts', default=12,
                    type=int, help='number of cuts')
parser.add_argument('--pretrain', dest='pretrained', default=None,
                    type=str, help='pretrain model directory')
parser.add_argument('--simsiam_pretrain', dest='simsiam_pretrained', default=None,
                    type=str, help='simsiam pretrain model directory')
parser.add_argument('--freeze', dest='freeze', action='store_true',
                    help='whether to freeze the encoder when loading SSL pretrained model')
parser.add_argument('--val_interval', dest='val_interval',
                    default=10, type=int, help='valid interval')
parser.add_argument('--output_dir', dest='output_dir',
                    default='output', type=str, help='directory to save output')
parser.add_argument('--model_path', dest='model_path',
                    default=None, type=str, help='path to the trained model')
parser.add_argument('--program_type', dest='program_type',
                    default='train_eval', type=str, 
                    help='which program to run [train_eval, eval, infer_time_test, draw_TSNE, draw_ROC_CM]')
parser.add_argument('--device', dest='device',
                    default='gpu', type=str, help='which device to run')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.program_type == 'train_eval':
        train_eval_model(args)
    elif args.program_type == 'eval':
        eval_model(args)
    elif args.program_type == 'infer_time_test':
        infer_time_test(args)
    elif args.program_type == 'draw_TSNE':
        draw_TSNE(args)
    elif args.program_type == 'draw_ROC_CM':
        draw_ROC_CM(args)
    else:
        raise ValueError('program type not supported')
