
# coding: utf-8

import os
from pathlib import Path
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torch.backends.cudnn as cudnn

import numpy as np
import cupy as cp
import cupyx.scipy
import cupyx.scipy.ndimage
import cupyx
from PIL import Image
from scipy import ndimage

from ..utils import read_as_3d_array
from .augmentations import *



class Object3DTo2D(object):
    def __init__(self, img_num, resolution):
        self.img_num = img_num
        self.tranform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(
                                    (resolution, resolution), 
                                    interpolation=transforms.InterpolationMode.NEAREST),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225])
                                ])
    
    def __call__(self, obj3d):
        imgs = []
        for _ in range(self.img_num):
            minres = min(obj3d.shape[0], obj3d.shape[1], obj3d.shape[2])
            proj_dir = random.randint(0, 1)
            sel_axis = random.randint(0, 2)
            sel_idx = random.randint(1, minres - 2)
            if sel_axis == 0:
                if proj_dir == 0:
                    img = cp.mean(obj3d[sel_idx:, :, :], sel_axis)
                else:
                    img = cp.mean(obj3d[:sel_idx, :, :], sel_axis)
            elif sel_axis == 1:
                if proj_dir == 0:
                    img = cp.mean(obj3d[:, sel_idx:, :], sel_axis)
                else:
                    img = cp.mean(obj3d[:, :sel_idx, :], sel_axis)
            elif sel_axis == 2:
                if proj_dir == 0:
                    img = cp.mean(obj3d[:, :, sel_idx:], sel_axis)
                else:
                    img = cp.mean(obj3d[:, :, :sel_idx], sel_axis)

            img = torch.from_numpy(img).float()
            img = img.expand(3, img.shape[0], img.shape[1])  # convert to rgb chanels
            img = self.tranform(img)
            imgs.append(img)

        return torch.stack(imgs)


class dataAugmentation(object):
    def __init__(self, ops=None):
        self.ops = ops

    def __call__(self, sample):
        if self.ops is not None:
            for op in self.ops:
                sample = eval(op)(sample)
        return sample


class FeatureDataset(Dataset):
    def __init__(self, list_IDs, resolution, output_type='3d', num_cuts=12, data_augmentation=None):
        self.list_IDs = list_IDs
        self.resolution = resolution
        self.output_type = output_type
        self.num_cuts = num_cuts
        self.data_augmentation = dataAugmentation(data_augmentation)
        self.createImgs = Object3DTo2D(self.num_cuts, self.resolution)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        idx = index
        ID = self.list_IDs[idx][0]
        rotation = self.list_IDs[idx][1]

        filename = ID + '.binvox'
        filepath = os.path.join('data', os.path.join(str(self.resolution), filename))
        with open(filepath, 'rb') as f:
            sample = read_as_3d_array(f).data

        if rotation == 1:
            sample = cp.rot90(sample, 2, (0, 1))
        elif rotation == 2:
            sample = cp.rot90(sample, 1, (0, 1))
        elif rotation == 3:
            sample = cp.rot90(sample, 1, (1, 0))
        elif rotation == 4:
            sample = cp.rot90(sample, 1, (2, 0))
        elif rotation == 5:
            sample = cp.rot90(sample, 1, (0, 2))

        sample = self.data_augmentation(sample)
        label = int(ID.split('_')[0])

        if self.output_type == '3d':
            sample = np.expand_dims(sample, axis=0)
            sample = torch.from_numpy(sample.copy()).float()
            sample = 2.0 * (sample - 0.5)
        elif self.output_type == '2d_multiple':
            sample = self.createImgs(sample)
        elif self.output_type == '2d_single':
            sample = self.createImgs(sample)
            label = torch.zeros(self.num_cuts, dtype=torch.int64) + label

        return sample, label


def createPartition(data_path, num_classes = 24, resolution=16, num_train=30, num_val_test=30):
    counter = np.zeros(num_classes, np.int64)
    partition = {}
    for i in range(num_classes):
        partition['train', i] = []
        partition['val', i] = []
        partition['test', i] = []

    with open(os.devnull, 'w') as devnull:
        path = Path(os.path.join(data_path, str(resolution)))
        for filename in sorted(path.glob('*.binvox')):
            namelist = os.path.basename(filename).split('_')
            label = int(namelist[0])
            counter[label] += 1

            items = []
            for i in range(6):
                items.append((os.path.basename(filename).split('.')[0], i))

            if counter[label] % 10 < 8:
                partition['train', label] += items
            elif counter[label] % 10 == 8:
                partition['val', label] += items
            elif counter[label] % 10 == 9:
                partition['test', label] += items

    ret = {}
    ret['train'] = []
    ret['val'] = []
    ret['test'] = []

    for i in range(num_classes):
        random.shuffle(partition['train', i])
        random.shuffle(partition['val', i])
        random.shuffle(partition['test', i])

        ret['train'] += partition['train', i][0:num_train]
        ret['val'] += partition['val', i][0:num_val_test]
        ret['test'] += partition['test', i][0:num_val_test]

    random.shuffle(ret['train'])
    random.shuffle(ret['val'])
    random.shuffle(ret['test'])

    return ret


class SimsiamDataset(Dataset):
    def __init__(self, data_path, num_classes = 24, num_train=4800, resolution=64, output_type='3d', num_cuts=12, transform=None):
        self.resolution = resolution
        self.output_type = output_type
        self.num_cuts = num_cuts
        self.transform = dataAugmentation(transform)
        self.createImgs = Object3DTo2D(self.num_cuts, self.resolution)

        self.list_IDs = []
        with open(os.devnull, 'w') as devnull:
            path = Path(os.path.join(data_path, str(resolution)))
            for filename in sorted(path.glob('*.binvox')):
                items = []
                for i in range(6):
                    items.append((os.path.basename(filename).split('.')[0], i))
                self.list_IDs.extend(items)

        random.shuffle(self.list_IDs)
        total_train = num_classes * num_train
        assert total_train <= len(self.list_IDs)
        self.list_IDs = self.list_IDs[0:total_train]

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        idx = index
        ID = self.list_IDs[idx][0]
        rotation = self.list_IDs[idx][1]

        filename = ID + '.binvox'
        filepath = os.path.join('data', os.path.join(str(self.resolution), filename))
        with open(filepath, 'rb') as f:
            sample = read_as_3d_array(f).data

        if rotation == 1:
            sample = cp.rot90(sample, 2, (0, 1))
        elif rotation == 2:
            sample = cp.rot90(sample, 1, (0, 1))
        elif rotation == 3:
            sample = cp.rot90(sample, 1, (1, 0))
        elif rotation == 4:
            sample = cp.rot90(sample, 1, (2, 0))
        elif rotation == 5:
            sample = cp.rot90(sample, 1, (0, 2))

        q = self.transform(sample)
        k = self.transform(sample)

        if self.output_type == '3d':
            q, k = np.expand_dims(q, axis=0), np.expand_dims(k, axis=0)
            q, k = torch.from_numpy(q.copy()).float(), torch.from_numpy(k.copy()).float()
            q, k = 2.0 * (q - 0.5), 2.0 * (k - 0.5)
        elif self.output_type == '2d_multiple':
            q, k = self.createImgs(q), self.createImgs(k)
        elif self.output_type == '2d_single':
            q, k = self.createImgs(q), self.createImgs(k)

        return [q, k]
