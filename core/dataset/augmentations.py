
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


def randomRotation(sample):
    rotation = random.randint(0, 23)
    if rotation == 1:
        sample = cp.rot90(sample, 1, (1, 2))
    elif rotation == 2:
        sample = cp.rot90(sample, 2, (1, 2))
    elif rotation == 3:
        sample = cp.rot90(sample, 1, (2, 1))
    elif rotation == 4:
        sample = cp.rot90(sample, 1, (0, 1))
    elif rotation == 5:
        sample = cp.rot90(sample, 1, (0, 1))
        sample = cp.rot90(sample, 1, (1, 2))
    elif rotation == 6:
        sample = cp.rot90(sample, 1, (0, 1))
        sample = cp.rot90(sample, 2, (1, 2))
    elif rotation == 7:
        sample = cp.rot90(sample, 1, (0, 1))
        sample = cp.rot90(sample, 1, (2, 1))
    elif rotation == 8:
        sample = cp.rot90(sample, 1, (1, 0))
    elif rotation == 9:
        sample = cp.rot90(sample, 1, (1, 0))
        sample = cp.rot90(sample, 1, (1, 2))
    elif rotation == 10:
        sample = cp.rot90(sample, 1, (1, 0))
        sample = cp.rot90(sample, 2, (1, 2))
    elif rotation == 11:
        sample = cp.rot90(sample, 1, (1, 0))
        sample = cp.rot90(sample, 1, (2, 1))
    elif rotation == 12:
        sample = cp.rot90(sample, 2, (1, 0))
    elif rotation == 13:
        sample = cp.rot90(sample, 2, (1, 0))
        sample = cp.rot90(sample, 1, (1, 2))
    elif rotation == 14:
        sample = cp.rot90(sample, 2, (1, 0))
        sample = cp.rot90(sample, 2, (1, 2))
    elif rotation == 15:
        sample = cp.rot90(sample, 2, (1, 0))
        sample = cp.rot90(sample, 1, (2, 1))
    elif rotation == 16:
        sample = cp.rot90(sample, 1, (0, 2))
    elif rotation == 17:
        sample = cp.rot90(sample, 1, (0, 2))
        sample = cp.rot90(sample, 1, (1, 2))
    elif rotation == 18:
        sample = cp.rot90(sample, 1, (0, 2))
        sample = cp.rot90(sample, 2, (1, 2))
    elif rotation == 19:
        sample = cp.rot90(sample, 1, (0, 2))
        sample = cp.rot90(sample, 1, (2, 1))
    elif rotation == 20:
        sample = cp.rot90(sample, 1, (2, 0))
    elif rotation == 21:
        sample = cp.rot90(sample, 1, (2, 0))
        sample = cp.rot90(sample, 1, (1, 2))
    elif rotation == 22:
        sample = cp.rot90(sample, 1, (2, 0))
        sample = cp.rot90(sample, 2, (1, 2))
    elif rotation == 23:
        sample = cp.rot90(sample, 1, (2, 0))
        sample = cp.rot90(sample, 1, (2, 1))

    return sample


def randomScaleCrop(sample):
    resolution = int(sample.shape[0])
    strategy = random.randint(0, 9)
    if strategy == 0:
        factor = random.uniform(1.0625, 1.25)
        sample = ndimage.zoom(sample, factor, order=0)
        startx = random.randint(0, sample.shape[0] - resolution)
        starty = random.randint(0, sample.shape[1] - resolution)
        startz = random.randint(0, sample.shape[2] - resolution)
        sample = sample[startx:startx+resolution,
                        starty:starty+resolution, startz:startz+resolution]
    elif strategy == 1:
        factor = random.uniform(0.9375, 0.75)
        sample = ndimage.zoom(sample, factor, order=0)
        padxwl = random.randint(0, resolution - sample.shape[0])
        padxwr = resolution - padxwl - sample.shape[0]
        padywl = random.randint(0, resolution - sample.shape[1])
        padywr = resolution - padywl - sample.shape[1]
        padzwl = random.randint(0, resolution - sample.shape[2])
        padzwr = resolution - padzwl - sample.shape[2]
        sample = np.pad(sample, ((padxwl, padxwr),
                                 (padywl, padywr), (padzwl, padzwr)), mode='edge')
    elif strategy == 2:
        padr = int(resolution/8)
        loc = 2*padr
        startx = random.randint(0, loc)
        starty = padr
        startz = padr
        sample = np.pad(sample, ((padr, padr), (padr, padr),
                                 (padr, padr)), mode='edge')
        sample = sample[startx:startx+resolution,
                        starty:starty+resolution, startz:startz+resolution]
    elif strategy == 3:
        padr = int(resolution/8)
        loc = 2*padr
        startx = padr
        starty = random.randint(0, loc)
        startz = padr
        sample = np.pad(sample, ((padr, padr), (padr, padr),
                                 (padr, padr)), mode='edge')
        sample = sample[startx:startx+resolution,
                        starty:starty+resolution, startz:startz+resolution]
    elif strategy == 4:
        padr = int(resolution/8)
        loc = 2*padr
        startx = padr
        starty = padr
        startz = random.randint(0, loc)
        sample = np.pad(sample, ((padr, padr), (padr, padr),
                                 (padr, padr)), mode='edge')
        sample = sample[startx:startx+resolution,
                        starty:starty+resolution, startz:startz+resolution]

    return sample


def randomScale(sample):
    resolution = int(sample.shape[0])
    strategy = random.randint(0, 2)
    if strategy == 0:
        factor = random.uniform(1.0625, 1.1)
        sample = ndimage.zoom(sample, factor, order=0)
        startx = random.randint(0, sample.shape[0] - resolution)
        starty = random.randint(0, sample.shape[1] - resolution)
        startz = random.randint(0, sample.shape[2] - resolution)
        sample = sample[startx:startx+resolution,
                        starty:starty+resolution, startz:startz+resolution]
    elif strategy == 1:
        factor = random.uniform(0.9375, 0.75)
        sample = ndimage.zoom(sample, factor, order=0)
        padxwl = random.randint(0, resolution - sample.shape[0])
        padxwr = resolution - padxwl - sample.shape[0]
        padywl = random.randint(0, resolution - sample.shape[1])
        padywr = resolution - padywl - sample.shape[1]
        padzwl = random.randint(0, resolution - sample.shape[2])
        padzwr = resolution - padzwl - sample.shape[2]
        sample = np.pad(sample, ((padxwl, padxwr),
                                 (padywl, padywr), (padzwl, padzwr)), mode='edge')
    
    return sample


def randomPadCrop(sample):
    resolution = int(sample.shape[0])
    strategy = random.randint(0, 3)
    if strategy == 0:
        padr = int(resolution/8)
        loc = 2*padr
        startx = random.randint(0, loc)
        starty = padr
        startz = padr
        sample = np.pad(sample, ((padr, padr), (padr, padr),
                                 (padr, padr)), mode='edge')
        sample = sample[startx:startx+resolution,
                        starty:starty+resolution, startz:startz+resolution]
    elif strategy == 1:
        padr = int(resolution/8)
        loc = 2*padr
        startx = padr
        starty = random.randint(0, loc)
        startz = padr
        sample = np.pad(sample, ((padr, padr), (padr, padr),
                                 (padr, padr)), mode='edge')
        sample = sample[startx:startx+resolution,
                        starty:starty+resolution, startz:startz+resolution]
    elif strategy == 2:
        padr = int(resolution/8)
        loc = 2*padr
        startx = padr
        starty = padr
        startz = random.randint(0, loc)
        sample = np.pad(sample, ((padr, padr), (padr, padr),
                                 (padr, padr)), mode='edge')
        sample = sample[startx:startx+resolution,
                        starty:starty+resolution, startz:startz+resolution]

    return sample



def cutout3D(sample):
    # parameters
    max_holes = 3
    max_cutout_size = 12
    # the random number of holes
    holes = random.randint(0, max_holes)
    if holes == 0:
        return sample
    # cutout
    resolution = int(sample.shape[0])
    for n in range(max_holes):
        y = np.random.randint(resolution)
        x = np.random.randint(resolution)
        z = np.random.randint(resolution)

        sizey = np.random.randint(4, max_cutout_size)
        sizex = np.random.randint(4, max_cutout_size)
        sizez = np.random.randint(4, max_cutout_size)

        y1 = np.clip(y - sizey // 2, 0, resolution)
        y2 = np.clip(y + sizey // 2, 0, resolution)
        x1 = np.clip(x - sizex // 2, 0, resolution)
        x2 = np.clip(x + sizex // 2, 0, resolution)
        z1 = np.clip(z - sizez // 2, 0, resolution)
        z2 = np.clip(z + sizez // 2, 0, resolution)

        sample[y1: y2, x1: x2, z1: z2] = 0

    return sample


