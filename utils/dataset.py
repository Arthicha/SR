import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import sys
import cv2
import numpy as np
from math import ceil, floor
import time

def random_crop(hr, size, scale):
    h, w = hr.shape[:-1]
    x = random.randint(0, w-size*scale)
    y = random.randint(0, h-size*scale)
    t = time.time()
    crop_hr = hr[y:y+size*scale, x:x+size*scale]
    crop_lr = cv2.resize(crop_hr,(size,size))
    return crop_hr, crop_lr


def random_flip_and_rotate(im1,im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    return im1, im2


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale,batch_size):
        super(TrainDataset, self).__init__()

        self.size = size
        self.path = path
        self.batch_size = batch_size
        self.h5f = h5py.File(self.path, "r")
        
        self.scale = [scale]
        self.resample()

    def resample(self):
        file = self.h5f[str(np.random.randint(len(self.h5f.keys()), size=1)[0])]

        indx = np.random.choice(len(file), self.batch_size, replace=False)
        indx.sort()
        self.hr = [file[str(i)][:] for i in indx ]
        del file

    def __getitem__(self, index):
        size = self.size
        item = [random_crop(hr, size, self.scale[0]) for hr in self.hr]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]
        return item

    def __len__(self):
        return len(self.hr)
        
