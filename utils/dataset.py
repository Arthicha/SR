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
    return crop_hr


def random_flip_and_rotate(im1):
    if random.random() < 0.5:
        im1 = np.flipud(im1)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)

    return im1


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

        file = self.h5f[str(np.random.randint(16, size=1)[0])]

        indx = np.random.choice(len(file), self.batch_size, replace=False)
        indx.sort()
        self.hr = [file[str(i)][:] for i in indx ]
        del file

    def __getitem__(self, index):
        size = self.size
        item = [random_crop(hr, size, self.scale[0]) for hr in self.hr]
        item = [random_flip_and_rotate(hr) for hr in item]
        return item#[(self.transform(hr), self.transform(lr), self.transform(cr), self.transform(mk)) for hr, lr, cr, mk in item]

    def __len__(self):
        return len(self.hr)
        
