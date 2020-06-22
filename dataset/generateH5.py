import os
import glob
import h5py
import numpy as np
import cv2
import sys
import os
import random
from PIL import Image
from math import ceil, floor
import time
import argparse
from tqdm import tqdm

dataset_dir = "DIV2K"
dataset_type = "train"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="DIV2K_train")
    parser.add_argument("--hdf5_name", type=str, default="DIV2K_trainx.h5")
    parser.add_argument("--num_per_group", type=int, default=50)
    return parser.parse_args()

def main(cfg):
    file = 0
    im_paths = glob.glob(os.path.join(cfg.dataset_path,"*.png"))
    im_paths.sort()
    f = h5py.File(cfg.hdf5_name, "w")
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    grp = f.create_group(str(file))
    name = 0
    for _,imp in zip(tqdm(range(len(im_paths))),enumerate(im_paths)):
        i, path = imp
        im = cv2.imread(path)
        w,h,c = im.shape
        grp.create_dataset(str(name), data=im)
        name += 1
        if (i != len(im_paths)-1):
            if name == cfg.num_per_group:
                name = 0
                file += 1
                grp = f.create_group(str(file))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)