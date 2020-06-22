import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import TrainDataset
from utils.transformer import customTransform, deTransform
import sys
import cv2
import math
import torchvision.transforms as transforms
import time

class Solver():
    def __init__(self, model, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.refiner = model().to(self.device)

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr)
        
        self.epoch = 0
        if cfg.ckpt_name != 'None':
            self.load(cfg.ckpt_name)
        self.step = self.epoch*cfg.update_every
        learning_rate = self.decay_learning_rate()

        #for param_group in self.optim.param_groups:
        for param_group in self.optim.param_groups:
            param_group["lr"] = learning_rate

        self.loss_fn = nn.L1Loss()

        if cfg.verbose:
            num_params = sum(p.numel() for p in self.refiner.parameters() if p.requires_grad)
            print("# of params:", num_params)

        os.makedirs(cfg.saved_ckpt_dir, exist_ok=True)


    def fit(self):
        cfg = self.cfg
        self.refiner.zero_grad()
        self.refiner.train()
        self.train_loader = TrainDataset(cfg.train_data_path, size=cfg.patch_size, scale=cfg.scale,batch_size=cfg.batch_size)

        while True:
            self.train_loader.resample()
            inputs = self.train_loader[0]
            
            hr = customTransform(inputs,0)
            lr = customTransform(inputs,1)

            del inputs
            
            if 1:
                sr = self.refiner(lr)
                l1loss = self.loss_fn(sr, hr)/cfg.update_every
                loss = l1loss
                loss.backward()
                
                if self.step % cfg.update_every == 0:
                    nn.utils.clip_grad_norm_(self.refiner.parameters(), cfg.clip)
                    self.optim.step()
                    self.refiner.zero_grad()
                    self.epoch += 1
                    learning_rate = self.decay_learning_rate()
                    for param_group in self.optim.param_groups:
                        param_group["lr"] = learning_rate
            
            if 1: 
                cv2.imshow('sr',deTransform(sr[:1]))
                cv2.imshow('hr',deTransform(hr[:1]))
                cv2.waitKey(1)
                self.step += 1
                if cfg.verbose and self.step % (cfg.update_every*10) == 0:
                    print('epoch', self.epoch, 'l1_loss', l1loss.item())
                    if cfg.verbose and self.step % (cfg.update_every*100) == 0:
                        self.save()

    

    def load(self, path):
        states = torch.load(path)
        state_dict = self.refiner.state_dict()
        for k, v in states.items():
            if k in state_dict.keys():
                state_dict.update({k: v})
        self.refiner.load_state_dict(state_dict)

    def save(self):
        torch.save(self.refiner.state_dict(), self.cfg.saved_ckpt_dir+'/checkpoint_e'+str(self.epoch)+'.pth')

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.epoch // self.cfg.decay))
        return lr
