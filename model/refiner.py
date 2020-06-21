import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from copy import deepcopy
import model.ops as ops
import numpy as np
import cv2
import sys


class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()
        self.channel = 32
        self.b1 = ops.EResidualBlock(self.channel, self.channel, group=1)
        self.b2 = ops.EResidualBlock(self.channel, self.channel, group=1)
        self.c1 = ops.BasicBlock(self.channel*2, self.channel, 1, 1, 0)
        self.c2 = ops.BasicBlock(self.channel*3, self.channel, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        return o2

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.carnchannel = 32
        

        self.conv1_e = nn.Conv2d(1, self.carnchannel//2, 3, 1, 2, bias=False).to(self.device)
        self.conv2 = nn.Conv2d(self.carnchannel//2, self.carnchannel//2, 3, 1, 1, bias=False).to(self.device)

        torch.nn.init.xavier_normal_(self.conv1_e.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)


        self.entry = nn.Conv2d(3, self.carnchannel//2, 3, 1, 1, bias=True).to(self.device)
        torch.nn.init.xavier_normal_(self.entry.weight)

        self.b1 = Block(self.carnchannel, self.carnchannel).to(self.device)
        self.b2 = Block(self.carnchannel, self.carnchannel).to(self.device)
        self.c1 = ops.BasicBlock(self.carnchannel*2, self.carnchannel, 1, 1, 0).to(self.device)
        self.c2 = ops.BasicBlock(self.carnchannel*3, self.carnchannel, 1, 1, 0).to(self.device)

        self.upsample = ops.UpsampleBlock(self.carnchannel, scale=4).to(self.device)
        self.exit = nn.Conv2d(self.carnchannel, 3, 3, 1, 1, bias=True).to(self.device)
        torch.nn.init.xavier_normal_(self.exit.weight)

    def carn(self, ip):
        b1 = self.b1(ip)
        c1 = torch.cat([ip, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        return o2

    def forward(self,lr ,coarse=False):

        lr = F.leaky_relu(self.entry(lr),inplace=True)

        sr = torch.cat([lr,lr*0],dim=1)
        sr = self.carn(sr)
        sr = self.upsample(sr)
        sr = self.exit(sr)

        return sr


    def init_w(self,m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_normal(m.weight)
            #init.normal_(m.weight.data, 0.0, 0.2)
