import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from torch.nn import Parameter
import sys
import torch



def init_weights(modules):
    pass

def init_w(m):
    #pass
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)
        #init.uniform_(m.weight.data, -0.0057735, 0.0057735)


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, bias=True),
            nn.LeakyReLU(inplace=True)
        )

        self.body.apply(init_w)
        #self.CA = CALayer(in_channels,reduction=8)
        #init_weights(self.modules)
        
    def forward(self, x):
        #out = self.CA(x)
        out = self.body(x)
        return out




class EResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group, bias=True),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group, bias=True),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=True),
            #nn.BatchNorm2d(out_channels),
        )

        #init_weights(self.body)
        self.body.apply(init_w)
        
    def forward(self, x):
        out = self.body(x)
        out = F.leaky_relu(out + x)
        return out



class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super(UpsampleBlock, self).__init__()

        self.up =  _UpsampleBlock(n_channels, scale=scale, group=4)


    def forward(self, x):
        return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, 
				 n_channels, scale, 
				 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group, bias=True), nn.LeakyReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group, bias=True), nn.LeakyReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
        
        self.body = nn.Sequential(*modules)
        self.body.apply(init_w)
        #init_weights(self.body)
        
    def forward(self, x):
        out = self.body(x)
        return out
