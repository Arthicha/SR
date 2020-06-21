import torch
import numpy as np


def Transform(im,halfprecision=False):
	im = torch.FloatTensor(im).cuda()/255
	im = im.unsqueeze(0).permute(0,3,1,2)
	if halfprecision:
		im = im.half()
	return im

def deTransform(im,mx=255):
	im = im.permute(0,2,3,1).squeeze(0)
	im = torch.clamp(im,0,1)*mx
	im = im.type(torch.uint8).cpu().numpy()
	return im