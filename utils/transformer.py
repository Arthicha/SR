import torch
import numpy as np
import cv2

def Transform(im,halfprecision=False):
	print(im.shape)
	im = torch.FloatTensor(im).cuda()/255
	print(im.shape)
	im = im.unsqueeze(0).permute(0,3,1,2)
	if halfprecision:
		im = im.half()

	return im

def deTransform(im,mx=255):
	im = im.permute(0,2,3,1).squeeze(0)
	im = torch.clamp(im,0,1)*mx
	im = im.type(torch.uint8).cpu().numpy()
	return im


def customTransform(inputs, index):
    im = np.array([input[index] for input in inputs])
    im = np.transpose(im, (0, 3, 1, 2))
    im = np.array(im, dtype=np.float32)  /255.0
    im = torch.FloatTensor(im)
    im = im.cuda()
    return im

