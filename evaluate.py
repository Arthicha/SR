import glob, os, sys, argparse, copy
import cv2, time
from PIL import Image
import PIL.Image as pil_image
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from model.refiner import Net as model
from model.FSRCNN import FSRCNN as fsrcnn
from model.vdsr import Net as vdsr
from model.edsr import EDSR as edsr
from model.carn_m import Net as carn
from utils.transformer import Transform,deTransform
from utils.imresize import imresize
from utils.evaluator import psnr, ssim
from utils.fsrcnn_utils import preprocess, convert_ycbcr_to_rgb
import utils.brisque.brisque as brisq

parser = argparse.ArgumentParser()
SR_METHODS = ['sr','fsrcnn','vdsr','edsr','carn']
parser.add_argument("--method", type=str,choices=["sr","fsrcnn", "hr",'vdsr','edsr','carn'], default="hr")
parser.add_argument("--image_width", type=int, default=1280)
parser.add_argument("--image_height", type=int, default=960)
parser.add_argument("--jpg_quality", type=int, default=100)
cfg = parser.parse_args()

if cfg.method in SR_METHODS:
	if cfg.method == 'sr':
		neural_net = model().cuda()
		neural_net.load_state_dict(torch.load("checkpoint/checkpoint.pth"))
	elif cfg.method == 'fsrcnn':
		neural_net = fsrcnn().cuda()
		neural_net.load_state_dict(torch.load("checkpoint/fsrcnn.pth"))
	elif cfg.method == 'vdsr':
		sys.path.append('model')
		neural_net = torch.load('checkpoint/vdsr.pth')["model"]
	elif cfg.method == 'edsr':
		neural_net = edsr().cuda()
		neural_net.load_state_dict(torch.load("checkpoint/edsr.pth"))
	elif cfg.method == 'carn':
		neural_net = carn().cuda()
		neural_net.load_state_dict(torch.load("checkpoint/carn_m.pth"))
	neural_net.eval()
	neural_net.half()

vpaths = glob.glob(os.path.join('dataset/PEXELS',"*.mp4"))
vpaths.sort()

MASK = cv2.imread('./dataset/mask.png')
MASK = cv2.resize(MASK,(cfg.image_width,cfg.image_height))
MASK = Transform(MASK)
center = np.zeros((cfg.image_height,cfg.image_width,3),np.uint8)

ematrix = [0.0,0.0,0.0,0.0]
nframe = 0
for path in vpaths:
	cap = cv2.VideoCapture(path)
	while 1:
		try:
			frame = cv2.resize(cap.read()[1], (cfg.image_width,cfg.image_height))
			original = copy.deepcopy(frame)

			centre = frame[frame.shape[0]//4:3*frame.shape[0]//4,frame.shape[1]//4:3*frame.shape[1]//4,:] if cfg.method in SR_METHODS else None
			
			if cfg.method == 'fsrcnn':
				frame = np.array(Image.fromarray(frame).resize((cfg.image_width//4,cfg.image_height//4), resample=pil_image.BICUBIC))
			elif cfg.method in ['vdsr','edsr','carn']:
				frame = imresize(frame,scalar_scale=0.25)
			elif cfg.method in ['sr']:
				frame = cv2.resize(frame, (cfg.image_width//4,cfg.image_height//4),interpolation=cv2.INTER_LINEAR)
			else:
				frame = frame


			centre = cv2.imencode(".jpg", centre, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpg_quality])[1] if cfg.method in SR_METHODS else None # convert to jpg
			centre = cv2.imdecode(centre, cv2.IMREAD_COLOR) if cfg.method in SR_METHODS else None

			frame = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpg_quality])[1] # convert to jpg
			frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

			if cfg.method in SR_METHODS:
				lr = frame
				# preprocess
				ti = time.perf_counter()
				if cfg.method in ['fsrcnn','vdsr']:
					if cfg.method == 'vdsr':
						lr = cv2.resize(lr,(cfg.image_width,cfg.image_height),interpolation=cv2.INTER_CUBIC)
					lr,ycbcr = preprocess(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))

				lr = Transform(lr,halfprecision=True)
				hr = neural_net(lr.half())
				# postprocess
				if cfg.method in ['fsrcnn','vdsr']:
					hr = hr.clamp(0.0, 1.0)
					ycbcr = Transform(cv2.resize(ycbcr,(hr.shape[-1],hr.shape[-2])))
					ycbcr[:,:1,:,:] = hr
					hr = convert_ycbcr_to_rgb(ycbcr*255)/255
				center[cfg.image_height//4:3*cfg.image_height//4,cfg.image_width//4:3*cfg.image_width//4,:] = centre
				centre = Transform(center)
				hr = centre*(1-MASK) + hr*(MASK)
				hr =deTransform(hr)
				frame = hr
				tf = time.perf_counter()
			try:
				nframe += 1
				ematrix[0] += psnr(hr,original)
				ematrix[1] += ssim(hr,original)
				ematrix[2] += brisq.getBRISQUE(hr)
				ematrix[3] += tf-ti
			except:
				print('ERRORRRRRRRRRRRR')
			print('frame',nframe,':',np.array(ematrix)/nframe)
			#cv2.imshow('frame',frame)
			#cv2.waitKey(1)
		except:
			break