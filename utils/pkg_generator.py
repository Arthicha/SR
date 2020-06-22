import socket
import cv2
import numpy as np
import math



def sendPackage(sock,img,cfg,start_pkg_num=0):
	NUM_BYTES = 0
	PROTOCOL_DATA_DELIMITER=b"[HEADER]"

	for n_pkg in range(0,cfg.nthread):
		img_height_per_pkg  = img.shape[0]//cfg.nthread
		img_py_start = n_pkg * img_height_per_pkg
		img_slice = img[img_py_start:img_py_start + img_height_per_pkg, :]
		img_jpeg = cv2.imencode(".jpg", img_slice, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpg_quality])[1] # convert to jpg
		HEADER_PKG_NUM = PROTOCOL_DATA_DELIMITER + bytes([n_pkg+start_pkg_num])
		sock.send(HEADER_PKG_NUM)   # send HEADER
		if cfg.verbose:
			NUM_BYTES += len(HEADER_PKG_NUM) # count byte sent

		package_num = math.ceil(len(img_jpeg) / cfg.sock_buff_size)
		for x in range(package_num):
			i_start = x * cfg.sock_buff_size
			data_to_send    = img_jpeg[i_start:i_start+cfg.sock_buff_size] # send jpeg
			sock.send(data_to_send)
			if cfg.verbose:
				NUM_BYTES += len(data_to_send) # count byte sent
	return NUM_BYTES

