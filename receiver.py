#!/usr/bin/python3

import torch
import torch.nn as nn
import numpy as np
import cv2, socket, math, time, sys, os, threading, pygame, argparse, copy
from model.refiner import Net as model
from utils.transformer import Transform,deTransform
import SpoutSDK
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
from PIL import Image

''' DEFINE ARGUMENT PARSER'''
TRUE = ["True","true",'t','T','1']
FALSE = ["False",'false''f','F','0']
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str,choices=["sr", "hr"], default="sr")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=1112)
parser.add_argument("--image_width", type=int, default=1280)
parser.add_argument("--image_height", type=int, default=960)
parser.add_argument("--scale", type=int, default=4)
parser.add_argument("--nthread", type=int, default=8)
parser.add_argument("--max_receiver_fps", type=float, default=1/24)
parser.add_argument("--half_precision", type=str,choices=TRUE+FALSE, default='True')
parser.add_argument("--save_video", type=str,choices=TRUE+FALSE, default='False')
parser.add_argument("--save_nframe", type=int,default=100)
parser.add_argument("--video_name",type=str,default='output.avi')
parser.add_argument("--verbose", type=str,choices=TRUE+FALSE, default='True')
cfg = parser.parse_args()

MASK = cv2.imread('./dataset/mask.png')
MASK = cv2.resize(MASK,(cfg.image_width,cfg.image_height))
MASK = Transform(MASK)

if cfg.method == 'sr':
    neural_net = model().cuda()
    neural_net.load_state_dict(torch.load("checkpoint/checkpoint.pth"))
    neural_net.eval()
    neural_net.half() if cfg.half_precision in TRUE else None


EXIT = False
NUM_FRAME = 0
STATS = [-1.0]
GAMMA = 0.01

''' INITIAIZE PYGAME AND SPOUT'''
width = 800//800
height = 600//600
display = (width,height)
#pygame.init()
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
glDisable(GL_DEPTH_TEST)
glEnable(GL_ALPHA_TEST)
glEnable(GL_BLEND);
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
glClearColor(0.0,0.0,0.0,0.0)
glColor4f(1.0, 1.0, 1.0, 1.0);   
glTranslatef(0,0, -5)
glRotatef(25, 2, 1, 0)
spoutSender = SpoutSDK.SpoutSender()
spoutSenderWidth = width
spoutSenderHeight = height
spoutSender.CreateSender('Spout Python Sender', spoutSenderWidth, spoutSenderHeight, 0)
senderTextureID = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, 0)
glBindTexture(GL_TEXTURE_2D, senderTextureID)


''' CREATE SOCKET '''
SOCK_CONN   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
SOCK_CONN.bind((cfg.host, cfg.port))
SOCK_CONN.settimeout(10)
bufsize = SOCK_CONN.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
print("Initialized socket {}:{} ...".format(cfg.host, cfg.port))
print("Socket receiving buffer size = ", bufsize)
print("Waiting for data ...")

saver = cv2.VideoWriter(cfg.video_name,cv2.VideoWriter_fourcc('M','J','P','G'), 24, (cfg.image_width,cfg.image_height)) if cfg.save_video in TRUE else None

''' CREATE EMPTY IMAGES '''
IMAGE   = np.zeros((cfg.image_height//cfg.scale, cfg.image_width//cfg.scale,3), np.uint8) if cfg.method == 'sr' else np.zeros((cfg.image_height, cfg.image_width,3), np.uint8)
CENTER   = np.zeros((cfg.image_height, cfg.image_width, 3), np.uint8)

def PkgReader():
    global SOCK_CONN, IMAGE, CENTER, cfg

    '''
    A function for update frames received.
    The updated frame is stored in IMAGE and CENTER.
    '''

    SOCKET_BUFFER_DATA  = b""
    PROTOCOL_DATA_DELIMITER = b"[HEADER]"

    SOCKET_BUFF_SIZE = int(cfg.image_height * cfg.image_width * 3*0.25) + len(PROTOCOL_DATA_DELIMITER) + 1
    index_begin = None
    index_end   = None

    while True:
        # read data
        SOCKET_BUFFER_DATA  += SOCK_CONN.recvfrom(SOCKET_BUFF_SIZE)[0]
        #region Find HEADER
        while(True):
            if index_begin == None:
                res = SOCKET_BUFFER_DATA.find(PROTOCOL_DATA_DELIMITER)
                if res >= 0:
                    index_begin = res
                else:
                    break
            else:
                res = SOCKET_BUFFER_DATA.find(PROTOCOL_DATA_DELIMITER, index_begin + len(PROTOCOL_DATA_DELIMITER))
                if res > index_begin:
                    index_end   = res
                    package_number = int.from_bytes(SOCKET_BUFFER_DATA[index_begin + len(PROTOCOL_DATA_DELIMITER):index_begin + len(PROTOCOL_DATA_DELIMITER) + 1], "big")
                    if (package_number >= 0) and (package_number < (2*cfg.nthread+1)):
                        data_extract    = SOCKET_BUFFER_DATA[index_begin + len(PROTOCOL_DATA_DELIMITER) + 1:index_end]
                        if len(data_extract) > 0:
                            img     = cv2.imdecode(np.frombuffer(data_extract, np.uint8), cv2.IMREAD_COLOR)  # image decode
                            if (package_number < cfg.nthread) and (img is not None): # center
                                img_height_per_pkg  = cfg.image_height// (2*cfg.nthread)
                                img_py_start        = cfg.image_height//4 + int(package_number * img_height_per_pkg)
                                CENTER[img_py_start:img_py_start+img_height_per_pkg,cfg.image_width//4:3*cfg.image_width//4,:] = img
                            elif (package_number >= cfg.nthread) and (img is not None): # LR
                                img_height_per_pkg  = cfg.image_height//(cfg.scale*cfg.nthread) if cfg.method == 'sr' else cfg.image_height//cfg.nthread
                                img_py_start        = (package_number-cfg.nthread) * img_height_per_pkg
                                IMAGE[img_py_start:img_py_start + img_height_per_pkg, :,:]  = img
                    SOCKET_BUFFER_DATA  = SOCKET_BUFFER_DATA[index_end:]
                    index_begin = index_end = None
                else:
                    break

        if len(SOCKET_BUFFER_DATA) > SOCKET_BUFF_SIZE * 2:
            SOCKET_BUFFER_DATA  = b""



thread  = threading.Thread(target=PkgReader)
thread.start()

while(1):
    t_begin = time.perf_counter()
    hr = IMAGE

    ''' PERFORM SUPER RESOLUTION IF NECESSARY'''
    if cfg.method == 'sr':
        lr = Transform(hr)
        hr = neural_net(lr.half() if cfg.half_precision else lr)
        center = Transform(CENTER)
        hr = center*(1-MASK) + hr*(MASK)
        hr =deTransform(hr)

    ''' CONVERT IMAGE TO TEXTURE AND SEND THROUGH SPOUT '''
    glActiveTexture(GL_TEXTURE0)
    glClearColor(0.0,0.0,0.0,0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glRotatef(1, 3, 1, 1)
    glBindTexture(GL_TEXTURE_2D, senderTextureID)

    tx_image = Image.fromarray(hr)     
    ix = tx_image.size[0]
    iy = tx_image.size[1]
    tx_image = tx_image.tobytes('raw', 'BGRX', 0, -1)
    glBindTexture(GL_TEXTURE_2D, senderTextureID)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tx_image)
    glBindTexture(GL_TEXTURE_2D, 0)
    spoutSender.SendTexture(int(senderTextureID), GL_TEXTURE_2D, ix, iy, True, 0)

    cv2.imshow('image',hr)



    if cfg.save_video in TRUE:
        saver.release() if NUM_FRAME >= cfg.save_nframe else None
        saver.write(hr)
        

    NUM_FRAME += 1

    
    if cfg.verbose in TRUE:
        t = (time.perf_counter()-t_begin)
        STATS[0] = (1.0-GAMMA)*STATS[0] + GAMMA*(1.0/t) if STATS[0] >= 0.0 else 1.0/t
        if NUM_FRAME %100 == 0:
            print('---------------------------------------------------------------')
            print('Frame Rate',STATS[0],'FPS')
            NUM_FRAME = 0

    if cv2.waitKey(1) & 0xff == 27:
        print("Esc is pressed.\nExit")
        print("Closing socket ...")
        SOCK_CONN.close()
        EXIT = True
        sys.exit()

    # reduce fps
    t_end   = time.perf_counter()
    if t_end - t_begin < (cfg.max_receiver_fps):
        time.sleep(cfg.max_receiver_fps - (t_end - t_begin))
