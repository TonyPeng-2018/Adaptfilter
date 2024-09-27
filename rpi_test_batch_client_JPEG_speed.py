# this file is for device on the client side

# load the dataset

# This file is for trainning
# Run this on the server, or as we called offline.

import argparse
import base64
import cv2
import datetime
from Models import gatedmodel, mobilenetv2, resnet
import numpy as np
import os
import PIL
import psutil
import sys
import time
import torch
from tqdm import tqdm
from Utils import utils, encoder

gate_confidence = 0.85
batch_size = 100

dataset = "ccpd"
i_stop = 1

# 2. dataset
# directly read bmp image from the storage
data_root = "../data/" + dataset + "-client/"
# images_list = os.listdir(data_root)
# images_list.remove('labels.txt')
# # remove ending with jpg
# images_list = [x for x in images_list if x.endswith('.bmp')]
# images_list = sorted(images_list)
images_list = [str(x) + ".bmp" for x in range(batch_size)]

client_time = [0] * batch_size * i_stop

jpeg25_folder = "../data/" + dataset + "-jpeg25/"
jpeg75_folder = "../data/" + dataset + "-jpeg75/"
cjpeg_folder = "../data/" + dataset + "-cjpeg/"

if not os.path.exists(jpeg25_folder):
    os.makedirs(jpeg25_folder)
if not os.path.exists(jpeg75_folder):
    os.makedirs(jpeg75_folder)
if not os.path.exists(cjpeg_folder):
    os.makedirs(cjpeg_folder)

f = open("rpi-jpeg.txt", "w")

from PIL import Image

JPEG_time_25 = [0] * batch_size * i_stop
JPEG_time_75 = [0] * batch_size * i_stop
CJPEG_time = [0] * batch_size * i_stop
for i, i_path in tqdm(enumerate(images_list)):
    image_path = data_root + i_path
    image = Image.open(image_path).convert("RGB")
    if dataset == "cifar-10":
        image = image.resize((32, 32))
    elif dataset == "imagenet":
        image = image.resize((224, 224))
    image = np.array(image)
    # 3. compress the image
    # 3.1 JPEG 25
    s_time = time.time()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
    result, encimg = cv2.imencode(".jpg", image, encode_param)
    encimg = encimg.tobytes()
    encimg = base64.b64encode(encimg)
    e_time = time.time()
    with open(jpeg25_folder + i_path[:-4], "wb") as f2:
        f2.write(encimg)

    JPEG_time_25[i] = e_time - s_time

    # 3.3 JPEG 75
    s_time = time.time()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    # result, encimg = cv2.imencode('.jpg', image, encode_param)
    result, encimg = cv2.imencode(".jpg", image, encode_param)
    encimg = encimg.tobytes()
    encimg = base64.b64encode(encimg)
    e_time = time.time()
    with open(jpeg75_folder + i_path[:-4], "wb") as f2:
        f2.write(encimg)

    JPEG_time_75[i] = e_time - s_time

    # 3.4 progressive JPEG
    s_time = time.time()
    encode_param = [int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
    # result, encimg = cv2.imencode('.jpg', image, encode_param)
    result, encimg = cv2.imencode(".jpg", image, encode_param)
    encimg = encimg.tobytes()
    encimg = base64.b64encode(encimg)
    e_time = time.time()
    with open(cjpeg_folder + i_path[:-4], "wb") as f2:
        f2.write(encimg)

    CJPEG_time[i] = e_time - s_time

    f.write(
        i_path
        + " "
        + str(JPEG_time_25[i])
        + " "
        + str(JPEG_time_75[i])
        + " "
        + str(CJPEG_time[i])
        + "\n"
    )
avg_jpeg25 = [0] * i_stop
avg_jpeg75 = [0] * i_stop
avg_cjpeg = [0] * i_stop

for i in range(i_stop):
    avg_jpeg25[i] = (
        sum(JPEG_time_25[i * batch_size : (i + 1) * batch_size]) / batch_size * 1000
    )
    avg_jpeg75[i] = (
        sum(JPEG_time_75[i * batch_size : (i + 1) * batch_size]) / batch_size * 1000
    )
    avg_cjpeg[i] = (
        sum(CJPEG_time[i * batch_size : (i + 1) * batch_size]) / batch_size * 100
    )
# print average time
print("avg_jpeg25:", avg_jpeg25)
print("avg_jpeg75:", avg_jpeg75)
print("avg_cjpeg:", avg_cjpeg)
