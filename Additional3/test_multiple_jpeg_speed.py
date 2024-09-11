# this file is for device on the client side

# load the dataset

# This file is for trainning
# Run this on the server, or as we called offline. 

import argparse
import base64
import cv2
import datetime
from Models import gatedmodel,mobilenetv2, resnet
import numpy as np
import os
import PIL
import psutil
import sys
import time
import torch
from tqdm import tqdm
from Utils import utils, encoder

batch_size = 600

dataset = 'ccpd-'
i_stop = 1

# 2. dataset
# directly read bmp image from the storage
data_root = '../data/'+dataset+'-client/'
images_list = os.listdir(data_root)
images_list.remove('labels.txt')
# remove ending with jpg
images_list = [str(x)+'.bmp' for x in range(600)]
images_list = images_list[:batch_size * i_stop]

client_time = [0] * batch_size * i_stop

jpeg_folders_quality = [10, 20, 30, 40, 50, 60, 70, 80, 90]
jpeg_folders = []
for i in jpeg_folders_quality:
    jpeg_folders.append('../data/last-'+dataset+'-jpeg'+str(i)+'/')
    if not os.path.exists('../data/last-'+dataset+'-jpeg'+str(i)+'/'):
        os.makedirs('../data/last-'+dataset+'-jpeg'+str(i)+'/')
from PIL import Image

for i, i_path in tqdm(enumerate(images_list)):
    image_path = data_root + i_path
    image = Image.open(image_path).convert('RGB')
    if dataset == 'cifar-10':
        image = image.resize((32, 32))
    elif dataset == 'imagenet':
        image = image.resize((224, 224))
    image = np.array(image)
    # 3. compress the image
    # 3.1 JPEG 25
    for j in range(len(jpeg_folders_quality)):
        # store jpeg image
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_folders_quality[j]]
        # store the image
        new_image = cv2.imwrite(jpeg_folders[j] + i_path[:-4] + '.jpg', image, encode_param)
