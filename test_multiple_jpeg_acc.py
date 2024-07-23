# load resnet and label

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

dataset = 'imagenet-20'

# 2. dataset
# directly read bmp image from the storage

data_root = '../data/'+dataset+'-client/'
label = open(data_root + 'labels.txt', 'r')
label = label.read()
label = label.split('\n')

jpeg_folders_quality = [10, 20, 30, 40, 50, 60, 70, 80, 90]
img_folders = ['../data/last-'+dataset+'-jpeg'+str(x)+'/' for x in jpeg_folders_quality]

client, server = resnet.resnet_splitter(weight_root='./Weights/imagenet/', layers=50, device='cuda:0')
client = client.eval()
server = server.eval()
client = client.to('cuda:0')
server = server.to('cuda:0')

accuracies = [0]*len(jpeg_folders_quality)
server_time = [0]*len(jpeg_folders_quality)

import Utils.utils as utils
from torchvision import transforms
from PIL import Image
mean, std = utils.image_transform('imagenet')
normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
with torch.no_grad():
    for i, img_folder in tqdm(enumerate(img_folders)):
        img_list = [str(x) + '.jpg' for x in range(100)]
        for j, img in enumerate(img_list):
            image_path = img_folder + img
            image = Image.open(image_path).convert('RGB')
            if dataset == 'cifar-10':
                image = image.resize((32, 32))
            elif dataset == 'imagenet-20':
                image = image.resize((224, 224))
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = normal(image)
            # transform using mean and std
            image = image.unsqueeze(0)
            image = image.to('cuda:0')
            # forward
            stime = time.time()
            output = client(image).detach()
            output = server(output)
            etime = time.time()
            server_time[i] += etime - stime
            # get the accuracy
            _, predicted = torch.max(output, 1)
            if str(predicted.item()) == label[j]:
                accuracies[i] += 1
        accuracies[i] = accuracies[i] / len(img_list)
        server_time[i] = server_time[i] / len(img_list)*1000

# change to 2 dicimal

accuracies = [round(x, 2) for x in accuracies]
server_time = [round(x, 2) for x in server_time]
print('accuracies ', accuracies)
print('server_time ', server_time)