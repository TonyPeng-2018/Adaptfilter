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
from torchvision import transforms
import torchsummary

m_s_mobile = [1,2,4,8,16]
m_s_resnet = [1,2,4,8,16,32]

m_sizes = {'mobile': m_s_mobile, 'resnet': m_s_resnet}
r_sizes = {'cifar-10': (32,32), 'imagenet': (224,224), 'ccpd': (224,224)}
r_rates = {'mobile': 2, 'resnet': 4}
classes = {'cifar-10': 10, 'imagenet': 1000, 'ccpd': 34}
weight_root = {'cifar-10': './Weights/cifar-10/', 
               'imagenet': './Weights/imagenet/', 
               'ccpd': './Weights/ccpd/'}

# if sys args > 1
dataset = sys.argv[1]
model = sys.argv[2]
confidence = float(sys.argv[3])
weight = weight_root[dataset]
i_stop = 100

width, height = r_sizes[dataset][0]//r_rates[model], \
                r_sizes[dataset][1]//r_rates[model]
middle_size = m_sizes[model]
if model == 'resnet':
    client = resnet.resnet_splitter_client(num_classes=classes[dataset], weight_root=weight+'/', device='cpu', layers=50)
if model == 'mobile':
    client = mobilenetv2.mobilenetv2_splitter_client(num_classes=classes[dataset], weight_root=weight+'/', device='cpu')

middle_models = []
if model == 'resnet':
    for i in range(len(middle_size)):
        m_model = resnet.resnet_middle(middle=middle_size[i])
        # load weights
        m_model.load_state_dict(torch.load(weight+'middle/'+model+'_'+dataset+'_'+'middle_'+str(middle_size[i])+'.pth',
                                map_location=torch.device('cpu')))
        middle_models.append(m_model)

if model == 'mobile':
    for i in range(len(middle_size)):
        m_model = mobilenetv2.MobileNetV2_middle(middle=middle_size[i])
        # load weights
        m_model.load_state_dict(torch.load(weight+'middle/'+model+'_'+dataset+'_'+'middle_'+str(middle_size[i])+'.pth',
                                map_location=torch.device('cpu')))
        middle_models.append(m_model)

gate_models = []
if model == 'resnet':
    for i in range(len(middle_size)):
        g_model = gatedmodel.ExitGate(in_planes=middle_size[i],
                                           height = height, width=width)
        # load weights
        g_model.load_state_dict(torch.load(weight+'gate/'+model+'_'+dataset+'_'+'gate_'+str(middle_size[i])+'.pth',
                                map_location=torch.device('cpu')))
        gate_models.append(g_model)

if model == 'mobile':
    for i in range(len(middle_size)):
        g_model = gatedmodel.ExitGate(in_planes=middle_size[i],
                                           height = height, width=width)
        # load weights
        g_model.load_state_dict(torch.load(weight+'gate/'+model+'_'+dataset+'_'+'gate_'+str(middle_size[i])+'.pth',
                                map_location=torch.device('cpu')))
        gate_models.append(g_model)

# eval
client.eval()
for i in range(len(middle_size)):
    middle_models[i].eval()
    gate_models[i].eval()

# quantize
client = torch.ao.quantization.quantize_dynamic(client, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
for i in range(len(middle_size)):
    middle_models[i] = torch.ao.quantization.quantize_dynamic(middle_models[i], {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    gate_models[i] = torch.ao.quantization.quantize_dynamic(gate_models[i], {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

# 2. dataset
# directly read bmp image from the storage
if dataset == 'imagenet':
    data_set = 'imagenet-20'
else:
    data_set = dataset
data_root = '../data/'+data_set+'-client/'
n_images = 100
images_list = [data_root + str(x)+'.bmp' for x in range(n_images)]

client_time = 0

frequency = np.zeros(len(middle_size)+1)
mean, std = utils.image_transform(dataset)
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize((r_sizes[dataset][0], r_sizes[dataset][1])),
        ])

with torch.no_grad():
    for i, i_path in tqdm(enumerate(images_list)):
        if i >= i_stop:
            break
        image = cv2.imread(i_path, cv2.IMREAD_COLOR)
        image = image.astype(np.float32)/255.0
        image = transform(image)
        image = image.unsqueeze(0)
        
        s_time = time.time()
        client_out = client(image).detach()
        for j in range(len(middle_size)):
            middle_in = middle_models[j].in_layer(client_out)
            gate_out = gate_models[j](middle_in)
            
            if gate_out.max() > confidence:
                middle_int = utils.float_to_uint(middle_in)
                middle_int = middle_int.numpy().copy(order='C')
                middle_int = middle_int.astype(np.uint8)
                send_in = base64.b64encode(middle_int)
                frequency[j] += 1
                break
        if j == len(middle_size):
            frequency[j] += 1
        s1_time = time.time()
        client_time += s1_time - s_time

client_time = client_time * 1000 / 100

# print the list without [ and ]
out_string = str(client_time).replace('[','').replace(']','')
print(dataset, model)
print(out_string)
# print numpy frequency with comma
frequency = str(frequency.astype(int)).replace('[','').replace(']','').replace('  ',',')
print(frequency)