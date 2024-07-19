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

gate_confidence = 0.85
batch_size = 60

middle_sizes = {'mobile': [1,2,4,8,16], 'resnet': [1,2,4,8,16,32]}
reduced_sizes = {'cifar-10': (32,32), 'imagenet': (224,224)}
reduced_rates = {'mobile': 2, 'resnet': 4}

dataset = 'imagenet'
model = 'resnet'
i_stop = 10

width, height = reduced_sizes[dataset][0]/reduced_rates[model], \
                reduced_sizes[dataset][1]/reduced_rates[model]
middle_size = middle_sizes[model]

# client include client, middle and gate
if model == 'mobile':
    client = mobilenetv2.mobilenetv2_splitter_client(
                        weight_root='./Weights/'+dataset+'/', 
                        device='cpu')
    middle_models = []
    for i in range(len(middle_size)):
        middle_models.append(mobilenetv2.MobileNetV2_middle(middle=middle_size[i]))
        middle_models[i].load_state_dict(torch.load
                                         ('./Weights/'+dataset+'/middle/'+model+'_'+dataset+
                                          '_middle_'+str(middle_size[i])+'.pth',map_location=torch.device('cpu'))
                                          )

elif model == 'resnet':
    client = resnet.resnet_splitter_client(weight_root='./Weights/'+dataset+'/', device='cpu', layers=50)

    middle_models = []
    for i in range(len(middle_size)):
        middle_models.append(resnet.resnet_middle(middle=middle_size[i]))
        middle_models[i].load_state_dict(torch.load
                                         ('./Weights/'+dataset+'/middle/'+model+'_'+dataset+
                                          '_middle_'+str(middle_size[i])+'.pth', map_location=torch.device('cpu'))
                                          )

gate_models = []
for i in range(len(middle_size)):
    gate_models.append(gatedmodel.ExitGate(in_planes=middle_size[i],
                                           height = height, width=width))
    gate_models[i].load_state_dict(torch.load('./Weights/'+dataset+'/gate/'+model+'_'+dataset+
                                             '_gate_'+str(middle_size[i])+'.pth', map_location=torch.device('cpu')))

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
if dataset == 'cifar-10':
    data_root = '../data/'+dataset+'-client/'
elif dataset == 'imagenet':
    data_root = '../data/'+dataset+'-20-client/'
images_list = os.listdir(data_root)
images_list.remove('labels.txt')
# remove ending with jpg
images_list = [x for x in images_list if x.endswith('.bmp')]
images_list = sorted(images_list)

client_time = [0] * batch_size * i_stop

gate_frequency = [0] * (len(middle_size) + 1)

gate_emb_folder = '../data/'+dataset+'-'+model+'-gate-emb/'
if not os.path.exists(gate_emb_folder):
    os.makedirs(gate_emb_folder)
f = open('rpi-gate-emb.txt', 'w')

import torchvision.transforms as transforms
from PIL import Image

if dataset == 'cifar-10':
    normal = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
elif dataset == 'imagenet':
    normal = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
# this is test the overspeed, so we don't need to load the models
with torch.no_grad():
    for i, i_path in tqdm(enumerate(images_list)):
        
        image_path = data_root + i_path
        image = Image.open(image_path).convert('RGB')

        s_time = time.time()
        image = normal(image)
        image = image.unsqueeze(0)
        gate_exit_flag = -1
        
        client_out = client(image).detach()
    
        for j in range(len(middle_size)):
            middle_in = middle_models[j].in_layer(client_out)
            gate_out = gate_models[j](middle_in)
            if gate_out > gate_confidence:
                middle_in, mmin, mmax = utils.normalize_return(middle_in)
                middle_int = utils.float_to_uint(middle_in)
                middle_int = middle_int.numpy().copy(order='C')
                middle_int = middle_int.astype(np.uint8)
                send_in = base64.b64encode(middle_int)
                gate_exit_flag = j
                break
        if gate_exit_flag == -1: # send all
            client_out, mmin, mmax = utils.normalize_return(client_out)
            middle_int = utils.float_to_uint(client_out)
            middle_int = middle_int.numpy().copy(order='C')
            middle_int = middle_int.astype(np.uint8)
            send_in = base64.b64encode(middle_int)
        
        s1_time = time.time()
        client_time[i] += s1_time - s_time
        gate_frequency[gate_exit_flag] += 1

        f2 = open(gate_emb_folder+i_path[:-4], 'wb')
        f2.write(send_in)
        f2.close()
        f2 = open(gate_emb_folder+i_path[:-4]+'_helper', 'w')
        f2.write(str(mmax.item()) + ',' + str(mmin.item()) + '\n')
        f.write('client_time[%d]: %f\n' % (j, client_time[j]))
    for j in range(len(gate_frequency)):
        f.write('gate_frequency[%d]: %d\n' % (j, gate_frequency[j]))
    avg_client_time = [0] * (len(client_time)//60)
    for i in range (0, len(client_time), 60):
        avg_client_time[i//60] = sum(client_time[i:i+60]) / 60 * 1000
        f.write('avg_client_time[%d]: %f\n' % (i, avg_client_time[i//60]))
f.close()

# print average time
print('avg_client_time:', avg_client_time)
print('gate_frequency:', gate_frequency)
