from Models import mobilenetv2, resnet, last_classifier, encoder, decoder, upsampler, downsampler, gatedmodel
import sys
import torch

model_type = sys.argv[1]
num_of_layers = int(sys.argv[2]) # 2 for mobilenet, 1 for resnet

if 'mobilenet' in model_type:
    in_ch = 32
    n_ch = 5
    img_h, img_w = 112, 112
    client, _ = mobilenetv2.mobilenetv2_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0',partition=-1)
    down = downsampler.Downsampler(in_ch=in_ch, num_of_layers=num_of_layers)

elif 'resnet' in model_type:
    in_ch = 64
    n_ch = 6
    img_h, img_w = 56, 56
    client, _ = resnet.resnet_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0', layers=50)
    down = downsampler.Downsampler(in_ch=in_ch, num_of_layers=num_of_layers)

encs = []
gates = []
img_h, img_w = img_h//(2**num_of_layers), img_w//(2**num_of_layers)
# print(img_h, img_w)
for i in range(n_ch+num_of_layers):
    enc = encoder.Encoder(in_ch=in_ch*(2**num_of_layers), out_ch=(2**i))
    gate = gatedmodel.ExitGate(in_planes=2**i, height=img_h, width=img_w)
    encs.append(enc.eval())
    gates.append(gate.eval())

import datetime

from Dataloaders import dataloader_image_20_new

import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from Utils import utils

import sys
import numpy as np
import os
import time
import base64
import gzip

client.eval()
down.eval()

gate_time = np.zeros((10, n_ch+num_of_layers+1))
sizes = np.zeros((10, n_ch+num_of_layers+1))

for k in range(10):
    data = torch.randn(1, 3, 224, 224)
    for i in tqdm(range(n_ch+num_of_layers)):

        t1 = time.time()
        output = client(data)
        down_output = down(output)
        for j in range(i+1):
            
            output = encs[j](down_output)
            gate_score = gates[j](output)

        output = utils.float_to_uint(output)
        output = output.numpy().copy(order="C")
        output = output.astype(np.uint8)
        output = output.tobytes()
        
        output = gzip.compress(output, compresslevel=9)
        send_in = base64.b64encode(output)
        t2 = time.time()
        gate_time[k, i] = t2-t1
        sizes[k, i] = sys.getsizeof(send_in)

    # send raw
    t1 = time.time()
    output = client(data)
    output_down = down(output)
    for j in range(n_ch+num_of_layers):
        output = encs[j](down_output)
        gate_score = gates[j](output)

    output = utils.float_to_uint(output_down)
    output = output.numpy().copy(order="C")
    output = output.astype(np.uint8)
    output = output.tobytes()
    output = gzip.compress(output, compresslevel=9)
    send_in = base64.b64encode(output)
    t2 = time.time()
    gate_time[k, -1] = t2-t1
    sizes[k, -1] = sys.getsizeof(send_in)
gt_m = str(np.median(gate_time*1000, axis=0)).replace('[', '').replace(']', '')
gs_m = str(np.median(sizes, axis=0)).replace('[', '').replace(']', '').replace(' ', '')
print(gt_m)
print(gs_m)