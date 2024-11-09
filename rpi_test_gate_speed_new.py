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

classifier = last_classifier.last_layer_classifier(1000, 20)

encs = []
gates = []
img_h, img_w = img_h//(2**num_of_layers), img_w//(2**num_of_layers)
for i in range(n_ch+num_of_layers):
    enc = encoder.Encoder(in_ch=in_ch*(2**num_of_layers), out_ch=(i**2))
    gate = gatedmodel.ExitGate(in_planes=i*2, height=img_h, width=img_w)
    encs.append(enc)
    gates.append(gate)

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
classifier.eval()
down.eval()
enc.eval()
gate.eval()

gate_time = np.zeros(n_ch+num_of_layers+1)

data = torch.randn(1, 2**(n_ch+num_of_layers), img_h, img_w).cuda()

for i in tqdm(range(n_ch+num_of_layers)):

    t1 = time.time()
    output = client(data)
    output = down(output)
    for j in range(i):
        
        output = enc(output)
        gate_score = gate(output)

    output = utils.float_to_uint(output)
    output = output.numpy().copy(order="C")
    output = output.astype(np.uint8)
    output = output.tobytes()
    output = gzip.compress(output, compresslevel=9)
    send_in = base64.b64encode(output)
    t2 = time.time()
    gate_time[i] = t2-t1

# send raw
t1 = time.time()
output = client(data)
output_down = down(output)
for j in range(n_ch+num_of_layers):
    output = encs[j](output)
    gate_score = gates[j](output)

output = utils.float_to_uint(output_down)
output = output.numpy().copy(order="C")
output = output.astype(np.uint8)
output = output.tobytes()
output = gzip.compress(output, compresslevel=9)
send_in = base64.b64encode(output)
t2 = time.time()
gate_time[-1] = t2-t1

print(gate_time)