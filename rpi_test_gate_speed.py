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

middle_sizes_mobile = [1, 2, 4, 8, 16]
middle_sizes_resnet = [1, 2, 4, 8, 16, 32]

middle_sizes = {"mobilenet": middle_sizes_mobile, "resnet": middle_sizes_resnet}
reduced_sizes = {"cifar-10": (32, 32), "imagenet": (224, 224), "ccpd": (224, 224)}
reduced_rates = {"mobilenet": 2, "resnet": 4}

dataset = "ccpd"
model = "resnet"
i_stop = 100

width, height = (
    reduced_sizes[dataset][0] / reduced_rates[model],
    reduced_sizes[dataset][1] / reduced_rates[model],
)
middle_size = middle_sizes[model]

# client include client, middle and gate

# client = mobilenetv2.mobilenetv2_splitter_client(
#     num_classes = 10,
#     weight_root='./Weights/'+dataset+'/',
#     device='cpu')
client = resnet.resnet_splitter_client(
    num_classes=34,
    weight_root="./Weights/" + "ccpd-small" + "/",
    device="cpu",
    layers=50,
)

middle_models = []
for i in range(len(middle_size)):
    # middle_models.append(mobilenetv2.MobileNetV2_middle(middle=middle_size[i]))
    middle_models.append(resnet.resnet_middle(middle=middle_size[i]))

gate_models = []
for i in range(len(middle_size)):
    gate_models.append(
        gatedmodel.ExitGate(in_planes=middle_size[i], height=height, width=width)
    )

# eval
client.eval()
for i in range(len(middle_size)):
    middle_models[i].eval()
    gate_models[i].eval()

# quantize
client = torch.ao.quantization.quantize_dynamic(
    client, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)
for i in range(len(middle_size)):
    middle_models[i] = torch.ao.quantization.quantize_dynamic(
        middle_models[i], {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    gate_models[i] = torch.ao.quantization.quantize_dynamic(
        gate_models[i], {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

# 2. dataset
# directly read bmp image from the storage
data_root = "../data/" + dataset + "-client/"
images_list = os.listdir(data_root)
images_list.remove("labels.txt")
# remove ending with jpg
images_list = [x for x in images_list if x.endswith(".bmp")]
images_list = sorted(images_list)

gate_time = [0]* (len(middle_size)+1)

# this is test the overspeed, so we don't need to load the models
with torch.no_grad():
    for i, i_path in tqdm(enumerate(images_list)):
        if i >= i_stop:
            break

        image_path = data_root + i_path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image)
        image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)

        with torch.no_grad():
            client_out = client(image).detach()
            s_time = time.time()
            for j in range(len(middle_size)):
                middle_in = middle_models[j].in_layer(client_out)
                gate_out = gate_models[j](middle_in)
                middle_int = utils.float_to_uint(middle_in)
                middle_int = middle_int.numpy().copy(order="C")
                middle_int = middle_int.astype(np.uint8)
                send_in = base64.b64encode(middle_int)
                s1_time = time.time()
                gate_time[j] += s1_time - s_time
            gate_time[-1] += s1_time - s_time

gate_time = [x/i_stop*1000 for x in gate_time]

# print the list without [ and ]
out_string = str(gate_time).replace("[", "").replace("]", "")
print(dataset, model)
print(out_string)
