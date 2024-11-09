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
from torchvision import transforms
import torchsummary
from PIL import Image
import gzip

m_s_mobile = [1, 2, 4, 8, 16]
m_s_resnet = [1, 2, 4, 8, 16, 32]

m_sizes = {"mobile": m_s_mobile, "resnet": m_s_resnet}
r_sizes = {"cifar-10": (32, 32), "imagenet": (224, 224), "ccpd": (224, 224)}
r_rates = {"mobile": 2, "resnet": 4}
classes = {"cifar-10": 10, "imagenet": 1000, "ccpd": 34}
weight_root = {
    "cifar-10": "./Weights/cifar-10/",
    "imagenet": "./Weights/imagenet/",
    "ccpd": "./Weights/ccpd/",
}

# if sys args > 1
dataset = sys.argv[1]
model = sys.argv[2]
data_set = dataset if dataset != "imagenet" else "imagenet-20"
weight = weight_root[dataset]
i_stop = 600

width, height = (
    r_sizes[dataset][0] // r_rates[model],
    r_sizes[dataset][1] // r_rates[model],
)
middle_size = m_sizes[model]

l_time = time.time()
if model == "resnet":
    client, server = resnet.resnet_splitter(
        num_classes=classes[dataset], weight_root=weight + "/", device="cpu", layers=50
    )
if model == "mobile":
    client, server = mobilenetv2.mobilenetv2_splitter(
        num_classes=classes[dataset], weight_root=weight + "/", device="cpu"
    )
l_time = time.time() - l_time

middle_models = []
middle_models2 = []
if model == "resnet":
    for i in range(len(middle_size)):
        m_model = resnet.resnet_middle(middle=middle_size[i])
        # load weights
        m_model.load_state_dict(
            torch.load(
                weight
                + "middle/"
                + model
                + "_"
                + dataset
                + "_"
                + "middle_"
                + str(middle_size[i])
                + ".pth"
            )
        )
        middle_models.append(m_model)

        m_model2 = resnet.resnet_middle(middle=middle_size[i])
        # load weights
        m_model2.load_state_dict(
            torch.load(
                weight
                + "middle/"
                + model
                + "_"
                + dataset
                + "_"
                + "middle_"
                + str(middle_size[i])
                + ".pth"
            )
        )
        middle_models2.append(m_model2)

if model == "mobile":
    for i in range(len(middle_size)):
        m_model = mobilenetv2.MobileNetV2_middle(middle=middle_size[i])
        # load weights
        m_model.load_state_dict(
            torch.load(
                weight
                + "middle/"
                + model
                + "_"
                + dataset
                + "_"
                + "middle_"
                + str(middle_size[i])
                + ".pth"
            )
        )
        middle_models.append(m_model)

        m_model2 = mobilenetv2.MobileNetV2_middle(middle=middle_size[i])
        # load weights
        m_model2.load_state_dict(
            torch.load(
                weight
                + "middle/"
                + model
                + "_"
                + dataset
                + "_"
                + "middle_"
                + str(middle_size[i])
                + ".pth"
            )
        )
        middle_models2.append(m_model2)

gate_models = []
if model == "resnet":
    for i in range(len(middle_size)):
        g_model = gatedmodel.ExitGate(
            in_planes=middle_size[i], height=height, width=width
        )
        # load weights
        g_model.load_state_dict(
            torch.load(
                weight
                + "gate/"
                + model
                + "_"
                + dataset
                + "_"
                + "gate_"
                + str(middle_size[i])
                + ".pth"
            )
        )
        gate_models.append(g_model)

if model == "mobile":
    for i in range(len(middle_size)):
        g_model = gatedmodel.ExitGate(
            in_planes=middle_size[i], height=height, width=width
        )
        # load weights
        g_model.load_state_dict(
            torch.load(
                weight
                + "gate/"
                + model
                + "_"
                + dataset
                + "_"
                + "gate_"
                + str(middle_size[i])
                + ".pth"
            )
        )
        gate_models.append(g_model)

# eval
client.eval()
for i in range(len(middle_size)):
    middle_models[i].eval()
    gate_models[i].eval()
    middle_models2[i].eval()
server.eval()

# cuda
l2_time = time.time()
for i in range(len(middle_size)):
    middle_models2[i].to("cuda:0")
server.to("cuda:0")
l2_time = time.time() - l2_time
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
data_root = "../data/" + data_set + "-raw-image/"
n_images = i_stop
images_list = [data_root + str(x) + ".bmp" for x in range(n_images)]

if dataset == "cifar-10":
    normal = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
elif dataset == "imagenet":
    normal = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
elif dataset == "ccpd":
    normal = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

gate_folder = "../data/" + data_set + "-" + model + "-gate-emb/"
if not os.path.exists(gate_folder):
    os.makedirs(gate_folder)


with open("../data/" + data_set + "-raw-image/labels.txt", "r") as f:
    labels = f.readlines()
    # remove '\n
    labels = [x.strip() for x in labels]
server_time = 0
accuracy = 0
frequency = np.zeros(len(middle_size) + 1)

# from Models import last_classifier
# last_class_model = last_classifier.last_layer_classifier(in_channel=1000, out_channel=20)
# last_class_model.load_state_dict(torch.load("./Weights/imagenet/last_layer_model.pth"))
# last_class_model.eval()
# last_class_model.to("cuda:0")
# import last_classifier_mapping as lcm
# last_layer_mapping = lcm.last_layer_mapping
# labels = [last_layer_mapping[int(x)] for x in labels]

with torch.no_grad():
    for i, i_path in tqdm(enumerate(images_list)):
        exit_flag = -1

        if i >= i_stop:
            break
        image = Image.open(i_path)
        image = normal(image)
        image = image.unsqueeze(0)

        client_out = client(image).detach()
        client_out = client_out.to("cuda:0")
        
        server_out = server(client_out)
        # if dataset == "imagenet":
        #     server_out = last_class_model(server_out)
        pred = torch.argmax(server_out, 1)
            
        if pred == int(labels[i]):
            accuracy += 1
        frequency[exit_flag] += 1

# print the list without [ and ]
print("dataset:", dataset, "model:", model)
# print numpy frequency with commas
frequency = frequency.astype(int)
frequency = frequency.tolist()
# remove [ and ] from the list
frequency = str(frequency).replace("[", "").replace("]", "")
print("frequency:", frequency)
print("accuracy:%.4f" % (accuracy / i_stop))

