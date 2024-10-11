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

gate_confidences = [0.55, 0.65, 0.75, 0.85, 0.95, 0.99]
for g_conf in gate_confidences:
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
    if model == "resnet":
        client, server = resnet.resnet_splitter(
            num_classes=classes[dataset], weight_root=weight + "/", device="cpu", layers=50
        )
    if model == "mobile":
        client, server = mobilenetv2.mobilenetv2_splitter(
            num_classes=classes[dataset], weight_root=weight + "/", device="cpu"
        )
    del(server)
    middle_models = []
    if model == "resnet":
        for i in range(len(middle_size)):
            m_model = resnet.resnet_middle(middle=middle_size[i])
            # load weights
            m_model.load_state_dict(
                torch.load(weight+ "middle/"+ model+ "_"+ dataset+ "_"+ "middle_"+ str(middle_size[i])+ ".pth",
                map_location=torch.device("cpu"))
            )
            middle_models.append(m_model)

    if model == "mobile":
        for i in range(len(middle_size)):
            m_model = mobilenetv2.MobileNetV2_middle(middle=middle_size[i])
            # load weights
            m_model.load_state_dict(
                torch.load(weight+ "middle/"+ model+ "_"+ dataset+ "_"+ "middle_"+ str(middle_size[i])+ ".pth",
                map_location=torch.device("cpu"))
            )
            middle_models.append(m_model)

    gate_models = []
    for i in range(len(middle_size)):
        g_model = gatedmodel.ExitGate(
            in_planes=middle_size[i], height=height, width=width
        )
        # load weights
        g_model.load_state_dict(
            torch.load(weight+ "gate/"+ model+ "_"+ dataset+ "_"+ "gate_"+ str(middle_size[i])+ ".pth",
            map_location=torch.device("cpu"))
        )
        gate_models.append(g_model)

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
            middle_models[i], {torch.nn.Conv2d}, dtype=torch.qint8
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

    gate_folder = "../data/" + data_set + "-" + model + "-conf-" +str(g_conf)+ "/"
    if not os.path.exists(gate_folder):
        os.makedirs(gate_folder)

    with open("../data/" + data_set + "-raw-image/labels.txt", "r") as f:
        labels = f.readlines()

    frequency = np.zeros(len(middle_size) + 1)
    client_time = 0
    with torch.no_grad():
        for i, i_path in tqdm(enumerate(images_list)):
            exit_flag = -1

            if i >= i_stop:
                break
            image = Image.open(i_path).convert("RGB")
            image = normal(image)
            image = image.unsqueeze(0)

            c_time = time.time()
            client_out = client(image).detach()
            for j in range(len(middle_size)):
                middle_in = middle_models[j].in_layer(client_out)
                gate_out = gate_models[j](middle_in)

                if gate_out.max() > g_conf:
                    exit_flag = j

                if exit_flag > -1:
                    middle_in, mmin, mmax = utils.normalize_return(middle_in)
                    middle_int = utils.float_to_uint(middle_in)
                    middle_int = middle_int.numpy().copy(order="C")
                    middle_int = middle_int.astype(np.uint8)
                    # gzip
                    middle_int = middle_int.tobytes()
                    middle_int = gzip.compress(middle_int)
                    # print('middle_int:', middle_int)
                    send_msg = base64.b64encode(middle_int)
                    break

            if exit_flag == -1:
                client_out, mmin, mmax = utils.normalize_return(client_out)
                middle_int = utils.float_to_uint(client_out)
                middle_int = middle_int.numpy().copy(order="C")
                middle_int = middle_int.astype(np.uint8)
                middle_int = middle_int.tobytes()
                middle_int = gzip.compress(middle_int)
                send_msg = base64.b64encode(middle_int)
                # write the send_msg to the fodler
            client_time += time.time() - c_time
            frequency[exit_flag] += 1
            with open(gate_folder + str(i), "wb") as f:
                f.write(send_msg)
            with open(gate_folder + str(i) + "_h", "wb") as f:
                # dataset, model, max, min
                msg = data_set + "," + model + "," + str(mmax) + "," + str(mmin)
                msg = msg.encode()
                f.write(msg)

    # print the list without [ and ]
    print("dataset:", dataset, "model:", model, "confidence:", g_conf)
    # print numpy frequency with comma
    frequency = (
        str(frequency.astype(int)).replace("[", "").replace("]", "").replace("  ", ",")
    )
    print("frequency:", frequency)
    print("client_time: %.2f" % (client_time * 1000 / i_stop))
