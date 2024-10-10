# test all test data, without and with gate
import argparse
import base64
import cv2
import datetime
from Models import gatedmodel, mobilenetv2, resnet
from Dataloaders import dataloader_imagenet
import numpy as np
import os
import PIL
import psutil
import sys
import time
import torch
from tqdm import tqdm
from Utils import utils, encoder

confidence = [0.85]

_, test, _ = dataloader_imagenet.Dataloader_imagenet_integrated(test_batch=1)

middle_sizes = {"mobile": [1, 2, 4, 8, 16], "resnet": [1, 2, 4, 8, 16, 32]}
reduced_sizes = {"cifar-10": (32, 32), "imagenet": (224, 224)}
reduced_rates = {"mobile": 2, "resnet": 4}

dataset = "imagenet"
model = "resnet"

width, height = (
    reduced_sizes[dataset][0] / reduced_rates[model],
    reduced_sizes[dataset][1] / reduced_rates[model],
)
middle_size = middle_sizes[model]

# client include client, middle and gate
if model == "mobile":
    client, server = mobilenetv2.MobileNetV2_splitter(
        weight_root="./Weights/" + dataset + "/")
    m_models = []
    m_models2 = []
    for i in range(len(middle_size)):
        m_model = mobilenetv2.MobileNetV2_middle(middle=middle_size[i])
        m_model.load_state_dict(
            torch.load("./Weights/"+ dataset+ "/middle/"+ model+ "_"+ dataset
                + "_middle_"+ str(middle_size[i])+ ".pth"))
        m_models.append(m_model)
        m_model2 = mobilenetv2.MobileNetV2_middle(middle=middle_size[i])
        m_model2.load_state_dict(
            torch.load("./Weights/"+ dataset+ "/middle/"+ model+ "_"+ dataset
                + "_middle_"+ str(middle_size[i])+ ".pth"))
        m_models2.append(m_model2)
    

elif model == "resnet":
    client, server = resnet.resnet_splitter(
        weight_root="./Weights/" + dataset + "/", layers=50)
    m_models = []
    m_models2 = []
    for i in range(len(middle_size)):
        m_model=resnet.resnet_middle(middle=middle_size[i])
        m_model.load_state_dict(
            torch.load("./Weights/"+ dataset+ "/middle/"+ model+ "_"
                + dataset+ "_middle_"+ str(middle_size[i])+ ".pth"))
        m_models.append(m_model)
        m_model2=resnet.resnet_middle(middle=middle_size[i])
        m_model2.load_state_dict(
            torch.load("./Weights/"+ dataset+ "/middle/"+ model+ "_"
                + dataset+ "_middle_"+ str(middle_size[i])+ ".pth"))
        m_models2.append(m_model2)

g_models = []
for i in range(len(middle_size)):
    g_models.append(
        gatedmodel.ExitGate(in_planes=middle_size[i], height=height, width=width)
    )
    g_models[i].load_state_dict(
        torch.load("./Weights/"+ dataset+ "/gate/"+ model+ "_"
            + dataset+ "_gate_"+ str(middle_size[i])+ ".pth"))

server = server.to("cuda:0")
client = client.to("cuda:0")
for i in range(len(middle_size)):
    m_models[i] = m_models[i].to("cuda:0")
    m_models2[i] = m_models2[i].to("cuda:0")
    g_models[i] = g_models[i].to("cpu")  

    # eval
client.eval()
server.eval()
for i in range(len(middle_size)):
    m_models[i].eval()
    m_models2[i].eval()
    g_models[i].eval()  

# quantize
client = torch.ao.quantization.quantize_dynamic(
    client, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)
for i in range(len(middle_size)):
    m_models[i] = torch.ao.quantization.quantize_dynamic(
        m_models[i], {torch.nn.Conv2d}, dtype=torch.qint8
    )
    g_models[i] = torch.ao.quantization.quantize_dynamic(
        g_models[i], {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    # m_models2[i] = torch.ao.quantization.quantize_dynamic(m_models2[i], {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
# images_list = os.listdir(data_root)
# images_list.remove('labels.txt')
# # remove ending with jpg
# images_list = [x for x in images_list if x.endswith('.bmp')]
# images_list = sorted(images_list)

# this is test the overspeed, so we don't need to load the models
with torch.no_grad():
    # acc includes org, 1,2,4,8,16,32, gate
    accu = [0] * (len(middle_size) + 2)
    for i, data in tqdm(enumerate(test)):
        if i == 100:
            break
        image, label = data
        image = image.to("cuda:0")
        label = label.to("cuda:0")

        gate_exit_flag = -1
        client_out = client(image).detach()
        # org
        pred = server(client_out).detach()
        _, predicted = torch.max(pred, 1)
        accu[0] += torch.sum(predicted == label).item()
        # each middle
        for j in range(len(middle_size)):
            if j < 5:
                continue

            middle_in = m_models[j].in_layer(client_out).detach().clone()
            middle_in = middle_in.to("cpu")
            middle_in, mmin, mmax = utils.normalize_return(middle_in)
            middle_int = utils.float_to_uint(middle_in)
            middle_int = middle_int.numpy().copy(order="C")
            middle_int = middle_int.astype(np.uint8)
            middle_int = torch.from_numpy(middle_int).float()
            middle_int = middle_int / 255
            middle_int = utils.renormalize(middle_int, mmin, mmax)
            
            middle_int = middle_int.to("cuda:0")
            middle_int = middle_int.to(dtype=torch.float32)
            middle_int = m_models2[j].out_layer(middle_int)
            print(client_out[0][0][0])
            print(middle_int[0][0][0])
            sys.exit()
            output = server(middle_int).detach()
            _, predicted = torch.max(output, 1)
            accu[j + 1] += torch.sum(predicted == label).item()

        # for gate_confidence in confidence:
        #     for j in range(len(middle_size)):
        #         middle_in = m_models[j].in_layer(client_out).detach()
        #         middle_in = middle_in.to("cpu")
        #         gate_out = g_models[j](middle_in)
        #         if gate_out > gate_confidence:
        #             middle_in, mmin, mmax = utils.normalize_return(middle_in)
        #             middle_int = utils.float_to_uint(middle_in)
        #             middle_int = middle_int.cpu()
        #             middle_int = middle_int.numpy().copy(order="C")
        #             middle_int = middle_int.astype(np.uint8)
        #             send_in = base64.b64encode(middle_int)
        #             gate_exit_flag = j
        #             break
        #     if gate_exit_flag == -1:  # send all
        #         client_out, mmin, mmax = utils.normalize_return(client_out)
        #         middle_int = utils.float_to_uint(client_out)
        #         middle_int = middle_int.numpy().copy(order="C")
        #         middle_int = middle_int.astype(np.uint8)
        #         send_in = base64.b64encode(middle_int)

        #     # accuracy
        #     middle_int = torch.from_numpy(middle_int).float()
        #     middle_int = middle_int / 255
        #     middle_int = utils.renormalize(middle_int, mmin, mmax)
        #     middle_int = middle_int.to("cuda:0")
        #     middle_int = middle_int.to(dtype=torch.float32)
        #     if gate_exit_flag != -1:
        #         middle_int = m_models2[gate_exit_flag].out_layer(middle_int)
        #     output = server(middle_int)
        #     _, predicted = torch.max(output, 1)
        #     accu[-1] += torch.sum(predicted == label).item()

test_name_list = ["org", "1", "2", "4", "8", "16", "32", "gate"]
for i, acc in enumerate(accu):
    print(test_name_list[i], "accuracy:", acc / 100/400)






