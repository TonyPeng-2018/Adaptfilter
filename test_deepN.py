# this file is for testing the DeepN model. 

# load resnet and label

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
import Utils.utils as utils
from torchvision import transforms
from PIL import Image
import DeepN

i_stop = 100
dataset = sys.argv[1]
data_set = dataset if dataset != "imagenet" else "imagenet-20"
model = sys.argv[2]
classes = {"cifar-10": 10, "imagenet": 1000, "ccpd": 34}
weight_root = "./Weights/" + dataset + "/"

# 2. dataset
# directly read bmp image from the storage

data_root = "../data/" + data_set + "-raw-image/"
if data_set == "imagenet-20":
    data_root = "../data/" + data_set + "-raw-image-224/"
new_data_root = "../data/" + data_set + "-raw-image-jpg/"
if not os.path.exists(new_data_root):
    os.makedirs(new_data_root)
label = open(data_root + "labels.txt", "r")
label = label.read()
label = label.split("\n")

jpeg_folders_quality = [60]
if model == "resnet":
    client, server = resnet.resnet_splitter(
        weight_root="./Weights/" + dataset + "/", layers=50, device="cuda:0", num_classes=classes[dataset]
    )
elif model == "mobile":
    client, server = mobilenetv2.mobilenetv2_splitter(
        weight_root="./Weights/" + dataset + "/", device="cuda:0", num_classes=classes[dataset]
    )
client = client.eval()
server = server.eval()
client = client.to("cuda:0")
server = server.to("cuda:0")

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

accuracies = [0] * len(jpeg_folders_quality)
server_time = [0] * len(jpeg_folders_quality)
accuracies_2 = [0] * len(jpeg_folders_quality)
accuracies_3 = [0] * len(jpeg_folders_quality)

with torch.no_grad():
    for i, quality in tqdm(enumerate(jpeg_folders_quality)):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        for j in tqdm(range(i_stop)):
            image_path = data_root + str(j) + ".bmp"
            new_image_path = new_data_root + str(j) + ".jpg"
            new_image_path_2 = new_data_root + str(j) + "_2.jpg"
            new_image_path_3 = new_data_root + str(j) + "_3.jpg"
            qtable_path = new_data_root + str(j)+"_qt.txt"
            # create a qtable for specific image. 
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img)
            qtable = DeepN.Get_Qtable(img)
            if not os.path.exists(qtable_path):
                np.savetxt(qtable_path, qtable, fmt="%d")
            # if not os.path.exists(new_image_path):
            os.system("cjpeg -dct int -qtable "+qtable_path+" -baseline -quality " + str(quality) + " -outfile " + new_image_path + " " + image_path)
            os.system("cjpeg -quality " + str(quality) + " -outfile " + new_image_path_2 + " " + image_path)
            # cv2 compress
        # deepN JPEG compression
            image = cv2.imread(new_image_path, qtable)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            # show image            
            image = normal(image)
            # print image
            # transform using mean and std
            image = image.unsqueeze(0)
            image = image.to("cuda:0")
            # forward
            output = client(image).detach()
            output = server(output)
            # get the accuracy
            _, predicted = torch.max(output, 1)
            if str(predicted.item()) == label[j]:
                accuracies[i] += 1

        # cjpeg compression
            image = cv2.imread(new_image_path_2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            # show image
            image = normal(image)
            # print image
            # transform using mean and std
            image = image.unsqueeze(0)
            image = image.to("cuda:0")
            # forward
            output = client(image).detach()
            output = server(output)
            # get the accuracy
            _, predicted = torch.max(output, 1)
            if str(predicted.item()) == label[j]:
                accuracies_2[i] += 1
        # cv2 jpeg compression
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, image = cv2.imencode(".jpg", img, encode_param)
            image = cv2.imdecode(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # store image
            image = Image.fromarray(image)
            image.save(new_image_path_3)
            # show image
            image = normal(image)
            # print image
            # transform using mean and std
            image = image.unsqueeze(0)
            image = image.to("cuda:0")
            # forward
            output = client(image).detach()
            output = server(output)
            # get the accuracy
            _, predicted = torch.max(output, 1)
            if str(predicted.item()) == label[j]:
                accuracies_3[i] += 1

        accuracies[i] = accuracies[i] / i_stop
        server_time[i] = server_time[i] / i_stop * 1000
        accuracies_2[i] = accuracies_2[i] / i_stop
        accuracies_3[i] = accuracies_3[i] / i_stop

# change to 2 dicimal

accuracies = [round(x, 4) for x in accuracies]
server_time = [round(x, 4) for x in server_time]
accuracies_2 = [round(x, 4) for x in accuracies_2]
accuracies_3 = [round(x, 4) for x in accuracies_3]
print("accuracies ", accuracies)
print("accuracies_2 ", accuracies_2)
print("accuracies_3 ", accuracies_3)
print("server_time ", server_time)

