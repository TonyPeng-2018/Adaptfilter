from Models import mobilenetv2, resnet, last_classifier, encoder, decoder, upsampler, downsampler, gatedmodel
import sys
import torch

model_type = sys.argv[1]

device = torch.device('cuda:0')

if 'mobilenet' in model_type:
    client, server = mobilenetv2.mobilenetv2_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0',partition=-1)
    client_weight = f'Weights/imagenet-new/used/client/mobilenetv2.pth'
    server_weight = f'Weights/imagenet-new/used/server/mobilenetv2.pth'
    classifier_weight = f'Weights/imagenet-new/used/lastlayer/mobilenet.pth'

elif 'resnet' in model_type:
    client, server = resnet.resnet_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0', layers=50)
    client_weight = f'Weights/imagenet-new/used/client/resnet50.pth'
    server_weight = f'Weights/imagenet-new/used/server/resnet50.pth'
    classifier_weight = f'Weights/imagenet-new/used/lastlayer/resnet.pth'

classifier = last_classifier.last_layer_classifier(1000, 20)


client.load_state_dict(torch.load(client_weight))
server.load_state_dict(torch.load(server_weight))
classifier.load_state_dict(torch.load(classifier_weight)['model'])

from Dataloaders import dataloader_image_20_new

from tqdm import tqdm
from Utils import utils

import sys
import numpy as np
import os
import time
import base64
import gzip

client = client.to(device)
server = server.to(device)
classifier = classifier.to(device)

client.eval()
server.eval()

classifier.eval()
    
jpeg_accuracy = 0
all_jpeg_accuracy = []
for quality in [1]:
    test = dataloader_image_20_new.Dataloader_imagenet_20_integrated(train_batch=1, test_batch=1, test_only=True, JPEG=quality)
    for i, (data, labels) in tqdm(enumerate(test)):

        data, labels = data.to(device), labels['label'].to(device)
        output = client(data).detach()
        pred = server(output)
        pred = classifier(pred)
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        jpeg_accuracy += torch.sum(pred == labels).item()
    all_jpeg_accuracy.append(jpeg_accuracy/len(test.dataset))

    jpeg_accuracy = jpeg_accuracy/len(test.dataset)
    print('quality accuracy: ', jpeg_accuracy)
print('all_jpeg_accuracy: ', all_jpeg_accuracy)