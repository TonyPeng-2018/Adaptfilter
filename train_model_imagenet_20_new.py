from Models import mobilenetv2, resnet
import sys

# client, server = mobilenetv2.mobilenetv2_splitter(num_classes=20,
#                                                   weight_root='Weights/imagenet/',
#                                                   device='cuda:0',partition=-1)

model_type = sys.argv[1]
if 'mobilenet' in model_type:
    model = mobilenetv2.MobileNetV2(num_classes=20)
elif 'resnet' in model_type:
    model = resnet.resnet50(num_classes=20)

import datetime
model_time = datetime.datetime.now().strftime("%m%d%H%M%S")

# last_channel = server.last_channel

# from Models import last_classifier
# replace_layer = last_classifier.Last_classifier(last_channel, 20)

from Dataloaders import dataloader_image_20_new

train, _, val = dataloader_image_20_new.Dataloader_imagenet_20_integrated(train_batch=128, test_batch=64)
import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')
# client = client.to(device)
# server = server.to(device)
# client = client.train()
# server = server.train()

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer_client = optim.adam(client.parameters(), lr=0.001)
# optimizer_server = optim.adam(server.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

from tqdm import tqdm
from Utils import utils

import sys
import numpy as np
import os

epochs = 100
max_val_acc = 0

if f'Weights/training/{model_type}_{model_time}/' not in os.listdir('Weights/training/'):
    os.mkdir(f'Weights/training/{model_type}_{model_time}/')
        
for epoch in range(epochs):
    train_loss = 0.0
    model.train()
    for i, (data, labels) in tqdm(enumerate(train)):

        data, labels = data.to(device), labels['label'].to(device)

        pred = model(data)

        optimizer.zero_grad()
        loss = criterion(pred, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('train loss: ', train_loss)

    val_acc = 0

    for i, (data, labels) in tqdm(enumerate(val)):
        data, labels = data.to(device), labels['label'].to(device)
        pred = model(data)
        # get the number of 0 and 1
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        accuracy = torch.eq(pred, labels).float()

        # print the rate of gate exit
        val_acc += accuracy.mean().item()
    print('val_acc: ', val_acc)

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }, f'Weights/training/{model_type}_{model_time}/epoch-{epoch}-train-loss-{train_loss}-acc-{val_acc}.pth')