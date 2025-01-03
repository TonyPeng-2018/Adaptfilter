from Models import mobilenetv2, resnet, last_classifier
import sys
import torch

model_type = sys.argv[1]
if 'mobilenet' in model_type:
    client, server = mobilenetv2.mobilenetv2_splitter(num_classes=1000,
                                                  weight_root='Weights/imagenet-new',
                                                  device='cuda:0',partition=-1)
elif 'resnet' in model_type:
    client, server = resnet.resnet_splitter(num_classes=1000,
                                                  weight_root='Weights/imagenet-new/',
                                                  device='cuda:0', layers=50)
new_classifier = last_classifier.last_layer_classifier(1000, 20)

import datetime
model_time = datetime.datetime.now().strftime("%m%d%H%M%S")

# last_channel = server.last_channel

# from Models import last_classifier
# replace_layer = last_classifier.Last_classifier(last_channel, 20)

from Dataloaders import dataloader_image_20_new

train, _, val = dataloader_image_20_new.Dataloader_imagenet_20_integrated(train_batch=128, test_batch=64)

import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')

client = client.to(device)
server = server.to(device)
new_classifier = new_classifier.to(device)

# freeze model parameters
for param in client.parameters():
    param.requires_grad = False
for param in server.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(new_classifier.parameters(), lr=0.001)

from tqdm import tqdm
from Utils import utils

import sys
import numpy as np
import os

epochs = 100
max_val_acc = 0

if not os.path.exists(f'Weights/training/{model_type}_new_classifier_{model_time}/'):
    os.mkdir(f'Weights/training/{model_type}_new_classifier_{model_time}/')

for epoch in range(epochs):
    train_loss = 0.0

    client.train()
    server.train()
    new_classifier.train()

    for i, (data, labels) in tqdm(enumerate(train)):

        data, labels = data.to(device), labels['label'].to(device)

        client_out = client(data)
        server_out = server(client_out)
        pred = new_classifier(server_out)

        optimizer.zero_grad()
        loss = criterion(pred, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('train loss: ', train_loss/len(train.dataset))

    val_acc = 0

    client.eval()
    server.eval()
    new_classifier.eval()

    for i, (data, labels) in tqdm(enumerate(val)):

        data, labels = data.to(device), labels['label'].to(device)
        client_out = client(data)
        server_out = server(client_out)
        pred = new_classifier(server_out)
        # get the number of 0 and 1
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        accuracy = torch.eq(pred, labels).float()

        # print the rate of gate exit
        val_acc += accuracy.sum().item()
    val_acc = val_acc/len(val.dataset)
    print('val acc: ', val_acc)

    torch.save({
        'model': new_classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }, f'Weights/training/{model_type}_new_classifier_{model_time}/epoch-{epoch}-train-loss-{train_loss}-acc-{val_acc}.pth')
