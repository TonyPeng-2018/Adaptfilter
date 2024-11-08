from Models import mobilenetv2, resnet, last_classifier, encoder, decoder
import sys
import torch

model_type = sys.argv[1]
num_of_layers = int(sys.argv[2]) # 2 for mobilenet, 1 for resnet

if 'mobilenet' in model_type:
    client, server = mobilenetv2.mobilenetv2_splitter(num_classes=1000,
                                                  weight_root='Weights/imagenet-new/',
                                                  device='cuda:0',partition=-1)
    new_classifier = last_classifier.last_layer_classifier(1000, 20)
    class_weight = torch.load('Weights/imagenet-new/lastlayer/resnet.pth')
    new_classifier.load_state_dict(class_weight['model'])

elif 'resnet' in model_type:
    client, server = resnet.resnet_splitter(num_classes=1000,
                                                  weight_root='Weights/imagenet-new/',
                                                  device='cuda:0', layers=50)
    new_classifier = last_classifier.last_layer_classifier(1000, 20)
    class_weight = torch.load('Weights/imagenet-new/lastlayer/mobilenet.pth')
    new_classifier.load_state_dict(class_weight['model'])

if 'mobilenet' in model_type:
    enc = encoder.Encoder(in_ch=32, num_of_layers=num_of_layers)
    dec = decoder.Decoder(in_ch=32*2**num_of_layers, num_of_layers=num_of_layers)
elif 'resnet' in model_type:
    enc = encoder.Encoder(in_ch=64, num_of_layers=num_of_layers)
    dec = decoder.Decoder(in_ch=64*2**num_of_layers, num_of_layers=num_of_layers)

import datetime
model_time = datetime.datetime.now().strftime("%m%d%H%M%S")

# last_channel = server.last_channel

# from Models import last_classifier
# replace_layer = last_classifier.Last_classifier(last_channel, 20)

from Dataloaders import dataloader_image_20_new

train, _, val = dataloader_image_20_new.Dataloader_imagenet_20_integrated(train_batch=64, test_batch=32)

import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')

client = client.to(device)
server = server.to(device)
new_classifier = new_classifier.to(device)
enc = enc.to(device)
dec = dec.to(device)

# freeze model parameters
# for param in client.parameters():
#     param.requires_grad = False
# for param in server.parameters():
#     param.requires_grad = False
# for param in new_classifier.parameters():
#     param.requires_grad = False

criterion = nn.CrossEntropyLoss()
# optimizer_client = optim.adam(client.parameters(), lr=0.001)
# optimizer_server = optim.adam(server.parameters(), lr=0.001)
# list_params = list(client.parameters()) + list(server.parameters()) + list(new_classifier.parameters()) + list(enc.parameters()) + list(dec.parameters())
list_params = list(client.parameters()) + list(server.parameters()) + list(new_classifier.parameters())
list_params_2 = list(enc.parameters()) + list(dec.parameters())
optimizer = optim.Adam(list_params, lr=0.0001)
optimizer2 = optim.Adam(list_params_2, lr=0.001)
# optimizer_enc = optim.Adam(enc.parameters(), lr=0.001)
# optimizer_dec = optim.Adam(dec.parameters(), lr=0.001)

from tqdm import tqdm
from Utils import utils

import sys
import numpy as np
import os

epochs = 100
max_val_acc = 0

if not os.path.exists(f'Weights/training/{model_type}_coder_{num_of_layers}_{model_time}/'):
    os.mkdir(f'Weights/training/{model_type}_coder_{num_of_layers}_{model_time}/')

for epoch in range(epochs):
    train_loss = 0.0

    client.train()
    server.train()
    new_classifier.train()
    enc.train()
    dec.train()

    for i, (data, labels) in tqdm(enumerate(train)):

        data, labels = data.to(device), labels['label'].to(device)

        output = client(data)
        output = enc(output)
        output = dec(output)
        pred = server(output)
        pred = new_classifier(pred)

        # optimizer_enc.zero_grad()
        # optimizer_dec.zero_grad()
        optimizer.zero_grad()
        optimizer2.zero_grad()
        loss = criterion(pred, labels)
        train_loss += loss.item()
        loss.backward()
        # optimizer_enc.step()
        # optimizer_dec.step()
        optimizer.step()
        optimizer2.step()
    print('train loss: ', train_loss/len(train.dataset))

    val_acc = 0

    client.eval()
    server.eval()
    new_classifier.eval()
    enc.eval()
    dec.eval()

    for i, (data, labels) in tqdm(enumerate(val)):

        data, labels = data.to(device), labels['label'].to(device)
        output = client(data)
        output = enc(output)
        output = dec(output)
        pred = server(output)
        pred = new_classifier(pred)

        # get the number of 0 and 1
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        accuracy = torch.eq(pred, labels).float()

        # print the rate of gate exit
        val_acc += accuracy.sum().item()
    val_acc = val_acc/len(val.dataset)
    print('val acc: ', val_acc)

    torch.save({
        'client': client.state_dict(),
        'server': server.state_dict(),
        'new_classifier': new_classifier.state_dict(),
        'enc': enc.state_dict(),
        'dec': dec.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }, f'Weights/training/{model_type}_coder_{num_of_layers}_{model_time}/encoder_epoch-{epoch}-train-loss-{train_loss}-acc-{val_acc}.pth')
