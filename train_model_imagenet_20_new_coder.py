from Models import mobilenetv2, resnet, last_classifier, encoder, decoder, upsampler, downsampler
import sys
import torch

model_type = sys.argv[1]
num_of_layers = int(sys.argv[2]) # 2 for mobilenet, 1 for resnet
num_of_ch = int(sys.argv[3])

if 'mobilenet' in model_type:
    client, server = mobilenetv2.mobilenetv2_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0',partition=-1)
    all_weights = f'Weights/imagenet-new/pretrained/mobilenet_{num_of_layers}.pth'
    in_ch = 32
elif 'resnet' in model_type:
    client, server = resnet.resnet_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0', layers=50)
    all_weights = f'Weights/imagenet-new/pretrained/resnet_{num_of_layers}.pth'
    in_ch = 64
classifier = last_classifier.last_layer_classifier(1000, 20)
if 'mobilenet' in model_type:
    down = downsampler.Downsampler(in_ch=32, num_of_layers=num_of_layers)
    up = upsampler.Upsampler(in_ch=32*2**num_of_layers, num_of_layers=num_of_layers)
elif 'resnet' in model_type:
    down = downsampler.Downsampler(in_ch=64, num_of_layers=num_of_layers)
    up = upsampler.Upsampler(in_ch=64*2**num_of_layers, num_of_layers=num_of_layers)
    

checkpoint = torch.load(all_weights)
print(checkpoint.keys())
client.load_state_dict(checkpoint['client'])
server.load_state_dict(checkpoint['server'])
up.load_state_dict(checkpoint['upsampler'])
down.load_state_dict(checkpoint['downsampler'])
classifier.load_state_dict(checkpoint['new_classifier'])

import datetime
model_time = datetime.datetime.now().strftime("%m%d%H%M%S")

# last_channel = server.last_channel

# from Models import last_classifier
# replace_layer = last_classifier.Last_classifier(last_channel, 20)

from Dataloaders import dataloader_image_20_new

train, test, val = dataloader_image_20_new.Dataloader_imagenet_20_integrated(train_batch=128, test_batch=64)

import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')

client = client.to(device)
server = server.to(device)
classifier = classifier.to(device)
down = down.to(device)
up = up.to(device)

# freeze model parameters
for param in client.parameters():
    param.requires_grad = False
for param in server.parameters():
    param.requires_grad = False
for param in down.parameters():
    param.requires_grad = False
for param in up.parameters():
    param.requires_grad = False
for param in classifier.parameters():
    param.requires_grad = False

enc = encoder.Encoder(in_ch=in_ch*(2**num_of_layers), out_ch=num_of_ch)
dec = decoder.Decoder(in_ch=num_of_ch, out_ch=in_ch*(2**num_of_layers))

enc = enc.to(device)
dec = dec.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.001)

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
    classifier.train()
    down.train()
    up.train()
    enc.train()
    dec.train()

    # for i, (data, labels) in tqdm(enumerate(train)):

    #     data, labels = data.to(device), labels['label'].to(device)

    #     output = client(data)
    #     output = down(output)
    #     output = enc(output)
    #     output = dec(output)
    #     output = up(output)
    #     pred = server(output)
    #     pred = classifier(pred)

    #     optimizer.zero_grad()
    #     loss = criterion(pred, labels)
    #     train_loss += loss.item()
    #     loss.backward()
    #     optimizer.step()
    # train_loss = train_loss/len(train.dataset)
    # print('train loss: ', train_loss)

    val_acc = 0

    client.eval()
    server.eval()
    classifier.eval()
    down.eval()
    up.eval()
    enc.eval()
    dec.eval()

    for i, (data, labels) in tqdm(enumerate(test)):

        data, labels = data.to(device), labels['label'].to(device)

        output = client(data)
        output = down(output)
        # output = enc(output)
        # output = dec(output)
        output = up(output)
        output = server(output)
        pred = classifier(output)

        # get the number of 0 and 1
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        accuracy = torch.eq(pred, labels).float()

        # print the rate of gate exit
        val_acc += accuracy.sum().item()
    val_acc = val_acc/len(test.dataset)
    print('val acc: ', val_acc)

    # if val_acc > max_val_acc:
    #     max_val_acc = val_acc
    #     torch.save({
    #         'encoder': enc.state_dict(),
    #         'decoder': dec.state_dict(), 
    #         'optimizer': optimizer.state_dict(),
    #         'epoch': epoch,
    #         'val_acc': val_acc
    #     }, f'Weights/training/{model_type}_coder_{num_of_layers}_{model_time}/coder_best_model.pth')
    #     print('model saved' + ' train loss ', train_loss, ' val acc ', val_acc)
