from Models import mobilenetv2, resnet, last_classifier, encoder, decoder, upsampler, downsampler
import sys
import torch

model_type = sys.argv[1]
num_of_layers = int(sys.argv[2]) # 2 for mobilenet, 1 for resnet
num_of_coders = int(sys.argv[3]) # 1 for pyramid, 2 for pyramid heavy

if 'mobilenet' in model_type:
    client, server = mobilenetv2.mobilenetv2_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0',partition=-1)
    all_weights = f'Weights/imagenet-new/pretrained/mobilenet_{num_of_layers}.pth'
    client_weight = f'Weights/imagenet-new/pretrained/used/client/mobilenetv2.pth'
    server_weight = f'Weights/imagenet-new/pretrained/used/server/mobilenetv2.pth'
    class_weight = f'Weights/imagenet-new/pretrained/used/lastlayer/mobilenet.pth'
    in_ch = 32
elif 'resnet' in model_type:
    client, server = resnet.resnet_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0', layers=50)
    all_weights = f'Weights/imagenet-new/pretrained/resnet_{num_of_layers}.pth'
    client_weight = f'Weights/imagenet-new/pretrained/used/client/resnet50.pth'
    server_weight = f'Weights/imagenet-new/pretrained/used/server/resnet50.pth'
    class_weight = f'Weights/imagenet-new/pretrained/used/lastlayer/resnet.pth'
    in_ch = 64
classifier = last_classifier.last_layer_classifier(1000, 20)

if 'mobilenet' in model_type:
    down = downsampler.Downsampler(in_ch=32, num_of_layers=num_of_layers)
    up = upsampler.Upsampler(in_ch=32*2**num_of_layers, num_of_layers=num_of_layers)
elif 'resnet' in model_type:
    down = downsampler.Downsampler(in_ch=64, num_of_layers=num_of_layers)
    up = upsampler.Upsampler(in_ch=64*2**num_of_layers, num_of_layers=num_of_layers)
    

checkpoint = torch.load(all_weights)
# print(checkpoint.keys())
client.load_state_dict(checkpoint['client'])
server.load_state_dict(checkpoint['server'])
up.load_state_dict(checkpoint['upsampler'])
down.load_state_dict(checkpoint['downsampler'])
classifier.load_state_dict(checkpoint['new_classifier'])
# client.load_state_dict(torch.load(client_weight))
# server.load_state_dict(torch.load(server_weight))
# classifier.load_state_dict(torch.load(class_weight))

import datetime
model_time = datetime.datetime.now().strftime("%m%d%H%M%S")

# last_channel = server.last_channel

# from Models import last_classifier
# replace_layer = last_classifier.Last_classifier(last_channel, 20)

from Dataloaders import dataloader_image_20_new

train, test, val = dataloader_image_20_new.Dataloader_imagenet_20_integrated(train_batch=64, test_batch=32)

import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')

client = client.to(device)
server = server.to(device)
classifier = classifier.to(device)
down = down.to(device)
up = up.to(device)

# freeze model parameters
# for param in client.parameters():
#     param.requires_grad = False
# for param in server.parameters():
#     param.requires_grad = False
# for param in down.parameters():
#     param.requires_grad = False
# for param in up.parameters():
#     param.requires_grad = False
# for param in classifier.parameters():
#     param.requires_grad = False 

if num_of_coders == 1:
    enc = encoder.Encoder_Pyramid(in_ch=in_ch*(2**num_of_layers), min_ch=1)
    dec = decoder.Decoder_Pyramid(out_ch=in_ch*(2**num_of_layers), min_ch=1)
elif num_of_coders == 2:
    enc = encoder.Encoder_Pyramid_Heavy(in_ch=in_ch*(2**num_of_layers), min_ch=1)
    dec = decoder.Decoder_Pyramid_Heavy(out_ch=in_ch*(2**num_of_layers), min_ch=1)

enc = enc.to(device)
dec = dec.to(device)

criterion = nn.CrossEntropyLoss()

optimizer1 = optim.Adam(list(client.parameters()) + list(server.parameters()) + list(classifier.parameters()), lr=0.00001)
optimizer2 = optim.Adam(list(down.parameters()) + list(up.parameters()) +list(enc.parameters()) + list(dec.parameters()), lr=0.001)

from tqdm import tqdm
from Utils import utils

import sys
import numpy as np
import os

epochs = 30
max_val_acc = 0
record_val_acc = np.zeros(int(np.log2(in_ch)+num_of_layers))

if not os.path.exists(f'Weights/training/{model_type}_coder_{num_of_layers}_{model_time}/'):
    os.mkdir(f'Weights/training/{model_type}_coder_{num_of_layers}_{model_time}/')
print('saving to: ', f'Weights/training/{model_type}_coder_{num_of_layers}_{model_time}/')

for epoch in range(epochs):
    train_loss = 0.0

    client.train()
    server.train()
    classifier.train()
    down.train()
    up.train()
    enc.train()
    dec.train()

    for i, (data, labels) in tqdm(enumerate(train)):

        data, labels = data.to(device), labels['label'].to(device)

        output = client(data)
        output = down(output)
        outputs = enc(output)

        losses = None
        for output in outputs:
            output = dec(output)
            output = up(output)
            pred = server(output)
            pred = classifier(pred)
            if losses is None:
                losses = criterion(pred, labels)
            else:
                losses += criterion(pred, labels)

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        train_loss += losses.item()
        losses.backward()
        optimizer1.step()
        optimizer2.step()
    train_loss = train_loss/len(train.dataset)
    print('train loss: ', train_loss)

    val_acc = np.zeros(int(np.log2(in_ch)+num_of_layers))

    client.eval()
    server.eval()
    classifier.eval()
    down.eval()
    up.eval()
    enc.eval()
    dec.eval()

    for i, (data, labels) in tqdm(enumerate(val)):

        data, labels = data.to(device), labels['label'].to(device)

        output = client(data)
        output = down(output)
        outputs = enc(output)
        accuracies = []
        for j, output in enumerate(outputs):
            
            output = dec(output)
            output = up(output)
            output = server(output)
            pred = classifier(output)

            # get the number of 0 and 1
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            accuracy = torch.eq(pred, labels).float()
            val_acc[j] += accuracy.sum().item()

        # print the rate of gate exit
    val_acc = val_acc/len(val.dataset)
    val_acc_avg = np.sum(val_acc)/len(val_acc)
    print('val acc avg: ', val_acc_avg, 'val acc: ', val_acc)

    if val_acc_avg > max_val_acc:
        max_val_acc = val_acc_avg
        torch.save({
            'client': client.state_dict(),
            'server': server.state_dict(),
            'new_classifier': classifier.state_dict(),
            'downsampler': down.state_dict(),
            'upsampler': up.state_dict(),
            'encoder': enc.state_dict(),
            'decoder': dec.state_dict(), 
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'epoch': epoch,
            'val_acc': max_val_acc
        }, f'Weights/training/{model_type}_coder_{num_of_layers}_{model_time}/{model_type}_coder_{num_of_layers}_best.pth')
        print('model saved' + ' train loss ', train_loss, ' val acc ', val_acc_avg, 'max acc', max_val_acc)
    # store the last model
    torch.save({
        'client': client.state_dict(),
        'server': server.state_dict(),
        'new_classifier': classifier.state_dict(),
        'downsampler': down.state_dict(),
        'upsampler': up.state_dict(),
        'encoder': enc.state_dict(),
        'decoder': dec.state_dict(), 
        'optimizer1': optimizer1.state_dict(),
        'optimizer2': optimizer2.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc_avg
    }, f'Weights/training/{model_type}_coder_{num_of_layers}_{model_time}/{model_type}_coder_{num_of_layers}_last.pth')
