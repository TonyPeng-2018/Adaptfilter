from Models import mobilenetv2, resnet, last_classifier, encoder, decoder, upsampler, downsampler, gatedmodel
import sys
import torch

model_type = sys.argv[1]
num_of_layers = 1 # use 1 for accuracy
num_of_encoder = 2 # 1 is normal, 2 is heavy
weighted = int(sys.argv[2]) # 1 for unweighted, 2 for weighted
num_of_ch = int(sys.argv[3]) 

if 'mobilenet' in model_type:
    in_ch = 32
    img_h, img_w = 112, 112
    client, server = mobilenetv2.mobilenetv2_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0',partition=-1)
    if weighted == 1:
        all_weights = f'Weights/imagenet-new/mobilenet_pyramid_coder_{num_of_layers}_{num_of_encoder}.pth'
    elif weighted == 2:
        all_weights = f'Weights/imagenet-new/mobilenet_pyramid_coder_weight_{num_of_layers}_{num_of_encoder}.pth'
    down = downsampler.Downsampler(in_ch=in_ch, num_of_layers=num_of_layers)
    up = upsampler.Upsampler(in_ch=in_ch*(2**num_of_layers), num_of_layers=num_of_layers)

elif 'resnet' in model_type:
    in_ch = 64
    img_h, img_w = 56, 56
    client, server = resnet.resnet_splitter(num_classes=1000,
                                                  weight_root=None,
                                                  device='cuda:0', layers=50)
    if weighted == 1:
        all_weights = f'Weights/imagenet-new/resnet_pyramid_coder_{num_of_layers}_{num_of_encoder}.pth'
    elif weighted == 2:
        all_weights = f'Weights/imagenet-new/resnet_pyramid_coder_weight_{num_of_layers}_{num_of_encoder}.pth'
    down = downsampler.Downsampler(in_ch=in_ch, num_of_layers=num_of_layers)
    up = upsampler.Upsampler(in_ch=in_ch*(2**num_of_layers), num_of_layers=num_of_layers)


classifier = last_classifier.last_layer_classifier(1000, 20)
if num_of_encoder == 1:
    enc = encoder.Encoder_Pyramid(in_ch=in_ch*(2**num_of_layers),min_ch=1)
    dec = decoder.Decoder_Pyramid(out_ch=in_ch*(2**num_of_layers),min_ch=1)
elif num_of_encoder == 2:
    enc = encoder.Encoder_Pyramid_Heavy(in_ch=in_ch*(2**num_of_layers),min_ch=1)
    dec = decoder.Decoder_Pyramid_Heavy(out_ch=in_ch*(2**num_of_layers),min_ch=1)

checkpoint = torch.load(all_weights)
client.load_state_dict(checkpoint['client'])
server.load_state_dict(checkpoint['server'])
up.load_state_dict(checkpoint['upsampler'])
down.load_state_dict(checkpoint['downsampler'])
classifier.load_state_dict(checkpoint['new_classifier'])
enc.load_state_dict(checkpoint['encoder'])
dec.load_state_dict(checkpoint['decoder'])

img_h, img_w = img_h//(2**num_of_layers), img_w//(2**num_of_layers)

import datetime
model_time = datetime.datetime.now().strftime("%m%d%H%M%S")

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
enc = enc.to(device)
dec = dec.to(device)

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
for param in enc.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False

gate = gatedmodel.ExitGate(in_planes=num_of_ch, height=img_h, width=img_w)
gate = gate.to(device)

# criterion = nn.BCELoss()
# criterion2 = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

optimizer = optim.Adam(gate.parameters(), lr=0.01)

from tqdm import tqdm
from Utils import utils

import sys
import numpy as np
import os

epochs = 30
max_val_acc = 0

if not os.path.exists(f'Weights/training/{model_type}_gate_{num_of_layers}_{model_time}/'):
    os.mkdir(f'Weights/training/{model_type}_gate_{num_of_layers}_{model_time}/') 
print('saving to: ', f'Weights/training/{model_type}_gate_{num_of_layers}_{model_time}/')

client.eval()
server.eval()
classifier.eval()
down.eval()
up.eval()
enc.eval()
dec.eval()

gate_ind = np.log2(num_of_ch).astype(int)
for epoch in range(epochs):
    train_loss = 0.0
    gate.train()

    for i, (data, labels) in tqdm(enumerate(train)):

        data, labels = data.to(device), labels['label'].to(device)

        output = client(data)
        output = down(output)
        output = enc(output)
        output = output[len(output)-gate_ind-1]
        gate_score = gate(output)

        output = dec(output)
        output = up(output)
        pred = server(output)
        pred = classifier(pred)

        # the confidence of the correct label
        pred = torch.softmax(pred, dim=1)
        conf = pred[torch.arange(pred.size(0)), labels] # cool
        conf = conf.detach()

        optimizer.zero_grad()
        # print('g&c', gate_score[0], conf[0])
        conf = conf.unsqueeze(1)
        loss = criterion(gate_score, conf)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = train_loss/len(train.dataset)
    print('train loss: ', train_loss)

    val_acc = 0
    gate.eval()

    for i, (data, labels) in tqdm(enumerate(val)):

        data, labels = data.to(device), labels['label'].to(device)

        output = client(data)
        output = down(output)
        output = enc(output)
        output = output[len(output)-gate_ind-1]
        gate_score = gate(output)

        output = dec(output)
        output = up(output)
        output = server(output)
        pred = classifier(output)

        # get the number of 0 and 1
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)

        gate_threshold = 0.5
        gate_eval = torch.where(gate_score > gate_threshold, torch.tensor([1.]).to(device), torch.tensor([0.]).to(device))
        pred_eval = torch.where(pred == labels, torch.tensor([1.]).to(device), torch.tensor([0.]).to(device))
        pred_eval = pred_eval.unsqueeze(1)

        accuracy = torch.eq(gate_eval, pred_eval).float()
        # print the rate of gate exit
        val_acc += accuracy.sum().item()
    val_acc = val_acc/len(val.dataset)
    print('val acc: ', val_acc)

    if val_acc > max_val_acc:
        max_val_acc = val_acc
        torch.save({
            'gate': gate.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc
        }, f'Weights/training/{model_type}_gate_{num_of_layers}_{num_of_ch}_{model_time}/{model_type}_gate_{num_of_layers}_{num_of_ch}_best.pth')
        print('model saved' + ' train loss ', train_loss, ' val acc ', val_acc)
    torch.save({
        'gate': gate.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }, f'Weights/training/{model_type}_gate_{num_of_layers}_{num_of_ch}_{model_time}/{model_type}_gate_{num_of_layers}_{num_of_ch}_last.pth')
