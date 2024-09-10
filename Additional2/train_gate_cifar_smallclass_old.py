from Models import mobilenetv2
import sys

middle_size = int(sys.argv[1])
width = 16
height = 16
normal_para = 6

client, server = mobilenetv2.mobilenetv2_splitter(num_classes=10,
                                                  weight_root='/home/tonypeng/Workspace1/adaptfilter/Adaptfilter/Weights/cifar-10',
                                                  device='cuda:0',partition=-1)

from Dataloaders import dataloader_cifar10

train, _, val = dataloader_cifar10.Dataloader_cifar10_val(train_batch=128, test_batch=100, seed=2024)
import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')
client = client.to(device)
server = server.to(device)
client = client.eval()
server = server.eval()

middle = mobilenetv2.MobileNetV2_middle(middle=middle_size)
middle.load_state_dict(torch.load('model_middle_'+str(middle_size)+'.pth'))
middle = middle.to(device)
middle = middle.eval()

from Models import gatedmodel
gate = gatedmodel.Gated_MLP(in_size1=middle_size, in_size2=32, width=width, height=height, output_size=1) # sigmoid

gate = gate.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(gate.parameters(), lr=0.001)

from tqdm import tqdm
from Utils import utils

import sys
import torchsummary
import numpy as np
epochs = 100
min_val_loss = 1000000
for epoch in tqdm(range(epochs)):
    train_loss = 0.0
    gate.train()
    for i, data in enumerate(train):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        client_out = client(inputs)
        middle_out = middle.in_layer(client_out)
        middle_out2 = middle.out_layer(middle_out).detach()

        gate_out = gate(middle_out, middle_out2).squeeze()
        server_out = server(middle_out2)
        server_out2 = server(client_out)
        # get the number of 0 and 1
        server_out = torch.softmax(server_out, dim=1)
        
        server_out = torch.gather(server_out, 1, labels.view(-1,1))
        server_out2 = torch.softmax(server_out2, dim=1)
        server_out2 = torch.gather(server_out2, 1, labels.view(-1,1))
        server_diff = torch.abs(server_out - server_out2)
        print('server_out: ', server_out)
        print('gate_out: ', gate_out)
        print('server_out2: ', server_out2)
        print('server_diff: ', server_diff)


        # gate_out = torch.round(gate_out)
        # print('gate_out', gate_out)

        loss = criterion(gate_out, server_diff)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('train loss: ', train_loss)
    gate.eval()
    val_loss = 0.0
    val_acc = 0
    for i, data in enumerate(val):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        client_out = client(inputs)
        middle_out = middle.in_layer(client_out).detach()
        middle_out2 = middle.out_layer(middle_out).detach()

        gate_out = gate(middle_out, middle_out2).squeeze()

        server_out = server(middle_out2)
        server_out2 = server(client_out)
        # get the number of 0 and 1
        server_out = torch.softmax(server_out, dim=1)
        server_out = torch.argmax(server_out, dim=1)
        server_out2 = torch.softmax(server_out2, dim=1)
        server_out2 = torch.argmax(server_out2, dim=1)
        server_out = torch.eq(server_out, server_out2).float()

        # if the confidence is > 0.9 
        # gate_out = torch.round(gate_out)
        # gate_out_up = torch.gt(gate_out, 0.5).float()
        # check the accuracy for the gate
        # get the index of gate_out_up if it is 1
        # gate_out_up = torch.eq(gate_out_up, server_out).float()

        # gate_out_up_ind = torch.where(gate_out_up == 1)[0]
        # server_out = server_out[gate_out_up_ind]
        # gate_out = torch.ones_like(gate_out_up_ind).float()

        print('gate_out: ', len(gate_out), gate_out)
        print('server_out: ', len(gate_out), server_out)
        loss = torch.eq(gate_out, server_out).float()
        loss = torch.sum(loss)

        val_loss += loss/max(1,len(gate_out))

    # print('val acc: ', val_acc)
    print('val loss: ', val_loss)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(gate.state_dict(), 'mobile_cifar-10_gate_'+str(middle_size)+'.pth')
        print('save model')