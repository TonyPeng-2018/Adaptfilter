from Models import resnet
import sys

model_size = 'small'
middle_size = [1,2,4,8,16,32]
# middle_size = [1,2,4,8,16]
# middle_size = [1,2]
width = 56
height = 56

client, server = resnet.resnet_splitter(num_classes=34,layers=50, device='cuda:0', 
                                        weight_root='./Weights/ccpd-small', partition=-1)
from Dataloaders import dataloader_ccpd

train, _, val = dataloader_ccpd.Dataloader_ccpd_integrated(train_batch=64)
import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')
client = client.to(device)
server = server.to(device)
client = client.eval()
server = server.eval()

middles = []
for m in middle_size:
    middle = resnet.resnet_middle(middle=m)
    middle.load_state_dict(torch.load('./Weights/ccpd-small/middle/resnet_ccpd_small_middle_'+str(m)+'.pth'))
    middle = middle.to(device)
    middle = middle.eval()
    middles.append(middle)

from Models import gatedmodel
gates = []
for m in middle_size:
    gate = gatedmodel.ExitGate(in_planes=m, height=height, width=width) # sigmoid
    gate = gate.to(device)
    gates.append(gate)

criterion = nn.MSELoss()
ops = []
for gate in gates:
    optimizer = optim.Adam(gate.parameters(), lr=0.001) # 0.01 is too big
    ops.append(optimizer)

from tqdm import tqdm
from Utils import utils

import sys
import numpy as np
epochs = 50
max_val_acc = [0] * len(middle_size)
for epoch in tqdm(range(epochs)):
    train_loss = 0.0
    for j in range(len(middle_size)):
        gates[j].train()

    for i, data in enumerate(train):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        client_out = client(inputs).detach()
        for j in range(len(middle_size)):
            middle = middles[j]
            gate = gates[j]
            optimizer = ops[j]
            optimizer.zero_grad()
            middle_out = middle.in_layer(client_out).detach()
            gate_out = gate(middle_out).squeeze()
            middle_out2 = middle.out_layer(middle_out).detach()
            server_out = server(middle_out2)
            # get the number of 0 and 1
            server_out = torch.softmax(server_out, dim=1)
            server_out = torch.argmax(server_out, dim=1)
            server_out = torch.eq(server_out, labels).float()

            loss = criterion(gate_out, server_out)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
    print('train loss: ', train_loss)
    for j in range(len(middle_size)):
        gates[j].eval()
    val_accs = [0] * len(middle_size)
    gate_exits = [0] * len(middle_size)
    send_counts = [0] * len(middle_size)
    for i, data in enumerate(val):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        client_out = client(inputs).detach()
        for j in range (len(middle_size)):
            middle = middles[j]
            gate = gates[j]

            middle_out = middle.in_layer(client_out).detach()

            gate_out = gate(middle_out).squeeze()
            middle_out2 = middle.out_layer(middle_out).detach()
            server_out = server(middle_out2)
            # get the number of 0 and 1
            server_out = torch.softmax(server_out, dim=1)
            server_out = torch.argmax(server_out, dim=1)
            server_out = torch.eq(server_out, labels).float()

            # print('gate_out: ', torch.where(gate_out>0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)))

            # gate_out = torch.round(gate_out)
            gate_out = torch.gt(gate_out, 0.9).float()
            gate_exits[j] += torch.sum(gate_out).item()

            # get ther server_out [ gate_out == 1]
            s = torch.where(gate_out == 1)[0]
            server_out = server_out[s]
            gate_out = gate_out[s]
            gate_acc = torch.eq(server_out, gate_out).float()
            val_accs[j] += torch.sum(gate_acc).item()
            send_counts[j] += len(s)

            # print the rate of gate exit
    for j in range (len(middle_size)):
        gate_exits[j] = gate_exits[j] / len(val.dataset)

        val_accs[j] = val_accs[j] / max(1,send_counts[j])
        print('gate_exit: ', gate_exits[j])
        print('val_acc: ', val_accs[j])
        
        if val_accs[j] >= max_val_acc[j]:
            max_val_acc[j] = val_accs[j]
            torch.save(gates[j].state_dict(), 'resnet_ccpd_gate_'+str(middle_size[j])+'.pth')
            print('model saved')