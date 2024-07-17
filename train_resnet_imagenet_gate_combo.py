from Models import resnet
import sys

middle_size = [1,2,4,8,16,32]
# middle_size = [1,2,16]
width = 56
height = 56
device = 'cuda:0'

client, server = resnet.resnet_splitter(weight_root='./Weights/imagenet/', layers=50, device=device)

from Dataloaders import dataloader_image_20

train, _, val, _ = dataloader_image_20.Dataloader_imagenet_20_integrated(train_batch=128, test_batch=50)
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
    middle.load_state_dict(torch.load('resnet_imagenet_middle_'+str(m)+'.pth'))
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
epochs = 100
max_val_acc = [0] * len(middle_size)
for epoch in tqdm(range(epochs)):
    train_loss = 0.0
    for j in range(len(middle_size)):
        gates[j].train()
    for i, data in enumerate(train):
        inputs, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        client_out = client(inputs).detach()
        for j in range(len(middle_size)):
            middle = middles[j]
            gate = gates[j]
            optimizer = ops[j]
            middle_out = middle.in_layer(client_out).detach()
            gate_out = gate(middle_out).squeeze()
            middle_out2 = middle.out_layer(middle_out).detach()
            server_out = server(middle_out2)
            # get the number of 0 and 1
            server_out = torch.softmax(server_out, dim=1)
            server_out = torch.argmax(server_out, dim=1)
            server_out = torch.eq(server_out, labels).float()

            # n_zero = torch.sum(server_out == 0).item()
            # n_one = torch.sum(server_out == 1).item()
            # # randomly select some ones equal to number of 0
            # if n_one > n_zero:
            #     n_one = n_zero
            # else:
            #     n_zero = n_one
            # s_one_cpu = torch.where(server_out == 1)[0].cpu().numpy()
            # s_zero_cpu = torch.where(server_out == 0)[0].cpu().numpy()
            # s_one = np.random.choice(s_one_cpu, n_one, replace=False)
            # s_zero = np.random.choice(s_zero_cpu, n_zero, replace=False)

            # s = np.concatenate((s_one, s_zero))
            # # s to torch
            # s = torch.from_numpy(s).to(device)
            # server_out = server_out[s]
            # gate_out = gate_out[s]        
            # print('gate_out: ', gate_out)
            # print('server_out: ', server_out)
            
            # gate_out = torch.round(gate_out)
            # print('gate_out', gate_out)

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
        inputs, labels, _ = data
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
        
        if val_accs[j] > max_val_acc[j]:
            max_val_acc[j] = val_accs[j]
            torch.save(gate.state_dict(), 'resnet_imagenet_gate_'+str(middle_size[j])+'.pth')
            print('model saved')