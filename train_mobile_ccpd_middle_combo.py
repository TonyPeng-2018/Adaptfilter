from Models import mobilenetv2
import sys


model_size = 'small'
middle_size = [1,2,4,8,16]
client, server = mobilenetv2.mobilenetv2_splitter(num_classes=34, 
                                        weight_root='/home/tonypeng/Workspace1/adaptfilter/Adaptfilter/Weights/ccpd-small', 
                                        device='cuda:0')

from Dataloaders import dataloader_ccpd

train, _, val = dataloader_ccpd.Dataloader_ccpd_integrated()

import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')
client = client.to(device)
server = server.to(device)
client = client.eval()
server = server.eval()

middle_models =[]
for m in middle_size:
    middle = mobilenetv2.MobileNetV2_middle(middle=m)
    middle = middle.to(device)
    middle_models.append(middle)

criterion = nn.CrossEntropyLoss()
ops = []
for middle in middle_models:
    optimizer = optim.Adam(middle.parameters(), lr=0.001)
    ops.append(optimizer)

from tqdm import tqdm

epochs = 60
min_val_loss = [10000000] * len(middle_size)
for epoch in tqdm(range(epochs)):
    train_loss = [0] * len(middle_size)
    for j in range(len(middle_size)):
        middle_models[j].train()
    for i, data in enumerate(train):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        client_out = client(inputs).detach()
        for j in range(len(middle_size)):
            middle = middle_models[j]
            optimizer = ops[j]
            optimizer.zero_grad()
            outputs = middle(client_out)
            outputs = server(outputs)
            # print(outputs.size(), labels.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss[j] += loss.item()
    print('epoch: ', epoch)
    print('train loss: ', train_loss/len(train))
    for j in range(len(middle_size)):
        middle_models[j].eval()
    val_loss = [0] * len(middle_size)
    for i, data in enumerate(val):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        client_out = client(inputs).detach()
        for j in range(len(middle_size)):
            middle = middle_models[j]
            outputs = middle(client_out)
            outputs = server(outputs)
            # print(outputs.size(), labels.size())
            loss = criterion(outputs, labels)
            loss.backward()
            val_loss[j] += loss.item()

    for j in range(len(middle_size)):
        val_loss[j] /= len(val)
    print('val loss: ', val_loss)
    for j in range(len(middle_size)):
        middle = middle_models[j]
        if val_loss[j] < min_val_loss[j]:
            min_val_loss[j] = val_loss[j]
            torch.save(middle.state_dict(), 'mobile_ccpd_'+model_size+'_middle_'+str(middle_size[j])+'.pth')