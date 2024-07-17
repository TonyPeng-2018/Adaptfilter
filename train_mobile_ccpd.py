# get the mobile
from Models import mobilenetv2 

data_size = 'large'
data_device = 'tintin'

model = mobilenetv2.MobileNetV2(num_classes=34)

import torch
device = 'cuda:2'
model = model.to(device)

from Dataloaders import dataloader_ccpd
train, val, test = dataloader_ccpd.Dataloader_ccpd_integrated(train_batch=8, test_batch=100, size=data_size, device=data_device)

import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from tqdm import tqdm
from Utils import utils

import sys
import numpy as np

epochs = 50
min_val_loss = 1000000

for epoch in tqdm(range(epochs)):
    train_loss = 0.0
    model.train()
    for i, data in enumerate(train):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('train_loss: ', train_loss)
    val_loss = 0.0
    model.eval()
    correct = 0
    for i, data in enumerate(val):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        
        out = torch.argmax(outputs, dim=1)
        correct += torch.sum(out == labels).item()

    print('val_loss: ', val_loss)
    print('correct: ', correct/len(val))
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'mobile_ccpd_'+data_size+'.pth')
    # accuracy

    print('min_val_loss: ', min_val_loss)
    print('epoch: ', epoch)
    print('---------------------------------')
