from Models import mobilenetv2
import sys

model = mobilenetv2.MobileNetV2(num_classes=10)

from Dataloaders import dataloader_visdrone
# from Dataloaders import dataloader_cifar10

train, _, val = dataloader_visdrone.Dataloader_visdrone_integrated()
# train2, _, val2 = dataloader_cifar10.Dataloader_cifar10_val()

import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from tqdm import tqdm

epochs = 100
min_val_loss = 1000000
for epoch in tqdm(range(epochs)):
    model.train()
    for i, data in enumerate(train):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    model.eval()
    val_loss = 0.0
    val_acc = 0
    for i, data in enumerate(val):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        val_acc += torch.sum(outputs == labels).item()
    print('val loss: ', val_loss)
    print('val acc: ', val_acc / len(val.dataset))
    if val_acc / len(val.dataset) < min_val_loss:
        min_val_loss = val_acc / len(val.dataset)
        torch.save(model.state_dict(), 'mobile_visdrone.pth')