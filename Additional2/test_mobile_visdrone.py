from Models import mobilenetv2
import sys

model = mobilenetv2.MobileNetV2(num_classes=11)

from Dataloaders import dataloader_visdrone
# from Dataloaders import dataloader_cifar10

train, test, val = dataloader_visdrone.Dataloader_visdrone_integrated()
# train, test, val = dataloader_cifar10.Dataloader_cifar10_val()

import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:0')
model = model.to(device)
model = model.eval()
model.load_state_dict(torch.load('mobile_visdrone.pth'))

from tqdm import tqdm


acc = 0
n_test = 0
with torch.no_grad():

    for i, data in enumerate(test):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        acc += torch.sum(outputs == labels).item()
        n_test += len(labels)
        print(outputs, labels)
print('acc: ', acc / n_test)
        
