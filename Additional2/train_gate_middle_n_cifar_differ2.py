import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
gate = gatedmodel.GateCNN_v5(input_size=32, width=width, height=height, output_size=1)

gate = gate.to(device)

middle2 = mobilenetv2.MobileNetV2_middle(middle=2*middle_size)
middle2.load_state_dict(torch.load('model_middle_'+str(2*middle_size)+'.pth'))
middle2 = middle2.to(device)
middle2 = middle2.eval()

# criterion = nn.MSELoss()
# make not equaly distributed distribution
# a^(3*(1/x^3)) * (1/a)^3
optimizer = optim.Adam(gate.parameters(), lr=0.001)

from tqdm import tqdm
from Utils import utils
from torch.nn import functional as F

class my_loss(nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()
    def forward(self, x1, x2):
        loss = x1 * x2
        loss = (1 - loss)/2
        loss = torch.mean(loss)
        return loss

# criterion = my_loss()
criterion = nn.BCELoss()

import sys
import torchsummary
epochs = 200
min_val_loss = 1000000
for epoch in tqdm(range(epochs)):
    train_loss = 0.0
    gate.train()
    for i, data in enumerate(train):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = client(inputs)
        outputs_out = middle(outputs)
        outputs_out2 = middle2(outputs)

        target = server(outputs_out).detach()
        target = torch.functional.F.softmax(target, dim=1)
        target = torch.argmax(target, dim=1)

        target2 = server(outputs_out2).detach()
        target2 = torch.functional.F.softmax(target2, dim=1)
        target2 = torch.argmax(target2, dim=1)
        
        # target2 = server(outputs).detach()
        # target2 = torch.functional.F.softmax(target2, dim=1)
        # target2 = torch.argmax(target2, dim=1)

        # target_diff = torch.eq(target, target2).float()
        target_diff = target_diff.view(-1, 1)
        target_diff = target_diff.to(device)

        # target_diff = target_diff.view(-1, 1)

        # conf = gate_normal(a=normal_para, x=conf)

        outputs_diff = outputs_out - outputs_out2

        pred = gate(outputs_diff)
        # print(pred.size(), target_diff.size())
        loss = criterion(pred, target_diff)
        # print(loss)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('train loss: ', train_loss)
    gate.eval()
    val_loss = 0.0
    for i, data in enumerate(val):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = client(inputs)
        outputs_out = middle(outputs)

        target = server(outputs_out).detach()
        target = torch.functional.F.softmax(target, dim=1)
        target = torch.argmax(target, dim=1)

        target2 = server(outputs).detach()
        target2 = torch.functional.F.softmax(target2, dim=1)
        target2 = torch.argmax(target2, dim=1)

        # target_diff = torch.eq(target, target2).float()
        target_diff = torch.where(target == target2, torch.tensor(1.0), torch.tensor(0.0))
        target_diff = target_diff.view(-1, 1)
        target_diff = target_diff.to(device)

        # target_diff = target_diff.view(-1, 1)

        # conf = gate_normal(a=normal_para, x=conf)

        outputs_diff = outputs - outputs_out

        pred = gate(outputs_diff)
        # print(pred.size(), target_diff.size())
        loss = criterion(pred, target_diff)
        # print(loss)
        val_loss += loss.item()


    print('val loss: ', val_loss)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(gate.state_dict(), 'mobile_cifar-10_gate_diff_binary_'+str(middle_size)+'.pth')