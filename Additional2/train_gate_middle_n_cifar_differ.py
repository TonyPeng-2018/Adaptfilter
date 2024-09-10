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
gate = gatedmodel.GatedRegression(input_size=middle_size, width=width, height=height, output_size=1)

gate = gate.to(device)

def self_loss(pred, target):
    
    l2_loss = torch.nn.functional.mse_loss(pred, target)
criterion = self_loss()
# criterion = nn.MSELoss()
# make not equaly distributed distribution
# a^(3*(1/x^3)) * (1/a)^3
optimizer = optim.Adam(gate.parameters(), lr=0.1)

from tqdm import tqdm
from Utils import utils

gate_normal = utils.gate_normal3
gate_renormal = utils.gate_renormal3

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
        outputs = client(inputs).detach()
        outputs_in = middle.in_layer(outputs)

        conf = middle.out_layer(outputs_in)
        conf = server(conf)
        conf = torch.functional.F.softmax(conf, dim=1)
        conf = conf.gather(1, labels.view(-1, 1))

        conf2 = server(outputs)
        conf2 = torch.functional.F.softmax(conf2, dim=1)
        conf2 = conf2.gather(1, labels.view(-1, 1))

        conf_diff = conf - conf2
        
        # conf = gate_normal(a=normal_para, x=conf)

        pred = gate(outputs_in)
        loss = criterion(pred, conf_diff)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('train loss: ', train_loss)
    gate.eval()
    val_loss = 0.0
    for i, data in enumerate(val):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = client(inputs)
        outputs = middle.in_layer(outputs)

        conf = middle.out_layer(outputs)
        conf = server(conf)
        conf = torch.functional.F.softmax(conf, dim=1)
        conf = conf.gather(1, labels.view(-1, 1))
        
        pred = gate(outputs)
        pred = gate_renormal(a=normal_para, x=pred)
        loss = criterion(pred, conf)

        val_loss += loss.item()

    print('val loss: ', val_loss)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(gate.state_dict(), 'mobile_cifar-10_gate_'+str(middle_size)+'.pth')