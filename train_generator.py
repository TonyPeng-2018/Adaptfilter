# the generator used the part of features to make all data
# We suppose that it is a GAN, DCGAN

from Dataloaders import dataloader_cifar10
import datetime
from Models import mobilenetv2, mobilenetv3, resnet, generator, encoder_client
import numpy as np
import os
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils import utils

in_ch = 32
# logger
stime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logger = utils.APLogger('./Logs/train_generator_' + stime + '.log')
logger.write('Dataset: CIFAR-10\n, Model: mobilenetv2\n, Generator: GeneratorV1\n, Encoder: 2layerCNN\n, Batch: 128\n, Epochs: 50\n, Seed: 2024\n')

# client, and server model
client_model, server_model = mobilenetv2.mobilenetv2_splitter(num_classes=10, weight_root='./Weights/cifar-10/', partition=-1)
client_model.eval()
server_model.eval()
client_model = client_model.to('cuda:0')
server_model = server_model.to('cuda:0')
for param in client_model.parameters():
    param.requires_grad = False
for param in server_model.parameters():
    param.requires_grad = False

# dataloader using the 
train, _, val = dataloader_cifar10.Dataloader_cifar10_val(train_batch=128, test_batch=100, seed=2024)


epochs = 50
# we have 3 generators for 3 discriminators
Generators = []
g_rate = [0.1*x for x in range(1, 10)]
for i in range(len(g_rate)):
    Generators.append(generator.Generator(int(in_ch*g_rate[i]), hiddensize=32, outputsize=32).cuda())
    Generators[i] = Generators[i].cuda()

Encoders = []
for i in range(len(g_rate)):
    Encoders.append(encoder_client.Encoder_Client(in_ch=in_ch, out_ch=int(g_rate[i]*in_ch)).cuda())
    Encoders[i] = Encoders[i].cuda()

# make the optimizers
optimizers_G = []
for i in range(len(g_rate)):
    optimizers_G.append(optim.Adam(Generators[i].parameters(), lr=0.001))

optimizers_E = []
for i in range(len(g_rate)):
    optimizers_E.append(optim.Adam(Encoders[i].parameters(), lr=0.001))

criterion = nn.CrossEntropyLoss()

# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.98)

for epoch in tqdm(range(epochs)):
    for i in range (len(Generators)):

        netG = Generators[i].cuda()
        netE = Encoders[i].cuda()
        netG.train()
        netE.train()

        optG = optimizers_G[i]
        optE = optimizers_E[i]
        netE.zero_grad()
        netG.zero_grad()


        for j, data in enumerate(train):
            img, label = data
            img, label = img.cuda(), label.cuda()
            
            out = client_model(img)            
            # train the encoder
            out = netE(out)
            # train the generator
            out = netG(out)
            # train the encoder            
            out = server_model(out)
            out = torch.argmax(out, dim=1)
            # change it to float

            print(out.size())
            print(label.size())

            loss = criterion(out, label)
            loss.backward()
            optE.step()
            optG.step()

            if j % 100 == 0:
                print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, epochs, j, len(train), loss.item()))
        
