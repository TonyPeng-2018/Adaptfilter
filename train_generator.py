# the generator used the part of features to make all data
# We suppose that it is a GAN, DCGAN

from Dataloaders import dataloader_cifar10
import datetime
from Models import mobilenetv2, mobilenetv3, resnet, generator, encoder_client
import numpy as np
import os
import scipy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils import utils

in_ch = 32
g_rate = [0.5]
# logger
stime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logger = utils.APLogger('./Logs/cifar-10/train_generator_' + stime + '.log')
logger.write('Dataset: CIFAR-10\n, Model: mobilenetv2\n, Generator: GeneratorV3\n, Encoder: 2layerCNN\n, Batch: 128\n, Epochs: 100\n, Seed: 2024\n')
logger.write('Generator rate: ' + str(g_rate) + ' int transmission \n')

# client, and server model
client_model, server_model = mobilenetv2.mobilenetv2_splitter(num_classes=10, weight_root='./Weights/cifar-10/', partition=-1)
client_model.eval()
server_model.eval()
client_model = client_model.to('cuda:0')
server_model = server_model.to('cuda:0')
# for param in client_model.parameters():
#     param.requires_grad = False
# for param in server_model.parameters():
#     param.requires_grad = False

# dataloader using the 
train, _, val = dataloader_cifar10.Dataloader_cifar10_val(train_batch=128, test_batch=100, seed=2024)


epochs = 100
# we have 3 generators for 3 discriminators
Generators = []
for i in range(len(g_rate)): 
    Generators.append(generator.Generator(inputsize=int(in_ch*g_rate[0]), hiddensize=32, outputsize=32))
    Generators[i] = Generators[i].cuda()
# for i in range(len(g_rate)):
#     Generators.append([])
#     for i in range (in_ch):
#         Generators.append(generator.Generator3(img_size = (16,16), in_ch = int(g_rate[i]*in_ch), ind_ch=i))
#         Generators[0][i] = Generators[0][i].cuda()

Encoders = []
for i in range(len(g_rate)):
    Encoders.append(encoder_client.Encoder_Client(in_ch=in_ch, out_ch=int(g_rate[i]*in_ch)))
    Encoders[i] = Encoders[i].cuda()

# make the optimizers
optimizers_G = []
for i in range(len(g_rate)):
    optimizers_G.append(optim.Adam(Generators[i].parameters(), lr=0.001))

optimizers_E = []
for i in range(len(g_rate)):
    optimizers_E.append(optim.Adam(Encoders[i].parameters(), lr=0.001))

criterion = nn.CrossEntropyLoss() # CE loss doesn't work because it may generate another image
criterion = criterion.cuda()

# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.98)

min_val_loss = 1000000
for i in range (len(Generators)):

    netG = Generators[i].cuda()
    netE = Encoders[i].cuda()

    optG = optimizers_G[i]
    optE = optimizers_E[i]
    
    for epoch in tqdm(range(epochs)):

        netG.train()
        netE.train()
        netE.zero_grad()
        netG.zero_grad()

        for j, data in enumerate(train):
            img, label = data
            img, label = img.cuda(), label.cuda()
            
            out = client_model(img).detach()
            
            out_max = torch.max(out)
            out_min = torch.min(out)

            out = (out - out_min) / (out_max - out_min)

            # train the encoder
            out = netE(out)
            # train the generator
            
            out = out*255
            out = out.type(torch.uint8)


            out = out.type(torch.float32)
            out = out / 255

            out = netG(out)

            out = out * (out_max - out_min) + out_min
            # train the encoder            
            out = server_model(out)
            # out = torch.argmax(out, dim=1)

            # change it to float        
            loss = criterion(out, label)
            loss.backward()
            optE.step()
            optG.step()

            if j % 100 == 0:
                logger.write('[%d/%d][%d/%d] Loss: %.4f' % (epoch, epochs, j, len(train), loss.item()))

        # test the model
        netG.eval()
        netE.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for j, data in enumerate(val):
                img, label = data
                img, label = img.cuda(), label.cuda()
                out = client_model(img).detach()
                out_max = torch.max(out)
                out_min = torch.min(out)
                out = (out - out_min) / (out_max - out_min)
                out = netE(out)
                out = netG(out)
                out = out * (out_max - out_min) + out_min
                out = server_model(out)
                _, predicted = torch.max(out.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        logger.write('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        if min_val_loss > loss.item():
            min_val_loss = loss.item()
            torch.save(netG.state_dict(), './Weights/cifar-10/generator/generator_%.1f_'%g_rate[0] + stime +'.pth')
            torch.save(netE.state_dict(), './Weights/cifar-10/encoder/encoder_%.1f_'%g_rate[0] + stime +'.pth')
        
