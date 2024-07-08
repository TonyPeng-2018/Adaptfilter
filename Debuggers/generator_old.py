# the generator used the part of features to make all data
# We suppose that it is a GAN, DCGAN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Adaptfilter.Debuggers import mobilenetv2_revised
from Dataloaders import dataloader_cifar10
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import scipy

# client, and server model
client_model, server_model = mobilenetv2_revised.stupid_model_splitter(weight_path='./Weights/cifar-10/MobileNetV2.pth')

# def the generator
# https://github.com/pytorch/examples/blob/main/dcgan/main.py
class Generator(nn.Module):
    def __init__(self, inputsize, hiddensize, outputsize):
        super(Generator, self).__init__()
        self.inputsize = inputsize # 8, 16, 24
        self.outputsize = outputsize
        self.hiddensize = hiddensize
        self.section1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.inputsize, hiddensize * 8, 3, 1, padding=1, bias=False, dilation=1),
            nn.BatchNorm2d(hiddensize * 8),
            nn.ReLU(True)
        )
        self.section2 = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(hiddensize * 8, hiddensize * 4, 3, 1, padding=1 , bias=False, dilation=1),
            nn.BatchNorm2d(hiddensize * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(hiddensize * 4, hiddensize * 2, 3, 1, padding=1, bias=False, dilation=1),
            nn.BatchNorm2d(hiddensize * 2),
            nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(hiddensize * 2, hiddensize, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(hiddensize),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(hiddensize * 2, self.outputsize, 3, 1, padding=1, bias=False, dilation=1),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.section1(input)
        output = self.section2(output)
        return output

class Discriminator(nn.Module):
    def __init__(self, inputsize, hiddensize, outputsize, positionsize):
        super(Discriminator, self).__init__()
        self.inputsize = inputsize # 8, 16, 24
        self.outputsize = outputsize
        self.hiddensize = hiddensize
        self.positionsize = positionsize
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(inputsize, hiddensize, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(hiddensize, hiddensize*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(hiddensize * 2, hiddensize * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(hiddensize * 4, hiddensize * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(hiddensize * 8, inputsize, 3, 2, 1, bias=False),
            # nn.BatchNorm2d(hiddensize * 8),
            # nn.LeakyReLU(0.2, inplace=True),
        )

        self.position_encoder = nn.Sequential(
            nn.Linear(positionsize, 32),
            nn.ReLU(True)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(32+4096, 1),
            nn.Sigmoid()
        )
    def forward(self, input, input2):
        out1 = self.main(input) # hiddensize * n, ? ,32 ,32
        out2 = self.position_encoder(input2) # 32 -> 32
        # flatten and concatenate the two features 
        out1 = out1.flatten(start_dim=1)
        out = torch.cat((out1, out2), 1)
        output = self.output_layer(out)
        return output.view(-1, 1).squeeze(1)
    
# dataloader using the embeddings
class generator_dataloader():
    def __init__(self, embeddings_folder, labels_folder, gated):
        self.embeddings_folder = embeddings_folder
        self.gated = gated
        self.embeddings_files = sorted(os.listdir(embeddings_folder+str(gated)+'/embeddings/'))
        self.labels_folder = labels_folder
        self.labels_files = sorted(os.listdir(labels_folder+'embeddings/'))

    def __len__(self):
        return self.embeddings_files.__len__()

    def __getitem__(self, idx):
        self.embeddings = torch.load(self.embeddings_folder+str(self.gated)+'/embeddings/' + self.embeddings_files[idx])
        self.inds = self.embeddings[1]
        self.embeddings = self.embeddings[0]
        self.labels = torch.load(self.labels_folder+'embeddings/' + self.labels_files[idx])
        return self.embeddings, self.inds, self.labels

# dataloader using the embeddings
train_dataloaders = []
for i in range(3):
    train_dataloaders.append(generator_dataloader(embeddings_folder='../data/cifar-10-embedding-entropy/', labels_folder='../data/cifar-10-embedding-3/', gated=i))

epochs = 50
# we have 3 generators for 3 discriminators
Generators = []
inputsizes = [8, 16, 24]
for i in range(3):
    Generators.append(Generator(inputsize=inputsizes[i], hiddensize=32, outputsize=32).cuda())
Discriminators = []
for i in range(3):
    Discriminators.append(Discriminator(inputsize=32, hiddensize=32, outputsize=1, positionsize=32).cuda())

# make the optimizers
optimizers_G = []
optimizers_D = []
for i in range(3):
    optimizers_G.append(optim.Adam(Generators[i].parameters(), lr=0.0002, betas=(0.5, 0.999)))
    optimizers_D.append(optim.Adam(Discriminators[i].parameters(), lr=0.0002, betas=(0.5, 0.999)))

criterion = nn.BCELoss()

fixed_noises = []
batch_size = 128
for i in range(3):
    fixed_noises.append(torch.randn(batch_size, inputsizes[i], 1, 1).cuda())
real_flag = 1
fake_flag = 0

for epoch in tqdm(range(epochs)):
    for i in range (3):
        for j, data in enumerate(train_dataloaders[i]):
            emb, ind, label = data # emb 1,b,c',h,w, inds 1,b,c' labels 1,b,c,h,w
            # squeeze the embeddings
            emb, label = emb.squeeze(0), label.squeeze(0) # b,c',h,w
            # get the embeddings
            emb = emb.cuda()
            label = label.cuda()
            # create a n*c one hot vector
            one_hot = torch.zeros(emb.shape[0], 32).cuda()
            one_hot[ind] = 1
            # ind = ind.cuda() # it is not used here. How to do the positional encoding?
            # train the discriminator
            # train with real

            netG = Generators[i].cuda()
            netD = Discriminators[i].cuda()
            optG = optimizers_G[i]
            optD = optimizers_D[i]

            netD.zero_grad()
            real_cpu = label
            b_size = real_cpu.size(0)
            real_label = torch.full((b_size,), real_flag, device='cuda', dtype=real_cpu.dtype)
            output = netD(real_cpu, one_hot)
            errD_real = criterion(output, real_label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(b_size, inputsizes[i], 32, 32).cuda()
            fake = netG(noise)
            # print(fake.shape)
            fake_label = torch.full((b_size,), fake_flag, device='cuda', dtype=real_cpu.dtype)
            # add fake_positional encoding
            fake_ones = torch.zeros(b_size, 32).cuda()
            # add the fake ones
            fake_ind = torch.randint(0, 32, (inputsizes[i], 1)).cuda()
            fake_ones[fake_ind] = 1
            output = netD(fake.detach(), fake_ones)
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optD.step()

            # update Generator #
            netG.zero_grad()
            fake_label.fill_(real_flag)
            output = netD(fake, fake_ones)
            errG = criterion(output, fake_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optG.step()

        # print error
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(train_dataloaders[i]), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # save the model
        torch.save(netG.state_dict(), './Weights/cifar-10/generator_'+str(i)+'.pth')
        torch.save(netD.state_dict(), './Weights/cifar-10/discriminator_'+str(i)+'.pth')
        
