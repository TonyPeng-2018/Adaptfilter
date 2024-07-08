import torch.nn as nn
import torch

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
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.section1(input)
        output = self.section2(output)
        return output
    
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
class Generator2(nn.Module):
    def __init__(self, classes=32, img_size = (16,16), in_ch = 3):
        super(Generator2, self).__init__()
        self.classes = classes
        self.latent_dim = img_size[0] * img_size[1] * in_ch
        self.img_size = img_size
        self.label_emb = nn.Embedding(self.classes, self.classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(self.img_size[0] * self.img_size[1])),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), self.img_size)
        return img
    
class Generator3(nn.Module):
    def __init__(self, img_size = (16,16), in_ch = 3, all_ch = 32):
        super(Generator3, self).__init__()
        self.latent_dim = img_size[0] * img_size[1] * in_ch
        self.img_size = img_size
        self.all_ch = all_ch

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, self.latent_dim*2, normalize=False),
            *block(self.latent_dim*2, self.latent_dim*4),
            *block(self.latent_dim*4, self.latent_dim*8),
            *block(self.latent_dim*8, self.latent_dim*16),
            nn.Linear(self.latent_dim*16, int(self.img_size[0] * self.img_size[1] * self.all_ch)),
            nn.Tanh()
        )

    def forward(self, x):
        # Concatenate label embedding and image to produce input
        img = self.model(x)
        img = img.view(img.size(0), (self.all_ch, self.img_size[0], self.img_size[1]))
        return img