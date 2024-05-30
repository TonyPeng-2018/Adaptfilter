# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:38:51 2023
@author: Hostl
"""
import torch,torchvision,tqdm,pickle,os,sys,time
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

"""Baseline models """
"""Concated models"""
#mobilnet model
class Autoencodermobilenetc(nn.Module):
    def __init__(self):
        super(Autoencodermobilenetc, self).__init__()
        self.backbone       = models.mobilenet_v2(pretrained=True)
        self.backbone       = nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)        
        self.E100   = nn.Conv2d(in_channels=1280, out_channels=180, kernel_size=3, stride=1, padding=1)#adds to 718
        self.E75    = nn.Conv2d(in_channels=1280, out_channels=180, kernel_size=3, stride=1, padding=1)#adds to 538
        self.E50    = nn.Conv2d(in_channels=1280, out_channels=178, kernel_size=3, stride=1, padding=1)#adds to 358
        self.E25    = nn.Conv2d(in_channels=1280, out_channels=108, kernel_size=3, stride=1, padding=1)#adds to 180        
        self.E10    = nn.Conv2d(in_channels=1280, out_channels=44, kernel_size=3, stride=1, padding=1) #adds to 72
        self.E4     = nn.Conv2d(in_channels=1280, out_channels=6, kernel_size=3, stride=1, padding=1)  #adds to 28
        self.E3     = nn.Conv2d(in_channels=1280, out_channels=8, kernel_size=3, stride=1, padding=1)  #adds to 22
        self.E2     = nn.Conv2d(in_channels=1280, out_channels=6, kernel_size=3, stride=1, padding=1)  #adds to 14
        self.E1     = nn.Conv2d(in_channels=1280, out_channels=6, kernel_size=3, stride=1, padding=1)  #adds to 8
        self.Ep40   = nn.Conv2d(in_channels=1280, out_channels=2, kernel_size=3, stride=1, padding=1)                
        self.D100   = Decoder(718)
        self.D75    = Decoder(538)
        self.D50    = Decoder(358)
        self.D25    = Decoder(180)
        self.D10    = Decoder(72)       
        self.D4     = Decoder(28)
        self.D3     = Decoder(22)
        self.D2     = Decoder(14)
        self.D1     = Decoder(8)
        self.Dp40   = Decoder(2)                
    def forward(self, x):
        base = self.backbone(x)        
        r_40 = self.Ep40(base)
        r_1  = self.E1(base)
        r_2  = self.E2(base)
        r_3  = self.E3(base)
        r_4  = self.E4(base)
        r_10 = self.E10(base)
        r_25 = self.E25(base)
        r_50 = self.E50(base)
        r_75 = self.E75(base)
        r_100 = self.E100(base)                
        r_1 = torch.cat((r_1, r_40), dim=1)
        r_2 = torch.cat((r_2, r_1), dim=1)
        r_3 = torch.cat((r_3, r_2), dim=1)
        r_4 = torch.cat((r_4, r_3), dim=1)
        r_10 = torch.cat((r_10, r_4), dim=1)
        r_25 = torch.cat((r_25, r_10), dim=1)
        r_50 = torch.cat((r_50, r_25), dim=1)
        r_75 = torch.cat((r_75, r_50), dim=1)
        r_100 = torch.cat((r_100, r_75), dim=1)                
        res_100 = self.D100(r_100)
        res_75  = self.D75(r_75)
        res_50  = self.D50(r_50)
        res_25  = self.D25(r_25)        
        res_10 = self.D10(r_10)
        res_4  = self.D4(r_4)
        res_3  = self.D3(r_3)
        res_2  = self.D2(r_2)
        res_1  = self.D1(r_1)        
        res_p40  = self.Dp40(r_40)
        return res_100,res_75,res_50,res_25,res_10,res_4,res_3,res_2,res_1,res_p40
#Resnet models both cat and noncat
class Decoder(nn.Module):
    def __init__(self,starting_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(starting_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid for pixel values in the range [0, 1]
        )

    def forward(self, x):
        return self.decoder(x)

class Autoencoder50c(nn.Module):
    def __init__(self):
        super(Autoencoder50c, self).__init__()
        self.backbone       = models.resnet50(pretrained=True)
        self.backbone       = nn.Sequential(*list(self.backbone.children())[:-2])
        self.backbone[0]    = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)        
        self.D100   = Decoder(1024)
        self.D75    = Decoder(768)
        self.D50    = Decoder(512)
        self.D25    = Decoder(256)
        self.D10    = Decoder(102)       
        self.D4     = Decoder(42)
        self.D3     = Decoder(32)
        self.D2     = Decoder(22)
        self.D1     = Decoder(12)
        self.Dp40   = Decoder(4)                
        self.E100   = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=1, padding=1)#addsto1024
        self.E75    = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=1, padding=1)#addsto 768
        self.E50    = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=1, padding=1)#addsto 316
        self.E25    = nn.Conv2d(in_channels=2048, out_channels=154, kernel_size=3, stride=1, padding=1)#addsto 256
        self.E10    = nn.Conv2d(in_channels=2048, out_channels=60, kernel_size=3, stride=1, padding=1)#adds to 102
        self.E4     = nn.Conv2d(in_channels=2048, out_channels=10, kernel_size=3, stride=1, padding=1)#adds to 42
        self.E3     = nn.Conv2d(in_channels=2048, out_channels=10, kernel_size=3, stride=1, padding=1)#adds to 32
        self.E2     = nn.Conv2d(in_channels=2048, out_channels=10, kernel_size=3, stride=1, padding=1)#adds to 22
        self.E1     = nn.Conv2d(in_channels=2048, out_channels=8, kernel_size=3, stride=1, padding=1) #adds to 12
        self.Ep40   = nn.Conv2d(in_channels=2048, out_channels=4, kernel_size=3, stride=1, padding=1)                                
    def forward(self, x):        
        base = self.backbone(x)        
        r_40 = self.Ep40(base)
        r_1  = self.E1(base)
        r_2  = self.E2(base)
        r_3  = self.E3(base)
        r_4  = self.E4(base)
        r_10 = self.E10(base)
        r_25 = self.E25(base)
        r_50 = self.E50(base)
        r_75 = self.E75(base)
        r_100 = self.E100(base)                
        r_1 = torch.cat((r_1, r_40), dim=1)
        r_2 = torch.cat((r_2, r_1), dim=1)
        r_3 = torch.cat((r_3, r_2), dim=1)
        r_4 = torch.cat((r_4, r_3), dim=1)
        r_10 = torch.cat((r_10, r_4), dim=1)
        r_25 = torch.cat((r_25, r_10), dim=1)
        r_50 = torch.cat((r_50, r_25), dim=1)
        r_75 = torch.cat((r_75, r_50), dim=1)
        r_100 = torch.cat((r_100, r_75), dim=1)                
        res_100 = self.D100(r_100)
        res_75  = self.D75(r_75)
        res_50  = self.D50(r_50)
        res_25  = self.D25(r_25)        
        res_10 = self.D10(r_10)
        res_4  = self.D4(r_4)
        res_3  = self.D3(r_3)
        res_2  = self.D2(r_2)
        res_1  = self.D1(r_1)        
        res_p40  = self.Dp40(r_40)
        return res_100,res_75,res_50,res_25,res_10,res_4,res_3,res_2,res_1,res_p40

class Autoencoder18c(nn.Module):
    def __init__(self):
        super(Autoencoder18c, self).__init__()
        self.backbone       = models.resnet18(pretrained=True)
        self.backbone       = nn.Sequential(*list(self.backbone.children())[:-2])
        self.backbone[0]    = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.D100   = Decoder(1024)
        self.D75    = Decoder(768)
        self.D50    = Decoder(512)
        self.D25    = Decoder(256)
        self.D10    = Decoder(102)       
        self.D4     = Decoder(42)
        self.D3     = Decoder(32)
        self.D2     = Decoder(22)
        self.D1     = Decoder(12)
        self.Dp40   = Decoder(4)                
        self.E100   = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)#addsto1024
        self.E75    = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)#addsto 768
        self.E50    = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)#addsto 316
        self.E25    = nn.Conv2d(in_channels=512, out_channels=154, kernel_size=3, stride=1, padding=1)#addsto 256
        self.E10    = nn.Conv2d(in_channels=512, out_channels=60, kernel_size=3, stride=1, padding=1)#adds to 102
        self.E4     = nn.Conv2d(in_channels=512, out_channels=10, kernel_size=3, stride=1, padding=1)#adds to 42
        self.E3     = nn.Conv2d(in_channels=512, out_channels=10, kernel_size=3, stride=1, padding=1)#adds to 32
        self.E2     = nn.Conv2d(in_channels=512, out_channels=10, kernel_size=3, stride=1, padding=1)#adds to 22
        self.E1     = nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=1, padding=1) #adds to 12
        self.Ep40   = nn.Conv2d(in_channels=512, out_channels=4, kernel_size=3, stride=1, padding=1)                                
    def forward(self, x):        
        base = self.backbone(x)        
        r_40 = self.Ep40(base)
        r_1  = self.E1(base)
        r_2  = self.E2(base)
        r_3  = self.E3(base)
        r_4  = self.E4(base)
        r_10 = self.E10(base)
        r_25 = self.E25(base)
        r_50 = self.E50(base)
        r_75 = self.E75(base)
        r_100 = self.E100(base)                
        r_1 = torch.cat((r_1, r_40), dim=1)
        r_2 = torch.cat((r_2, r_1), dim=1)
        r_3 = torch.cat((r_3, r_2), dim=1)
        r_4 = torch.cat((r_4, r_3), dim=1)
        r_10 = torch.cat((r_10, r_4), dim=1)
        r_25 = torch.cat((r_25, r_10), dim=1)
        r_50 = torch.cat((r_50, r_25), dim=1)
        r_75 = torch.cat((r_75, r_50), dim=1)
        r_100 = torch.cat((r_100, r_75), dim=1)                
        res_100 = self.D100(r_100)
        res_75  = self.D75(r_75)
        res_50  = self.D50(r_50)
        res_25  = self.D25(r_25)        
        res_10 = self.D10(r_10)
        res_4  = self.D4(r_4)
        res_3  = self.D3(r_3)
        res_2  = self.D2(r_2)
        res_1  = self.D1(r_1)        
        res_p40  = self.Dp40(r_40)
        return res_100,res_75,res_50,res_25,res_10,res_4,res_3,res_2,res_1,res_p40
"""Nonconcated models"""
#mobilnet model
class Autoencodermobilenetnc(nn.Module):
    def __init__(self):
        super(Autoencodermobilenetnc, self).__init__()
        self.backbone       = models.mobilenet_v2(pretrained=True)
        self.backbone       = nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        self.E100   = nn.Conv2d(in_channels=1280, out_channels=718, kernel_size=3, stride=1, padding=1)
        self.E75    = nn.Conv2d(in_channels=1280, out_channels=538, kernel_size=3, stride=1, padding=1)
        self.E50    = nn.Conv2d(in_channels=1280, out_channels=358, kernel_size=3, stride=1, padding=1)
        self.E25    = nn.Conv2d(in_channels=1280, out_channels=180, kernel_size=3, stride=1, padding=1)        
        self.E10    = nn.Conv2d(in_channels=1280, out_channels=72, kernel_size=3, stride=1, padding=1)       
        self.E4     = nn.Conv2d(in_channels=1280, out_channels=28, kernel_size=3, stride=1, padding=1)
        self.E3     = nn.Conv2d(in_channels=1280, out_channels=22, kernel_size=3, stride=1, padding=1)
        self.E2     = nn.Conv2d(in_channels=1280, out_channels=14, kernel_size=3, stride=1, padding=1)
        self.E1     = nn.Conv2d(in_channels=1280, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.Ep40   = nn.Conv2d(in_channels=1280, out_channels=2, kernel_size=3, stride=1, padding=1)
                
        self.D100   = Decoder(718)
        self.D75    = Decoder(538)
        self.D50    = Decoder(358)
        self.D25    = Decoder(180)
        self.D10    = Decoder(72)       
        self.D4     = Decoder(28)
        self.D3     = Decoder(22)
        self.D2     = Decoder(14)
        self.D1     = Decoder(8)
        self.Dp40   = Decoder(2)
        
        
    def forward(self, x):
        
        base = self.backbone(x)
        
        res_100 = self.D100(self.E100(base))
        res_75  = self.D75(self.E75(base))
        res_50  = self.D50(self.E50(base))
        res_25  = self.D25(self.E25(base))
        
        res_10 = self.D10(self.E10(base))
        res_4  = self.D4(self.E4(base))
        res_3  = self.D3(self.E3(base))
        res_2  = self.D2(self.E2(base))
        res_1  = self.D1(self.E1(base))
        
        res_p40  = self.Dp40(self.Ep40(base))
        return res_100,res_75,res_50,res_25,res_10,res_4,res_3,res_2,res_1,res_p40

class Autoencoder50nc(nn.Module):
    def __init__(self):
        super(Autoencoder50nc, self).__init__()
        self.backbone       = models.resnet50(pretrained=True)
        self.backbone       = nn.Sequential(*list(self.backbone.children())[:-2])
        self.backbone[0]    = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.E100   = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.E75    = nn.Conv2d(in_channels=2048, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.E50    = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.E25    = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=1, padding=1)        
        self.E10    = nn.Conv2d(in_channels=2048, out_channels=102, kernel_size=3, stride=1, padding=1)       
        self.E4     = nn.Conv2d(in_channels=2048, out_channels=42, kernel_size=3, stride=1, padding=1)
        self.E3     = nn.Conv2d(in_channels=2048, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.E2     = nn.Conv2d(in_channels=2048, out_channels=22, kernel_size=3, stride=1, padding=1)
        self.E1     = nn.Conv2d(in_channels=2048, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.Ep40   = nn.Conv2d(in_channels=2048, out_channels=4, kernel_size=3, stride=1, padding=1)
                
        self.D100   = Decoder(1024)
        self.D75    = Decoder(768)
        self.D50    = Decoder(512)
        self.D25    = Decoder(256)
        self.D10    = Decoder(102)       
        self.D4     = Decoder(42)
        self.D3     = Decoder(32)
        self.D2     = Decoder(22)
        self.D1     = Decoder(12)
        self.Dp40   = Decoder(4)
        
        
    def forward(self, x):
        
        base = self.backbone(x)
        
        res_100 = self.D100(self.E100(base))
        res_75  = self.D75(self.E75(base))
        res_50  = self.D50(self.E50(base))
        res_25  = self.D25(self.E25(base))
        
        res_10 = self.D10(self.E10(base))
        res_4  = self.D4(self.E4(base))
        res_3  = self.D3(self.E3(base))
        res_2  = self.D2(self.E2(base))
        res_1  = self.D1(self.E1(base))
        
        res_p40  = self.Dp40(self.Ep40(base))
        return res_100,res_75,res_50,res_25,res_10,res_4,res_3,res_2,res_1,res_p40

class Autoencoder18nc(nn.Module):
    def __init__(self):
        super(Autoencoder18nc, self).__init__()
        self.backbone       = models.resnet18(pretrained=True)
        self.backbone       = nn.Sequential(*list(self.backbone.children())[:-2])
        self.backbone[0]    = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.E100   = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.E75    = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.E50    = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.E25    = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)        
        self.E10    = nn.Conv2d(in_channels=512, out_channels=102, kernel_size=3, stride=1, padding=1)       
        self.E4     = nn.Conv2d(in_channels=512, out_channels=42, kernel_size=3, stride=1, padding=1)
        self.E3     = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.E2     = nn.Conv2d(in_channels=512, out_channels=22, kernel_size=3, stride=1, padding=1)
        self.E1     = nn.Conv2d(in_channels=512, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.Ep40   = nn.Conv2d(in_channels=512, out_channels=4, kernel_size=3, stride=1, padding=1)
                
        self.D100   = Decoder(1024)
        self.D75    = Decoder(768)
        self.D50    = Decoder(512)
        self.D25    = Decoder(256)
        self.D10    = Decoder(102)       
        self.D4     = Decoder(42)
        self.D3     = Decoder(32)
        self.D2     = Decoder(22)
        self.D1     = Decoder(12)
        self.Dp40   = Decoder(4)
        
        
    def forward(self, x):
        
        base = self.backbone(x)
        
        res_100 = self.D100(self.E100(base))
        res_75  = self.D75(self.E75(base))
        res_50  = self.D50(self.E50(base))
        res_25  = self.D25(self.E25(base))
        
        res_10 = self.D10(self.E10(base))
        res_4  = self.D4(self.E4(base))
        res_3  = self.D3(self.E3(base))
        res_2  = self.D2(self.E2(base))
        res_1  = self.D1(self.E1(base))
        
        res_p40  = self.Dp40(self.Ep40(base))
        return res_100,res_75,res_50,res_25,res_10,res_4,res_3,res_2,res_1,res_p40

"""mobile_gated concated"""
class Encoder_mobile_con(nn.Module):
    def __init__(self,out_channels):
        super(Encoder_mobile_con, self).__init__()
        self.encoder = nn.Conv2d(in_channels=1280, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.encoder(x)
    
class Decoder_mobile_con(nn.Module):
    def __init__(self,starting_channels):
        super(Decoder_mobile_con, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(starting_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid for pixel values in the range [0, 1]
        )

    def forward(self, x):
        return self.decoder(x)
    
class ExitGate_mobile_con(nn.Module):
    def __init__(self, in_planes, height, width):
        super().__init__()
        self.pool = nn.AvgPool2d((int(height), int(width)))
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(64, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn0(self.pool(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = torch.flatten(x, 1)
        out = self.linear(x)
        out = self.sigmoid(out)

        return out
    
class mobilenet_concat_gate_eval(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.backbone       = models.mobilenet_v2(pretrained=True)
        self.backbone       = nn.Sequential(*list(self.backbone.children())[:-1]) # leave the feature layer here.
        self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # the input channel is 1?
        
        self.encoders = nn.ModuleList()
        self.encoders.append(Encoder_mobile_con(2))
        self.encoders.append(Encoder_mobile_con(6))
        self.encoders.append(Encoder_mobile_con(6))
        self.encoders.append(Encoder_mobile_con(8))
        self.encoders.append(Encoder_mobile_con(6))
        self.encoders.append(Encoder_mobile_con(44))
        self.encoders.append(Encoder_mobile_con(108))
        self.encoders.append(Encoder_mobile_con(178))
        self.encoders.append(Encoder_mobile_con(180))
        self.encoders.append(Encoder_mobile_con(180))
        
        self.decoders = nn.ModuleList()
        self.decoders.append(Decoder_mobile_con(2))
        self.decoders.append(Decoder_mobile_con(8))
        self.decoders.append(Decoder_mobile_con(14))
        self.decoders.append(Decoder_mobile_con(22))
        self.decoders.append(Decoder_mobile_con(28))
        self.decoders.append(Decoder_mobile_con(72))
        self.decoders.append(Decoder_mobile_con(180))
        self.decoders.append(Decoder_mobile_con(358))
        self.decoders.append(Decoder_mobile_con(538))
        self.decoders.append(Decoder_mobile_con(718))
        
        self.gates = nn.ModuleList()
        self.gates.append(ExitGate_mobile_con(2,height/32,width/32))
        self.gates.append(ExitGate_mobile_con(8,height/32,width/32))
        self.gates.append(ExitGate_mobile_con(14,height/32,width/32))
        self.gates.append(ExitGate_mobile_con(22,height/32,width/32))
        self.gates.append(ExitGate_mobile_con(28,height/32,width/32))
        self.gates.append(ExitGate_mobile_con(72,height/32,width/32))
        self.gates.append(ExitGate_mobile_con(180,height/32,width/32))
        self.gates.append(ExitGate_mobile_con(358,height/32,width/32))
        self.gates.append(ExitGate_mobile_con(538,height/32,width/32))  
        
    def forward(self, x):
        base = self.backbone(x) # run the feature layer of mobilnet first. 
        exitgate = 0
        already_exit = False
        reconstructed_pictures = []
        for i in range(9): # 2 channels, 6 channels ...
            encoder_output = self.encoders[i](base)
            if i ==0:
                concat = encoder_output
            else:
                concat = torch.cat((encoder_output, concat), dim=1)
            # run the 2, 6 ... more channels in multiple encoders. get the result. add a mobile net here????? run the mobile model first???
            if (self.gates[i](concat) > 0.5) and (already_exit == False):
                exitgate = i
                already_exit = True                
            reconstructed_pictures.append(self.decoders[i](concat))
        if already_exit == False:
            exitgate = 9
        concat = torch.cat((self.encoders[9](base), concat), dim=1)
        reconstructed_pictures.append(self.decoders[9](concat))
        # reconstructed_pictures are a list of length 10, correspond to the result of 10 encoders.
        # exitgate is a int from 0 to 8 shows which gate to exit
        if exitgate > 5:
            exitgate = 5
        return reconstructed_pictures, exitgate

"""mobile gated non concated"""    
class Encoder_mobile_nc(nn.Module):
    def __init__(self,out_channels):
        super(Encoder_mobile_nc, self).__init__()
        self.encoder = nn.Conv2d(in_channels=1280, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.encoder(x)
    
class Decoder_mobile_nc(nn.Module):
    def __init__(self,starting_channels):
        super(Decoder_mobile_nc, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(starting_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid for pixel values in the range [0, 1]
        )

    def forward(self, x):
        return self.decoder(x)
    
class ExitGate_mobile_nc(nn.Module):
    def __init__(self, in_planes, height, width):
        super().__init__()
        self.pool = nn.AvgPool2d((int(height), int(width)))
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(64, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn0(self.pool(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = torch.flatten(x, 1)
        out = self.linear(x)
        out = self.sigmoid(out)

        return out
    
class mobilenet_gate_eval_nc(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.backbone       = models.mobilenet_v2(pretrained=True)
        self.backbone       = nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        self.encoders = nn.ModuleList()
        self.encoders.append(Encoder_mobile_nc(2))
        self.encoders.append(Encoder_mobile_nc(8))
        self.encoders.append(Encoder_mobile_nc(14))
        self.encoders.append(Encoder_mobile_nc(22))
        self.encoders.append(Encoder_mobile_nc(28))
        self.encoders.append(Encoder_mobile_nc(72))
        self.encoders.append(Encoder_mobile_nc(180))
        self.encoders.append(Encoder_mobile_nc(358))
        self.encoders.append(Encoder_mobile_nc(538))
        self.encoders.append(Encoder_mobile_nc(718))
        
        self.decoders = nn.ModuleList()
        self.decoders.append(Decoder_mobile_nc(2))
        self.decoders.append(Decoder_mobile_nc(8))
        self.decoders.append(Decoder_mobile_nc(14))
        self.decoders.append(Decoder_mobile_nc(22))
        self.decoders.append(Decoder_mobile_nc(28))
        self.decoders.append(Decoder_mobile_nc(72))
        self.decoders.append(Decoder_mobile_nc(180))
        self.decoders.append(Decoder_mobile_nc(358))
        self.decoders.append(Decoder_mobile_nc(538))
        self.decoders.append(Decoder_mobile_nc(718))
        
        self.gates = nn.ModuleList()
        self.gates.append(ExitGate_mobile_nc(2,height/32,width/32))
        self.gates.append(ExitGate_mobile_nc(8,height/32,width/32))
        self.gates.append(ExitGate_mobile_nc(14,height/32,width/32))
        self.gates.append(ExitGate_mobile_nc(22,height/32,width/32))
        self.gates.append(ExitGate_mobile_nc(28,height/32,width/32))
        self.gates.append(ExitGate_mobile_nc(72,height/32,width/32))
        self.gates.append(ExitGate_mobile_nc(180,height/32,width/32))
        self.gates.append(ExitGate_mobile_nc(358,height/32,width/32))
        self.gates.append(ExitGate_mobile_nc(538,height/32,width/32))  
        
    def forward(self, x):
        base = self.backbone(x)
        exitgate = 0
        already_exit = False
        reconstructed_pictures = []
        for i in range(9):
            encoder_output = self.encoders[i](base)
            if (self.gates[i](encoder_output) > 0.5)  and (already_exit == False):
                exitgate = i
                already_exit = True
            reconstructed_pictures.append(self.decoders[i](encoder_output))
        if already_exit == False:
            exitgate = 9
        reconstructed_pictures.append(self.decoders[9](self.encoders[9](base)))
        # reconstructed_pictures are a list of length 10, correspond to the result of 10 encoders.
        # exitgate is the exit situation of 9 gates. >0.5 means exit, <=0.5 means not exit.
        if exitgate > 5:
            exitgate = 5
        return reconstructed_pictures, exitgate

"""resnet50 gated concated"""
class Encode_res50_con(nn.Module):
    def __init__(self,out_channels):
        super(Encode_res50_con, self).__init__()
        self.encoder = nn.Conv2d(in_channels=2048, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.encoder(x)
    
class Decoder_res50_con(nn.Module):
    def __init__(self,starting_channels):
        super(Decoder_res50_con, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(starting_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid for pixel values in the range [0, 1]
        )

    def forward(self, x):
        return self.decoder(x)
    
class ExitGate_res50_con(nn.Module):
    def __init__(self, in_planes, height, width):
        super().__init__()
        self.pool = nn.AvgPool2d((int(height), int(width)))
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(64, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn0(self.pool(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = torch.flatten(x, 1)
        out = self.linear(x)
        out = self.sigmoid(out)

        return out

class res50_concat_gate_eval(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.backbone       = models.resnet50(pretrained=True)
        self.backbone       = nn.Sequential(*list(self.backbone.children())[:-2])
        self.backbone[0]    = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoders = nn.ModuleList()
        self.encoders.append(Encode_res50_con(4))
        self.encoders.append(Encode_res50_con(8))
        self.encoders.append(Encode_res50_con(10))
        self.encoders.append(Encode_res50_con(10))
        self.encoders.append(Encode_res50_con(10))
        self.encoders.append(Encode_res50_con(60))
        self.encoders.append(Encode_res50_con(154))
        self.encoders.append(Encode_res50_con(256))
        self.encoders.append(Encode_res50_con(256))
        self.encoders.append(Encode_res50_con(256))
        
        self.decoders = nn.ModuleList()
        self.decoders.append(Decoder_res50_con(4))
        self.decoders.append(Decoder_res50_con(12))
        self.decoders.append(Decoder_res50_con(22))
        self.decoders.append(Decoder_res50_con(32))
        self.decoders.append(Decoder_res50_con(42))
        self.decoders.append(Decoder_res50_con(102))
        self.decoders.append(Decoder_res50_con(256))
        self.decoders.append(Decoder_res50_con(512))
        self.decoders.append(Decoder_res50_con(768))
        self.decoders.append(Decoder_res50_con(1024))
        
        self.gates = nn.ModuleList()
        self.gates.append(ExitGate_res50_con(4,height/32,width/32))
        self.gates.append(ExitGate_res50_con(12,height/32,width/32))
        self.gates.append(ExitGate_res50_con(22,height/32,width/32))
        self.gates.append(ExitGate_res50_con(32,height/32,width/32))
        self.gates.append(ExitGate_res50_con(42,height/32,width/32))
        self.gates.append(ExitGate_res50_con(102,height/32,width/32))
        self.gates.append(ExitGate_res50_con(256,height/32,width/32))
        self.gates.append(ExitGate_res50_con(512,height/32,width/32))
        self.gates.append(ExitGate_res50_con(768,height/32,width/32)) 
        
    def forward(self, x):
        base = self.backbone(x)
        exitgate = 0
        already_exit = False
        reconstructed_pictures = []
        for i in range(9):
            encoder_output = self.encoders[i](base)
            if i ==0:
                concat = encoder_output
            else:
                concat = torch.cat((encoder_output, concat), dim=1)
            if (self.gates[i](concat) > 0.5) & (already_exit == False):
                exitgate = i
                already_exit = True
            reconstructed_pictures.append(self.decoders[i](concat))
        if already_exit == False:
            exitgate = 9
        concat = torch.cat((self.encoders[9](base), concat), dim=1)
        reconstructed_pictures.append(self.decoders[9](concat))
        # reconstructed_pictures are a list of length 10, correspond to the result of 10 encoders.
        # exitgate is a int from 0 to 8 shows which gate to exit
        if exitgate > 5:
            exitgate = 5
        return reconstructed_pictures, exitgate

"""resnet50 gated nonconcat"""  
class Encoder_res50_nc(nn.Module):
    def __init__(self,out_channels):
        super(Encoder_res50_nc, self).__init__()
        self.encoder = nn.Conv2d(in_channels=2048, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.encoder(x)
    
class Decoder_res50_nc(nn.Module):
    def __init__(self,starting_channels):
        super(Decoder_res50_nc, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(starting_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid for pixel values in the range [0, 1]
        )

    def forward(self, x):
        return self.decoder(x)
    
class ExitGate_res50_nc(nn.Module):
    def __init__(self, in_planes, height, width):
        super().__init__()
        self.pool = nn.AvgPool2d((int(height), int(width)))
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(64, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn0(self.pool(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = torch.flatten(x, 1)
        out = self.linear(x)
        out = self.sigmoid(out)

        return out

class res50_gate_eval_nc(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.backbone       = models.resnet50(pretrained=True)
        self.backbone       = nn.Sequential(*list(self.backbone.children())[:-2])
        self.backbone[0]    = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoders = nn.ModuleList()
        self.encoders.append(Encoder_res50_nc(4))
        self.encoders.append(Encoder_res50_nc(12))
        self.encoders.append(Encoder_res50_nc(22))
        self.encoders.append(Encoder_res50_nc(32))
        self.encoders.append(Encoder_res50_nc(42))
        self.encoders.append(Encoder_res50_nc(102))
        self.encoders.append(Encoder_res50_nc(256))
        self.encoders.append(Encoder_res50_nc(512))
        self.encoders.append(Encoder_res50_nc(768))
        self.encoders.append(Encoder_res50_nc(1024))
        
        self.decoders = nn.ModuleList()
        self.decoders.append(Decoder_res50_nc(4))
        self.decoders.append(Decoder_res50_nc(12))
        self.decoders.append(Decoder_res50_nc(22))
        self.decoders.append(Decoder_res50_nc(32))
        self.decoders.append(Decoder_res50_nc(42))
        self.decoders.append(Decoder_res50_nc(102))
        self.decoders.append(Decoder_res50_nc(256))
        self.decoders.append(Decoder_res50_nc(512))
        self.decoders.append(Decoder_res50_nc(768))
        self.decoders.append(Decoder_res50_nc(1024))
        
        self.gates = nn.ModuleList()
        self.gates.append(ExitGate_res50_nc(4,height/32,width/32))
        self.gates.append(ExitGate_res50_nc(12,height/32,width/32))
        self.gates.append(ExitGate_res50_nc(22,height/32,width/32))
        self.gates.append(ExitGate_res50_nc(32,height/32,width/32))
        self.gates.append(ExitGate_res50_nc(42,height/32,width/32))
        self.gates.append(ExitGate_res50_nc(102,height/32,width/32))
        self.gates.append(ExitGate_res50_nc(256,height/32,width/32))
        self.gates.append(ExitGate_res50_nc(512,height/32,width/32))
        self.gates.append(ExitGate_res50_nc(768,height/32,width/32)) 
        
    def forward(self, x):
        base = self.backbone(x)
        exitgate = 0
        already_exit = False
        reconstructed_pictures = []
        for i in range(9):
            encoder_output = self.encoders[i](base)
            if (self.gates[i](encoder_output) > 0.5)  and (already_exit == False):
                exitgate = i
                already_exit = True
            reconstructed_pictures.append(self.decoders[i](encoder_output))
        if already_exit == False:
            exitgate = 9
        reconstructed_pictures.append(self.decoders[9](self.encoders[9](base)))
        # reconstructed_pictures are a list of length 10, correspond to the result of 10 encoders.
        # exitgate is a int from 0 to 8 shows which gate to exit
        if exitgate > 5:
            exitgate = 5
        return reconstructed_pictures, exitgate
    
    
#ccpd yolo path
img_save_path = "B:\\summer_project\\image_results\\dynamic_yolo\\"
#ccpd_test image path
ccpd_test = "C:\\Users\\Hostl\\Desktop\\autoencoder_data\\CCPD\\ccpd_whole_resize_512x320_grey_test\\"
#inference model paths
inference_models_path = "B:\\summer_project\\pitt_embeddings\\current_code_work\\ccpd_final_paper_fun\\yolo_prec_thres_gate_weights\\"

"""Load all 4 baseline models"""
def get_baseline_4(inference_models_path):
    #nonconcate models
    res50_ccpd_nc = inference_models_path + "res50data_ccpd_epochs-50_batchsize-20_linux-True_gpu-1.pth"
    mobile_ccpd_nc= inference_models_path + "mobiledata_ccpd_epochs-50_batchsize-20_linux-True_gpu-3.pth"
    #concate models
    res50_ccpd_c  = inference_models_path + "res50_CONCATED_data_ccpd_epochs-25_batchsize-20_linux-True_gpu-1.pth" 
    mobile_ccpd_c = inference_models_path + "mobile_CONCATED_data_ccpd_epochs-25_batchsize-20_linux-True_gpu-2.pth"
    #nonconcated dicts
    res50_ccpd_nc_dict  =  torch.load(res50_ccpd_nc,map_location=torch.device('cpu'))
    mobile_ccpd_nc_dict =  torch.load(mobile_ccpd_nc,map_location=torch.device('cpu'))
    #concated dicts
    res50_ccpd_c_dict   =  torch.load(res50_ccpd_c,map_location=torch.device('cpu'))
    mobile_ccpd_c_dict  =  torch.load(mobile_ccpd_c,map_location=torch.device('cpu'))
    #create the nonconcated models
    model_ccpd_res50_nc  = Autoencoder50nc()
    model_ccpd_mobile_nc = Autoencodermobilenetnc()
    #create the concated models
    model_ccpd_res50_c  = Autoencoder50c()
    model_ccpd_mobile_c = Autoencodermobilenetc()
    #Load the dictionaries for nonconcated
    model_ccpd_res50_nc.load_state_dict(res50_ccpd_nc_dict)
    model_ccpd_mobile_nc.load_state_dict(mobile_ccpd_nc_dict)
    #Load the dictionaries for nonconcated
    model_ccpd_res50_c.load_state_dict(res50_ccpd_c_dict)
    model_ccpd_mobile_c.load_state_dict(mobile_ccpd_c_dict)
    return model_ccpd_res50_nc,model_ccpd_mobile_nc,model_ccpd_res50_c,model_ccpd_mobile_c

#model_ccpd_res50_nc,model_ccpd_mobile_nc,model_ccpd_res50_c,model_ccpd_mobile_c = get_baseline_4(inference_models_path)

"""Load all 4 gated models at threshold 0.002"""
def get_gated_4():
    """ccpd models"""
    height = 512
    width  = 320
    """create the ccpd c models"""
    ccpd_m_c = mobilenet_concat_gate_eval(height, width)
    ccpd_r_c = res50_concat_gate_eval(height, width)
    """create the ccpd nc models"""
    ccpd_m_nc= mobilenet_gate_eval_nc(height, width)
    ccpd_r_nc= res50_gate_eval_nc(height, width)   
    #inference model paths
    inference_models_path = "B:\\summer_project\\pitt_embeddings\\current_code_work\\ccpd_final_paper_fun\\yolo_prec_thres_gate_weights\\"
    """CCPD dicts"""
    """Concated"""
    ccpd_dict_m_c = inference_models_path + "mobilenet_concat_ccpd_gate_yolo_precision_dynamic.pth"
    ccpd_dict_r_c = inference_models_path + "res50_concat_ccpd_gate_yolo_precision_dynamic.pth"
    """Noncated"""
    ccpd_dict_m_nc = inference_models_path + "mobilenet_nonconcat_ccpd_gate_yolo_precision_dynamic.pth"
    ccpd_dict_r_nc = inference_models_path + "res50_nonconcat_ccpd_gate_yolo_precision_dynamic.pth"  
    """CCPD dicts"""
    """Concated"""
    ccpd_dict_m_c = torch.load(ccpd_dict_m_c,map_location=torch.device('cpu'))
    ccpd_dict_r_c = torch.load(ccpd_dict_r_c,map_location=torch.device('cpu'))
    """Noncated"""
    ccpd_dict_m_nc = torch.load(ccpd_dict_m_nc,map_location=torch.device('cpu'))
    ccpd_dict_r_nc = torch.load(ccpd_dict_r_nc,map_location=torch.device('cpu'))     
    """ccpd models"""
    """Load the ccpd c models"""
    ccpd_m_c.load_state_dict(ccpd_dict_m_c)
    ccpd_r_c.load_state_dict(ccpd_dict_r_c)
    """Load the ccpd nc models"""
    ccpd_m_nc.load_state_dict(ccpd_dict_m_nc)
    ccpd_r_nc.load_state_dict(ccpd_dict_r_nc)   
    return ccpd_m_c,ccpd_r_c,ccpd_m_nc,ccpd_r_nc

ccpd_m_c,ccpd_r_c,ccpd_m_nc,ccpd_r_nc = get_gated_4()
model_ccpd_mobile_c_gated  = ccpd_m_c
model_ccpd_mobile_nc_gated = ccpd_m_nc
model_ccpd_res50_c_gated   = ccpd_r_c
model_ccpd_res50_nc_gated  = ccpd_r_nc

device = 'cuda'
"""Load the baseline models to the current device"""
#model_ccpd_res50_nc.to(device)
#model_ccpd_mobile_nc.to(device)
#model_ccpd_res50_c.to(device)
#model_ccpd_mobile_c.to(device)
"""Load the gated models to the current device"""
model_ccpd_mobile_c_gated.to(device)
model_ccpd_mobile_nc_gated.to(device)
model_ccpd_res50_c_gated.to(device)
model_ccpd_res50_nc_gated.to(device)

"""change all models to eval mode"""
#model_ccpd_res50_nc.eval()
#model_ccpd_mobile_nc.eval()
#model_ccpd_res50_c.eval()
#model_ccpd_mobile_c.eval()
model_ccpd_mobile_c_gated.eval()
model_ccpd_mobile_nc_gated.eval()
model_ccpd_res50_c_gated.eval()
model_ccpd_res50_nc_gated.eval()

"""Load the ccpd dataset"""
image_height = 512
image_width  = 320
ccpd_test_array = []
for im_name in tqdm.tqdm(os.listdir(ccpd_test)):
    image =  Image.open(ccpd_test+im_name).convert("L")
    image = np.array(image,dtype="float16")/255.
    image = np.reshape(image,(1,1,image_height,image_width))    
    ccpd_test_array.append(image)    
ccpd_test_array = np.array(ccpd_test_array,dtype="float32")

#mob_c  10 gates
#mob_nc 10 gates
#r50_c  10 gates
#r50_nc 10 gates

#mob_c  10 comp
#mob_nc 10 comp
#r50_c  10 comp
#r50_nc 10 comp



try:
    os.mkdir(img_save_path+"gated_mob_c")
    os.mkdir(img_save_path+"gated_mob_nc")
    os.mkdir(img_save_path+"gated_r50_c")
    os.mkdir(img_save_path+"gated_r50_nc")
    
    os.mkdir(img_save_path+"gated_mob_c_0")
    os.mkdir(img_save_path+"gated_mob_c_1")
    os.mkdir(img_save_path+"gated_mob_c_2")
    os.mkdir(img_save_path+"gated_mob_c_3")
    os.mkdir(img_save_path+"gated_mob_c_4")
    os.mkdir(img_save_path+"gated_mob_c_5")
    os.mkdir(img_save_path+"gated_mob_c_6")
    os.mkdir(img_save_path+"gated_mob_c_7")
    os.mkdir(img_save_path+"gated_mob_c_8")
    os.mkdir(img_save_path+"gated_mob_c_9")
    
    os.mkdir(img_save_path+"gated_mob_nc_0")
    os.mkdir(img_save_path+"gated_mob_nc_1")
    os.mkdir(img_save_path+"gated_mob_nc_2")
    os.mkdir(img_save_path+"gated_mob_nc_3")
    os.mkdir(img_save_path+"gated_mob_nc_4")
    os.mkdir(img_save_path+"gated_mob_nc_5")
    os.mkdir(img_save_path+"gated_mob_nc_6")
    os.mkdir(img_save_path+"gated_mob_nc_7")
    os.mkdir(img_save_path+"gated_mob_nc_8")
    os.mkdir(img_save_path+"gated_mob_nc_9")

    os.mkdir(img_save_path+"gated_r50_c_0")
    os.mkdir(img_save_path+"gated_r50_c_1")
    os.mkdir(img_save_path+"gated_r50_c_2")
    os.mkdir(img_save_path+"gated_r50_c_3")
    os.mkdir(img_save_path+"gated_r50_c_4")
    os.mkdir(img_save_path+"gated_r50_c_5")
    os.mkdir(img_save_path+"gated_r50_c_6")
    os.mkdir(img_save_path+"gated_r50_c_7")
    os.mkdir(img_save_path+"gated_r50_c_8")
    os.mkdir(img_save_path+"gated_r50_c_9")
    
    os.mkdir(img_save_path+"gated_r50_nc_0")
    os.mkdir(img_save_path+"gated_r50_nc_1")
    os.mkdir(img_save_path+"gated_r50_nc_2")
    os.mkdir(img_save_path+"gated_r50_nc_3")
    os.mkdir(img_save_path+"gated_r50_nc_4")
    os.mkdir(img_save_path+"gated_r50_nc_5")
    os.mkdir(img_save_path+"gated_r50_nc_6")
    os.mkdir(img_save_path+"gated_r50_nc_7")
    os.mkdir(img_save_path+"gated_r50_nc_8")
    os.mkdir(img_save_path+"gated_r50_nc_9")
    
    """
    os.mkdir(img_save_path+"base_mob_c_0")
    os.mkdir(img_save_path+"base_mob_c_1")
    os.mkdir(img_save_path+"base_mob_c_2")
    os.mkdir(img_save_path+"base_mob_c_3")
    os.mkdir(img_save_path+"base_mob_c_4")
    os.mkdir(img_save_path+"base_mob_c_5")
    os.mkdir(img_save_path+"base_mob_c_6")
    os.mkdir(img_save_path+"base_mob_c_7")
    os.mkdir(img_save_path+"base_mob_c_8")
    os.mkdir(img_save_path+"base_mob_c_9")
    
    os.mkdir(img_save_path+"base_mob_nc_0")
    os.mkdir(img_save_path+"base_mob_nc_1")
    os.mkdir(img_save_path+"base_mob_nc_2")
    os.mkdir(img_save_path+"base_mob_nc_3")
    os.mkdir(img_save_path+"base_mob_nc_4")
    os.mkdir(img_save_path+"base_mob_nc_5")
    os.mkdir(img_save_path+"base_mob_nc_6")
    os.mkdir(img_save_path+"base_mob_nc_7")
    os.mkdir(img_save_path+"base_mob_nc_8")
    os.mkdir(img_save_path+"base_mob_nc_9")
    
    os.mkdir(img_save_path+"base_r50_c_0")
    os.mkdir(img_save_path+"base_r50_c_1")
    os.mkdir(img_save_path+"base_r50_c_2")
    os.mkdir(img_save_path+"base_r50_c_3")
    os.mkdir(img_save_path+"base_r50_c_4")
    os.mkdir(img_save_path+"base_r50_c_5")
    os.mkdir(img_save_path+"base_r50_c_6")
    os.mkdir(img_save_path+"base_r50_c_7")
    os.mkdir(img_save_path+"base_r50_c_8")
    os.mkdir(img_save_path+"base_r50_c_9")
    
    os.mkdir(img_save_path+"base_r50_nc_0")
    os.mkdir(img_save_path+"base_r50_nc_1")
    os.mkdir(img_save_path+"base_r50_nc_2")
    os.mkdir(img_save_path+"base_r50_nc_3")
    os.mkdir(img_save_path+"base_r50_nc_4")
    os.mkdir(img_save_path+"base_r50_nc_5")
    os.mkdir(img_save_path+"base_r50_nc_6")
    os.mkdir(img_save_path+"base_r50_nc_7")
    os.mkdir(img_save_path+"base_r50_nc_8")
    os.mkdir(img_save_path+"base_r50_nc_9")
    """
except Exception:
    pass

def imgsave(tmp,name,c):
    tmp = tmp.to('cpu')
    tmp = (tmp.detach().numpy() * 255)
    tmp = np.reshape(tmp,(512,320))
    img = Image.fromarray(tmp)
    img.convert("L").save(name + "img_" + str(c) + ".jpg")   

c= 0
for image in tqdm.tqdm(ccpd_test_array):      
    #load the image
    image      = np.reshape(image,(1,1,image_height,image_width))    
    image      = torch.from_numpy(image).float().to(device)
    
    """Gated models"""
    """Mobile c"""
    res,gate = model_ccpd_mobile_c_gated.forward(image)
    res      = res[gate]
    name     = img_save_path+"gated_mob_c\\" 
    imgsave(res, name,c)
    name     = img_save_path+"gated_mob_c_" + str(gate) + "\\" 
    imgsave(res, name,c)
    """Mobile nc"""
    res,gate = model_ccpd_mobile_nc_gated.forward(image)
    res      = res[gate]  
    name     = img_save_path+"gated_mob_nc\\"
    imgsave(res, name,c)
    name     = img_save_path+"gated_mob_nc_" + str(gate) + "\\" 
    imgsave(res, name,c)
    """Res50 c"""
    res,gate = model_ccpd_res50_c_gated.forward(image)
    res      = res[gate]  
    name     = img_save_path+"gated_r50_c\\"
    imgsave(res, name,c)
    name     = img_save_path+"gated_r50_c_" + str(gate) + "\\" 
    imgsave(res, name,c)
    """Res50 nc"""
    res,gate = model_ccpd_res50_nc_gated.forward(image)
    res      = res[gate]  
    name     = img_save_path+"gated_r50_nc\\"
    imgsave(res, name,c)
    name     = img_save_path+"gated_r50_nc_" + str(gate) + "\\" 
    imgsave(res, name,c)
    
    """Baseline models"""    
    """
    res_100,res_75,res_50,res_25,res_10,res_4,res_3,res_2,res_1,res_p40 = model_ccpd_mobile_c(image)
    base = "base_mob_c"
    name     = img_save_path+base + "_0\\"
    imgsave(res_p40, name,c)
    name     = img_save_path+base + "_1\\"
    imgsave(res_1, name,c)
    name     = img_save_path+base + "_2\\"
    imgsave(res_2, name,c)
    name     = img_save_path+base + "_3\\"
    imgsave(res_3, name,c)
    name     = img_save_path+base + "_4\\"
    imgsave(res_4, name,c)
    name     = img_save_path+base + "_5\\"
    imgsave(res_10, name,c)
    name     = img_save_path+base + "_6\\"
    imgsave(res_25, name,c)
    name     = img_save_path+base + "_7\\"
    imgsave(res_50, name,c)
    name     = img_save_path+base + "_8\\"
    imgsave(res_75, name,c)
    name     = img_save_path+base + "_9\\"
    imgsave(res_100, name,c)    
    res_100,res_75,res_50,res_25,res_10,res_4,res_3,res_2,res_1,res_p40 = model_ccpd_mobile_nc(image)
    base = "base_mob_nc"
    name     = img_save_path+base + "_0\\"
    imgsave(res_p40, name,c)
    name     = img_save_path+base + "_1\\"
    imgsave(res_1, name,c)
    name     = img_save_path+base + "_2\\"
    imgsave(res_2, name,c)
    name     = img_save_path+base + "_3\\"
    imgsave(res_3, name,c)
    name     = img_save_path+base + "_4\\"
    imgsave(res_4, name,c)
    name     = img_save_path+base + "_5\\"
    imgsave(res_10, name,c)
    name     = img_save_path+base + "_6\\"
    imgsave(res_25, name,c)
    name     = img_save_path+base + "_7\\"
    imgsave(res_50, name,c)
    name     = img_save_path+base + "_8\\"
    imgsave(res_75, name,c)
    name     = img_save_path+base + "_9\\"
    imgsave(res_100, name,c)
    res_100,res_75,res_50,res_25,res_10,res_4,res_3,res_2,res_1,res_p40 = model_ccpd_res50_c(image)
    base = "base_r50_c"
    name     = img_save_path+base + "_0\\"
    imgsave(res_p40, name,c)
    name     = img_save_path+base + "_1\\"
    imgsave(res_1, name,c)
    name     = img_save_path+base + "_2\\"
    imgsave(res_2, name,c)
    name     = img_save_path+base + "_3\\"
    imgsave(res_3, name,c)
    name     = img_save_path+base + "_4\\"
    imgsave(res_4, name,c)
    name     = img_save_path+base + "_5\\"
    imgsave(res_10, name,c)
    name     = img_save_path+base + "_6\\"
    imgsave(res_25, name,c)
    name     = img_save_path+base + "_7\\"
    imgsave(res_50, name,c)
    name     = img_save_path+base + "_8\\"
    imgsave(res_75, name,c)
    name     = img_save_path+base + "_9\\"
    imgsave(res_100, name,c)
    res_100,res_75,res_50,res_25,res_10,res_4,res_3,res_2,res_1,res_p40 = model_ccpd_res50_nc(image)
    base = "base_r50_nc"
    name     = img_save_path+base + "_0\\"
    imgsave(res_p40, name,c)
    name     = img_save_path+base + "_1\\"
    imgsave(res_1, name,c)
    name     = img_save_path+base + "_2\\"
    imgsave(res_2, name,c)
    name     = img_save_path+base + "_3\\"
    imgsave(res_3, name,c)
    name     = img_save_path+base + "_4\\"
    imgsave(res_4, name,c)
    name     = img_save_path+base + "_5\\"
    imgsave(res_10, name,c)
    name     = img_save_path+base + "_6\\"
    imgsave(res_25, name,c)
    name     = img_save_path+base + "_7\\"
    imgsave(res_50, name,c)
    name     = img_save_path+base + "_8\\"
    imgsave(res_75, name,c)
    name     = img_save_path+base + "_9\\"
    imgsave(res_100, name,c)    
    """
    c+=1











