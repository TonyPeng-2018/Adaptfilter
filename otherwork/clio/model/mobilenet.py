import torch
import torchvision
from torch import nn
from torchvision import transforms, models


class mobilenet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.backbone = models.mobilenet_v2()
        self.backbone = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        # self.sig = nn.Sigmoid()
        
        
    def forward(self, x):
        # print(x.size())
        base = self.backbone(x)
        # base = self.sig(base)
        return base