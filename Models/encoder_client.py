import torch.nn as nn
import torch

class Encoder_Client(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder_Client, self).__init__()
        
        # simple two layers CNN
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh(out)

        return out
