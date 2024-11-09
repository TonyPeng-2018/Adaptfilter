import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        
        # simple two layers CNN
        # down sampling, this is for square image
        # suppose increase it from 28, 28 to 56, 56, or 16, 16
        self.in_ch = in_ch
        self.out_ch = out_ch
        decoders = []
        
        decoders.append(nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=1, padding=1))
        decoders.append(nn.BatchNorm2d(self.out_ch))
        decoders.append(nn.ReLU(True))

        self.decoder = nn.Sequential(*decoders)

    def forward(self, x):
        return self.decoder(x)
