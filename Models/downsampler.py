import torch.nn as nn
import torch

class Downsampler(nn.Module):
    def __init__(self, in_ch, num_of_layers = 2):
        super(Downsampler, self).__init__()
        
        # simple two layers CNN
        # down sampling, this is for square image
        # suppose reduce it from 56, 56 to 28, 28, or 16, 16
        self.in_ch = in_ch
        encoder_sequence = []
        for i in range(num_of_layers):
            encoder_sequence.append(nn.Conv2d(in_channels=self.in_ch, out_channels=2*self.in_ch, kernel_size=3, stride=2, padding=1))
            encoder_sequence.append(nn.BatchNorm2d(2*self.in_ch))
            encoder_sequence.append(nn.ReLU(True))
            self.in_ch = 2*self.in_ch
        self.encoder = nn.Sequential(*encoder_sequence)

    def forward(self, x):
        return self.encoder(x)