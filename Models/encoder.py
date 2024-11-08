import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, in_ch, num_of_layers = 2):
        super(Encoder, self).__init__()
        
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
    
class Encoder_Gate(nn.Module):
    def __init__(self, in_ch):
        super(Encoder_Gate, self).__init__()
        
        # reduce channel in_ch to out_ch
        self.in_ch = in_ch
        encoder_gate_sequence = []
        while self.in_ch >= 1:
            encoder_gate_sequence.append(nn.Conv2d(in_channels=self.in_ch, out_channels=self.in_ch//2, kernel_size=1))
            encoder_gate_sequence.append(nn.BatchNorm2d(in_ch))
            encoder_gate_sequence.append(nn.ReLU(True))
            self.in_ch = self.in_ch//2
        