import torch.nn as nn
import torch

class Upsampler(nn.Module):
    def __init__(self, in_ch, num_of_layers = 2):
        super(Upsampler, self).__init__()
        
        # simple two layers CNN
        # down sampling, this is for square image
        # suppose increase it from 28, 28 to 56, 56, or 16, 16
        self.in_ch = in_ch
        decoder_sequence = []
        for i in range(num_of_layers):
            decoder_sequence.append(nn.ConvTranspose2d(in_channels=self.in_ch, out_channels=self.in_ch//2, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_sequence.append(nn.BatchNorm2d(self.in_ch//2))
            decoder_sequence.append(nn.ReLU(True))
            self.in_ch = self.in_ch//2
        self.decoder = nn.Sequential(*decoder_sequence)

    def forward(self, x):
        return self.decoder(x)