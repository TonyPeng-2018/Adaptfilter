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

class Decoder_Pyramid(nn.Module):
    def __init__(self, out_ch, min_ch):
        super(Decoder_Pyramid, self).__init__()
        # reduce channel in_ch to out_ch
        self.in_ch = min_ch
        self.out_ch = out_ch
        decoders = []
        while self.in_ch < self.out_ch:
            new_decoders = []
            new_decoders.append(nn.Conv2d(in_channels=self.in_ch, out_channels=self.in_ch*2, kernel_size=1))
            new_decoders.append(nn.BatchNorm2d(self.in_ch*2))
            new_decoders.append(nn.ReLU(True))
            new_decoders = nn.Sequential(*new_decoders)
            self.in_ch = self.in_ch*2
            decoders.append(new_decoders)
        self.ch_list = [2**x for x in range (len(decoders))]
        self.decoders = nn.Sequential(*decoders)

    def forward(self, x):

        x_channel = x.size(1)
        for ind, decoder in enumerate(self.decoders):
            if x_channel <= self.ch_list[ind]:
                x = decoder(x)
        return x

class Decoder_Pyramid_Heavy(nn.Module):
    def __init__(self, out_ch, min_ch):
        super(Decoder_Pyramid_Heavy, self).__init__()
        # reduce channel in_ch to out_ch
        self.out_ch = out_ch
        start_ch = min_ch
        decoders = []
        while start_ch < self.out_ch:
            new_decoders = []
            new_decoders.append(nn.Conv2d(in_channels=start_ch, out_channels=start_ch*2, kernel_size=1))
            new_decoders.append(nn.BatchNorm2d(start_ch*2))
            new_decoders.append(nn.ReLU(True))
            new_decoders.append(nn.Conv2d(in_channels=start_ch*2, out_channels=start_ch*2, kernel_size=3, padding=1))
            new_decoders.append(nn.BatchNorm2d(start_ch*2))
            new_decoders.append(nn.ReLU(True))

            new_decoders = nn.Sequential(*new_decoders)
            start_ch = start_ch*2
            decoders.append(new_decoders)
        self.ch_list = [2**x for x in range (len(decoders))]
        self.decoders = nn.Sequential(*decoders)

    def forward(self, x):

        x_channel = x.size(1)
        # print(x.size())
        for i, decoder in enumerate(self.decoders):
            # print(self.ch_list[i])
            if x_channel <= self.ch_list[i]: 
                x = decoder(x)
        return x