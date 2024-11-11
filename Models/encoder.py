import torch.nn as nn
import torch
    
class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder, self).__init__()
        
        # reduce channel in_ch to out_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        encoders = []
    #     while self.in_ch >= 1:
    #         new_encoder = []
    #         new_encoder.append(nn.Conv2d(in_channels=self.in_ch, out_channels=self.in_ch//2, kernel_size=1))
    #         new_encoder.append(nn.BatchNorm2d(in_ch))
    #         new_encoder.append(nn.ReLU(True))
    #         new_encoder = nn.Sequential(*new_encoder)
    #         self.in_ch = self.in_ch//2
    #         encoders.append(new_encoder)
    #     encoders = nn.Sequential(*encoders)

        encoders.append(nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=1))
        encoders.append(nn.BatchNorm2d(self.out_ch))
        encoders.append(nn.ReLU(True))
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
    #     result_list = []
    #     for encoder in self.encoders:
    #         x = encoder(x)
    #         result_list.append(x)

    #     return result_list
        return self.encoders(x)

class Encoder_Pyramid(nn.Module):
    def __init__(self, in_ch, min_ch):
        super(Encoder_Pyramid, self).__init__()
        
        # reduce channel in_ch to out_ch
        self.in_ch = in_ch
        start_ch = in_ch
        encoders = []
        while start_ch >= min_ch:
            new_encoder = []
            new_encoder.append(nn.Conv2d(in_channels=start_ch, out_channels=start_ch//2, kernel_size=1))
            new_encoder.append(nn.BatchNorm2d(start_ch//2))
            new_encoder.append(nn.ReLU(True))
            new_encoder = nn.Sequential(*new_encoder)
            start_ch = start_ch//2
            encoders.append(new_encoder)
        self.encoders = nn.Sequential(*encoders)

        # encoders.append(nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=1))
        # encoders.append(nn.BatchNorm2d(self.out_ch))
        # encoders.append(nn.ReLU(True))
        # self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        result_list = []
        for encoder in self.encoders:
            x = encoder(x)
            result_list.append(x)

        return result_list
        # return self.encoders(x)

class Encoder_Pyramid_Heavy(nn.Module):
    def __init__(self, in_ch, min_ch):
        super(Encoder_Pyramid_Heavy, self).__init__()
        
        # reduce channel in_ch to out_ch
        self.in_ch = in_ch
        start_ch = in_ch
        encoders = []
        while start_ch > min_ch:
            new_encoder = []
            new_encoder.append(nn.Conv2d(in_channels=start_ch, out_channels=start_ch, kernel_size=3))
            new_encoder.append(nn.BatchNorm2d(start_ch))
            new_encoder.append(nn.ReLU(True))
            new_encoder.append(nn.Conv2d(in_channels=start_ch, out_channels=start_ch//2, kernel_size=1))
            new_encoder.append(nn.BatchNorm2d(start_ch//2))
            new_encoder.append(nn.ReLU(True))
            new_encoder = nn.Sequential(*new_encoder)
            start_ch = start_ch//2
            encoders.append(new_encoder)
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        result_list = []
        for encoder in self.encoders:
            x = encoder(x)
            result_list.append(x)

        return result_list
