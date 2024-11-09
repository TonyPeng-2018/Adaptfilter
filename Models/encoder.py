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

        