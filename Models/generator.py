import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, inputsize, hiddensize, outputsize):
        super(Generator, self).__init__()
        self.inputsize = inputsize # 8, 16, 24
        self.outputsize = outputsize
        self.hiddensize = hiddensize
        self.section1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.inputsize, hiddensize * 8, 3, 1, padding=1, bias=False, dilation=1),
            nn.BatchNorm2d(hiddensize * 8),
            nn.ReLU(True)
        )
        self.section2 = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(hiddensize * 8, hiddensize * 4, 3, 1, padding=1 , bias=False, dilation=1),
            nn.BatchNorm2d(hiddensize * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(hiddensize * 4, hiddensize * 2, 3, 1, padding=1, bias=False, dilation=1),
            nn.BatchNorm2d(hiddensize * 2),
            nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(hiddensize * 2, hiddensize, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(hiddensize),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(hiddensize * 2, self.outputsize, 3, 1, padding=1, bias=False, dilation=1),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.section1(input)
        output = self.section2(output)
        return output