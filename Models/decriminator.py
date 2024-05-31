import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, inputsize, hiddensize, outputsize, positionsize):
        super(Discriminator, self).__init__()
        self.inputsize = inputsize # 8, 16, 24
        self.outputsize = outputsize
        self.hiddensize = hiddensize
        self.positionsize = positionsize
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(inputsize, hiddensize, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(hiddensize, hiddensize*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(hiddensize * 2, hiddensize * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(hiddensize * 4, hiddensize * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(hiddensize * 8, inputsize, 3, 2, 1, bias=False),
            # nn.BatchNorm2d(hiddensize * 8),
            # nn.LeakyReLU(0.2, inplace=True),
        )

        self.position_encoder = nn.Sequential(
            nn.Linear(positionsize, 32),
            nn.ReLU(True)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(32+4096, 1),
            nn.Sigmoid()
        )
    def forward(self, input, input2):
        out1 = self.main(input) # hiddensize * n, ? ,32 ,32
        out2 = self.position_encoder(input2) # 32 -> 32
        # flatten and concatenate the two features 
        out1 = out1.flatten(start_dim=1)
        out = torch.cat((out1, out2), 1)
        output = self.output_layer(out)
        return output.view(-1, 1).squeeze(1)