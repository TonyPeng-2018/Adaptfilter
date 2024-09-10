# add a squeeze layer to the model
import torch
import torch.nn as nn
class depressor(nn.Module):
    def __init__(self, inputsize, outputsize):
        # ex input 16, output 4
        super().__init__()
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.section1 = nn.Sequential(
            nn.Conv2d(self.inputsize, self.outputsize, 3, 1, padding = 1),
            nn.BatchNorm2d(self.outputsize),
        )
    def forward(self, input):
        output = self.section1(input)
        return output
# https://github.com/Lornatang/CGAN-PyTorch/blob/master/cgan_pytorch/models/generator.py
class Generator(nn.Module):
    def __init__(self, image_size: int = 28, channels: int = 1, num_channels: int = 32, input_channels: int = 1, batch_size: int = 128):   
        # ex input 4, output 1
        super().__init__()
        self.image_size = image_size
        self.channels = channels

        self.label_embedding = nn.Embedding(num_channels, num_channels)
        self.input_size = self.image_size[0] * self.image_size[1]
        self.input_size += num_channels
        self.input_channels = input_channels
        self.output_size = self.image_size[0] * self.image_size[1]
        
        self.pre = nn.Sequential(
            nn.Conv2d(self.input_channels, self.input_channels//2, 3, 1, padding=1),
            nn.BatchNorm2d(self.input_channels//2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(self.input_channels//2, 1, 3, 1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        if batch_size > 1:
            self.main = nn.Sequential(
                nn.Linear(self.input_size, 2*self.input_size),
                nn.BatchNorm1d(2*self.input_size),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.Linear(2*self.input_size, 4*self.input_size),
                nn.BatchNorm1d(4*self.input_size),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.Linear(4*self.input_size, self.output_size),
                nn.Tanh()
            )
        else:
            self.main = nn.Sequential(
                nn.Linear(self.input_size, 2*self.input_size),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.Linear(2*self.input_size, 4*self.input_size),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.Linear(4*self.input_size, self.output_size),
                nn.Sigmoid()
            )

    def forward(self, inputs: torch.Tensor, labels: list) -> torch.Tensor:
        out = self.pre(inputs)
        out = out.view(out.size(0), -1)
        conditional_inputs = torch.cat([out, self.label_embedding(labels)], dim=-1)
        out = self.main(conditional_inputs)
        out = out.reshape(inputs.size(0), self.image_size[0], self.image_size[1])
        return out

# import client and server
from Models.mobilenetv2 import mobilenetv2_splitter, MobileNetV2
client, server = mobilenetv2_splitter(num_classes = 10, weight_root='/home/tonypeng/Workspace1/adaptfilter/Adaptfilter/Weights/cifar-10', device='cuda:0', partition=-1)
from Dataloaders import dataloader_cifar10
train, test, classes = dataloader_cifar10.Dataloader_cifar10(train_batch=128, test_batch=100)
client = client.to('cuda:0')
server = server.to('cuda:0')
client.eval()
server.eval()

depress = depressor(32, 4)
depress = depress.to('cuda:0')
depress.train()

generator = Generator(image_size=[16,16],input_channels=4)
generator = generator.to('cuda:0')
generator.train()

import torch.optim as optim
criterion = nn.MSELoss()
d_optimizer = optim.Adam(depress.parameters(), lr=0.001)
g_optimizer = optim.Adam(generator.parameters(), lr=0.001)

import numpy as np
def get_zero_rank(input):
    # input is torch
    output = input.clone()
    inputsize = input.size(2) * input.size(3)# 4*4
    zero_rate = torch.zeros(input.shape[:2]) # b,c
    zero_rank = torch.zeros(input.shape[:2]) # b,c
    for i, c in enumerate(input):
        for j in range(input[i].shape[0]): # c
            zeros = torch.where(input[i,j,:,:]<1e-4, 1, 0)
            zero_rate[i,j] = torch.sum(zeros)/inputsize
            if zero_rate[i,j] > 0.5:
                zero_rate[i,j] = 1
                output[i,j] = 0
        # sort the zeros rate 
    zero_rank = torch.argsort(zero_rate) # descend 
    one_hot = np.where(zero_rate>0.5, 0, 1)

    return zero_rank, output, one_hot

def normalization(tensor):
    pass

from tqdm import tqdm
for epoch in tqdm(range(10)):
    running_loss = 0.0
    for i, data in enumerate(train, 0):
        inputs, _ = data
        inputs = inputs.to('cuda:0')
        labels = client(inputs)

        # get the zero rank
        outputs = labels.to('cpu')
        zero_rank, outputs, one_hot = get_zero_rank(outputs)
        # print(outputs.size())
        # outputs = outputs[zero_rank, :, :]
        # print(outputs.size())
        outputs = outputs.to('cuda:0') # b,c,h,w
        ont_hot = torch.tensor(one_hot).to('cuda:0')

        d_optimizer.zero_grad()
        outputs = depress(labels)

        # generator 
        g_optimizer.zero_grad()

        # label1 = torch.zeros((labels.size(0), labels.size(1)), dtype=torch.int)
        # label2 = torch.ones((labels.size(0), labels.size(1)), dtype=torch.int)
        # for j in range(labels.size(0)):
        #     label1[j,0] = zero_rank[j,0]
        #     label2[j,10] = zero_rank[j,10]
        label1 = torch.ones((labels.size(0)), dtype=torch.int).cuda()
        print(label1.size())
        # label2 = torch.tensor([10]).cuda()
        
        outputs1 = generator(outputs, label1)
        # outputs2 = generator(outputs, label2)
        loss1 = criterion(outputs1, labels[:,0,:,:])
        # loss2 = criterion(outputs2, labels)
        # loss2.backward()
        d_optimizer.step()
        g_optimizer.step()

        running_loss += loss1.item()
        print('[%d, %5d] loss: %.10f' % (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

        torch.save(depress.state_dict(), '/home/tonypeng/Workspace1/adaptfilter/Adaptfilter/Weights/depressor.pth')
        torch.save(generator.state_dict(), '/home/tonypeng/Workspace1/adaptfilter/Adaptfilter/Weights/generator.pth')
        


        