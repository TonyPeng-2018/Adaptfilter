# Originally copied from https://blog.csdn.net/Haoyee1/article/details/124740736

import torch.nn as nn
import torch

class bottleneck1(nn.Module):
    expansion=1
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(bottleneck1, self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
 
        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.downsample=downsample
 
    def forward(self,x):
        a=x
        if self.downsample is True:
            a=self.downsample(x)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x+=a
        x=self.relu(x)
        return x
 
# this is for 50, 101 and 152
class bottleneck2(nn.Module):
    expansion=4
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(bottleneck2,self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        # the third layer is extended.
        self.conv3=nn.Conv2d(in_channels=out_channels,out_channels=out_channels*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels*self.expansion)
        self.downsample=downsample
        self.relu=nn.ReLU(inplace=True)
 
    def forward(self,x):
        a=x
        if self.downsample is True:
            a=self.downsample(x)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x+=a
        x=self.relu(x)
 
 
class resnet(nn.Module):
    in_channel = 64
    def __init__(self,block,block_num,num_classes=1000):
        super(resnet,self).__init__()
        #假设输入图片大小为600x600x3
        #600x600x3-->300x300x64
        # the size of the input is 224x224x3
        self.conv1=nn.Conv2d(3,self.in_channel,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn=nn.BatchNorm2d(self.in_channel)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)
        self.layer1=self.makelayer(block=block,channel=64,block_num=block_num[0])
        self.layer2=self.makelayer(block=block,channel=128,block_num=block_num[1],stride=2)
        self.layer3=self.makelayer(block=block,channel=256,block_num=block_num[2],stride=2)
        self.layer4=self.makelayer(block=block,channel=512,block_num=block_num[3],stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
 
 
    #block use bottleneck1 or bottleneck2
    #channel is the input layer channel
    #block_num is the number of blocks in each layer
    def makelayer(self,block,channel,block_num,stride=1):
        downsample=None
        #如果步距不为1则代表有残差结构或者expension不为1也有
        if stride!=1 or self.in_channel!=channel*block.expansion:
           downsample=nn.Sequential(nn.Conv2d(in_channels=self.in_channel,out_channels=channel*block.expansion,kernel_size=1,stride=stride,bias=False),
                                    nn.BatchNorm2d(channel*block.expansion))
        #把第一层的结构放到列表里
        layers=[]
        layers.append(block(self.in_channel,channel,stride,downsample))
        #第二层的输入是第一层的输出
        self.in_channel=channel*block.expansion
        for i in range(1,block_num):
            layers.append(block(self.in_channel,channel))
        return nn.Sequential(*layers)
 
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.maxpooling(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.avgpool(x)
        x=torch.flatten(x,dims=1)
        x=self.fc(x)
        return x
 
 
net=resnet(block=bottleneck2,block_num=[3,4,6,3])
 
print(net)