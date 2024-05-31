import torch
from torch import nn
from torchvision import models
import torchvision


slice_num = [1, 2, 4, 8, 12, 16, 20, 24]

class ExitGate(nn.Module):
    def __init__(self, in_planes, height, width):
        super().__init__()
        self.conv0 = nn.Conv2d(in_planes, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.pool = nn.AvgPool2d((int(height), int(width)))
        self.bn0 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(64, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn(self.conv0(x)))
        x = self.relu(self.bn0(self.pool(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = torch.flatten(x, 1)
        out = self.linear(x)
        out = self.sigmoid(out)

        return out
    
class mobilenet_slice(nn.Module):
    def __init__(self):
        super(mobilenet_slice, self).__init__()
        # Load the pre-trained MobileNet V2 model only once
        mobilenet_v2 = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        
        # Extract the features from the pre-trained model
        features = mobilenet_v2.features
        
        # Create part1 by slicing the first three layers of the MobileNet V2 features
        self.part1 = nn.Sequential(*features[:4])
        
        # Initialize ModuleList for the second part
        self.part2 = nn.ModuleList()
        
        self.upscale_layers = nn.ModuleList()
        
        # The rest of the features and the classifier are shared among slice_num entries
        remaining_features = nn.Sequential(*features[4:])
        classifier = mobilenet_v2.classifier
        
        # Create part2 modules for each entry in slice_num
        for i in slice_num:
            self.part2.append(nn.Sequential(
                remaining_features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                classifier
            ))
            self.upscale_layers.append(nn.Conv2d(in_channels=i, out_channels=24, kernel_size=1, stride=1, padding=0))
        
    def forward(self, x):
        # print(base.size())
        base = self.part1(x)
        # print(base.size())
        out = []
        # Apply each part2 module to the output of part1
        for i in range(len(slice_num)):
            sliced_base = base[:, :slice_num[i], :, :]
            upscaled_base = self.upscale_layers[i](sliced_base)
            result = self.part2[i](upscaled_base)
            out.append(result)
        return out
        
class mobilenet_slice_train(nn.Module):
    def __init__(self):
        super(mobilenet_slice_train, self).__init__()
        # Load the pre-trained MobileNet V2 model only once
        mobilenet_v2 = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        
        # Extract the features from the pre-trained model
        features = mobilenet_v2.features
        
        # Create part1 by slicing the first three layers of the MobileNet V2 features
        self.part1 = nn.Sequential(*features[:4])
        
        self.gates = nn.ModuleList()

        # Initialize ModuleList for the second part
        self.part2 = nn.ModuleList()
        
        self.upscale_layers = nn.ModuleList()
        
        # The rest of the features and the classifier are shared among slice_num entries
        remaining_features = nn.Sequential(*features[4:])
        classifier = mobilenet_v2.classifier
        
        # Create part2 modules for each entry in slice_num
        for i in slice_num:
            self.part2.append(nn.Sequential(
                remaining_features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                classifier
            ))
            self.upscale_layers.append(nn.Conv2d(in_channels=i, out_channels=24, kernel_size=1, stride=1, padding=0))
             
    def forward(self, x, i):
        base = self.part1(x)
        # Apply each part2 module to the output of part1
        sliced_base = base[:, :slice_num[i], :, :]
        upscaled_base = self.upscale_layers[i](sliced_base)
        out = self.part2[i](upscaled_base)
        return out
    
    
class mobilenet_slice_gate(nn.Module):
    def __init__(self):
        super(mobilenet_slice_gate, self).__init__()
        # Load the pre-trained MobileNet V2 model only once
        mobilenet_v2 = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        features = mobilenet_v2.features
        self.part1 = nn.Sequential(*features[:4])
        self.gates = nn.ModuleList()
        self.part2 = nn.ModuleList()
        self.upscale_layers = nn.ModuleList()
        remaining_features = nn.Sequential(*features[4:])
        classifier = mobilenet_v2.classifier
        
        # Create part2 modules for each entry in slice_num
        for i in slice_num:
            if  len(self.gates) < len(slice_num) - 1:
                self.gates.append(ExitGate(i, 56, 56))
            self.part2.append(nn.Sequential(
                remaining_features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                classifier
            ))
            self.upscale_layers.append(nn.Conv2d(in_channels=i, out_channels=24, kernel_size=1, stride=1, padding=0))
             
    def forward(self, x, i):
        if i < len(slice_num) -1:
            base = self.part1(x)
            # Apply each part2 module to the output of part1
            sliced_base = base[:, :slice_num[i], :, :]
            exitstate = self.gates[i](sliced_base)
            upscaled_base = self.upscale_layers[i](sliced_base)
            out = self.part2[i](upscaled_base)
            return out, exitstate
        else:
            base = self.part1(x)
            # Apply each part2 module to the output of part1
            sliced_base = base[:, :slice_num[i], :, :]
            upscaled_base = self.upscale_layers[i](sliced_base)
            out = self.part2[i](upscaled_base)
            return out, torch.tensor([1.0])
            
    
    
class mobilenet_slice_gate_train(nn.Module):
    def __init__(self):
        super(mobilenet_slice_gate_train, self).__init__()
        # Load the pre-trained MobileNet V2 model only once
        mobilenet_v2 = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        features = mobilenet_v2.features
        self.part1 = nn.Sequential(*features[:4])
        self.gates = nn.ModuleList()
        self.part2 = nn.ModuleList()
        self.upscale_layers = nn.ModuleList()
        remaining_features = nn.Sequential(*features[4:])
        classifier = mobilenet_v2.classifier
        
        # Create part2 modules for each entry in slice_num
        for i in slice_num:
            if  len(self.gates) < len(slice_num) - 1:
                self.gates.append(ExitGate(i, 56, 56))
            self.part2.append(nn.Sequential(
                remaining_features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                classifier
            ))
            self.upscale_layers.append(nn.Conv2d(in_channels=i, out_channels=24, kernel_size=1, stride=1, padding=0))
             
    def forward(self, x, i):
        base = self.part1(x)
        # Apply each part2 module to the output of part1
        sliced_base = base[:, :slice_num[i], :, :]
        exitstate = self.gates[i](sliced_base)
        upscaled_base = self.upscale_layers[i](sliced_base)
        out = self.part2[i](upscaled_base)
        return out, exitstate