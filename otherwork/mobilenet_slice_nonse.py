import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import torchvision


slice_num = [3, 6, 12, 24, 48]
slices = {3: 3, 6: 3, 12: 6, 24: 12, 48: 24}


class mobilenet_slice_train(nn.Module):
    def __init__(self):
        super(mobilenet_slice_train, self).__init__()
        # Load the pre-trained MobileNet V2 model only once
        mobilenet_v2 = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")

        # Extract the features from the pre-trained model
        features = mobilenet_v2.features
        self.alphas = nn.Parameter(torch.full((3,), 0.5))
        # Create part1 by slicing the first three layers of the MobileNet V2 features
        self.part1 = nn.Sequential(*features[:4])

        self.encoders = nn.ModuleList()
 
        self.upscale_layers = nn.ModuleList()

        # The rest of the features and the classifier are shared among slice_num entries
        remaining_layers = nn.Sequential(*features[4:])
        classifier = mobilenet_v2.classifier

        # Initialize ModuleList for the second part
        self.part2 = nn.Sequential(
            remaining_layers,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            classifier,
        )

        # Create part2 modules for each entry in slice_num
        for i in slice_num:
            C_in = 24
            C_out = slices[i]
            self.encoders.append(
                Block1(
                    l1=C_out * 2 + 24,
                    l2=C_out * 2 + 12,
                    l3=C_out + 6,
                    C_in=C_in,
                    C_out=C_out,
                )
            )
            self.upscale_layers.append(UpsampleNetwork(i, 24))

    def forward(self, x, i):
        base = self.part1(x)
        # Apply each part2 module to the output of part1
        sliced_bases = [self.encoders[k](base) for k in range(i + 1)]
        concat_bases = torch.cat(sliced_bases, dim=1)
        upscaled_base = self.upscale_layers[i](concat_bases)
        out = self.part2(upscaled_base)
        return out


class UpsampleNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels * 2 + 6, kernel_size=3, stride=1, padding=1
        )
        self.upsample1 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.conv2 = nn.Conv2d(
            in_channels * 2 + 6,
            in_channels * 2 + 12,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels * 2 + 12, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels * 2 + 6)
        self.bn2 = nn.BatchNorm2d(in_channels * 2 + 12)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.dropout(self.bn1(self.conv1(x))))
        x = self.upsample1(x)
        x = self.relu(self.dropout(self.bn2(self.conv2(x))))
        x = self.relu(self.dropout(self.bn3(self.conv3(x))))
        return x


class Block1(nn.Module):
    def __init__(self, l1, l2, l3, C_in=24, C_out=12):
        super(Block1, self).__init__()
        self.layers = nn.Sequential(
            MobileNetBlock(
                in_channels=C_in, out_channels=l1, size=3, stride=1, padding=1
            ),
            MobileNetBlock(
                in_channels=l1, out_channels=l2, size=3, stride=2, padding=1
            ),
            MobileNetBlock(
                in_channels=l2, out_channels=l3, size=3, stride=1, padding=1
            ),
            MobileNetBlock(
                in_channels=l3, out_channels=C_out, size=3, stride=2, padding=1
            ),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, stride, padding):
        super(MobileNetBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.dropout(self.bn1(self.depthwise(x))))
        x = self.relu(self.dropout(self.bn2(self.pointwise(x))))
        return x
