# this is a simple model for the regression task
import torch.nn as nn
import torch

# the first model is a simple regression model
class GatedRegression(nn.Module):
    def __init__(self, input_size, width, height, output_size=10):
        super(GatedRegression, self).__init__()
        # think about it 3*32*32 -> 1
        # think about the classification of the mobile net
        # the input size is b, c*p, h, w, the output size is b 
        # how to make sure more features help the server model?

        self.input_size = input_size * width * height # 8, 32, 32 - 24, 32, 32
        self.output_size = output_size
        # 1280 = 5*16*16
        self.structure = [self.input_size, self.input_size//32, self.output_size]
        self.linear1 = nn.Linear(self.structure[0], self.structure[1])
        self.linear2 = nn.Linear(self.structure[1], self.structure[2], bias=False)
        # self.linear3 = nn.Linear(self.structure[2], self.structure[3])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        # need to change it to 0-1
        # flatten the input first
        out = self.flatten(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out

# the second model is a simple regression model with 1 CNN layer
class GateCNN(nn.Module):
    def __init__(self, input_size, width, height, output_size=10):
        super().__init__()
        self.conv0 = nn.Conv2d(input_size, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.AvgPool2d((int(height), int(width)))
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(64, output_size, bias=False)

    def forward(self, x):
        out = self.relu(self.bn(self.conv0(x)))
        out = self.relu(self.bn0(self.pool(out)))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

# the second model is a simple regression model with 2 CNN layer
class GateCNN2(nn.Module):
    def __init__(self, input_size, width, height, output_size=10):
        super().__init__()
        self.conv0 = nn.Conv2d(input_size, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.pool = nn.AvgPool2d((int(height), int(width)))
        self.bn0 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(64, output_size, bias=False)

    def forward(self, x):
        out = self.relu(self.bn(self.conv0(x)))
        out = self.relu(self.bn0(self.pool(out)))
        out = self.relu(self.bn1(self.conv1(out)))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out
    
model_list = {'GateMLP': GatedRegression, 'GateCNN': GateCNN, 'GateCNN2': GateCNN2}

class GateCNN_POS(nn.Module):
    def __init__(self, i_size, width, height, o_size=1, n_ch=32):
        super(GatedRegression, self).__init__()
        # think about it 3*32*32 -> 1
        # think about the classification of the mobile net
        # the input size is b, c*p, h, w, the output size is b 
        # how to make sure more features help the server model?
        self.label_embedding = nn.Embedding(n_ch, n_ch)
        self.conv0 = nn.Conv2d(i_size, n_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(n_ch)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(n_ch, 2*n_ch, kernel_size=3, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        # linear size = 2*n_ch * (width//4) * (height//4) + emb
        self.linear_1 = 2*n_ch * (width//4) * (height//4) + n_ch
        self.linear = nn.Linear(self.linear_1, o_size, bias=False)

    def forward(self, x, chs):
        out = self.relu(self.bn(self.conv0(x)))
        out = self.relu(self.bn1(self.conv1(out)))
        out = torch.flatten()
        emb = self.label_embedding(chs)
        out = torch.cat([out, emb], dim=-1)
        out = self.linear(out)
        return out
    
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