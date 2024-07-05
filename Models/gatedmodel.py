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
    def __init__(self, i_size, width, height, o_size=1, n_ch=32, rate=0.5):
        super().__init__()
        # think about it 3*32*32 -> 1
        # think about the classification of the mobile net
        # the input size is b, c*p, h, w, the output size is b 
        # how to make sure more features help the server model?
        self.label_embedding = nn.Embedding(n_ch, n_ch)
        self.rate = rate
        self.conv0 = nn.Conv2d(i_size, n_ch, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(n_ch)
        self.relu = nn.ReLU()
        # linear size = 2*n_ch * (width//4) * (height//4) + emb
        self.linear_1 = n_ch*(width//2) * (height//2) + n_ch*int(n_ch*rate)
        self.linear = nn.Linear(self.linear_1, o_size, bias=False)

    def forward(self, x, chs):
        out = self.relu(self.bn(self.conv0(x)))
        out = torch.flatten(out, 1)
        emb = self.label_embedding(chs)
        emb = torch.flatten(emb, 1)
        out = torch.cat([out, emb], dim=-1)
        out = self.linear(out)
        return out
    
class GateMLP_POS(nn.Module):
    def __init__(self, i_size, width, height, o_size=1, n_ch=32, rate=0.5):
        super().__init__()
        

        self.input_size = i_size * width * height # 8, 32, 32 - 24, 32, 32
        self.output_size = o_size
        self.label_embedding = nn.Embedding(n_ch, n_ch)
        self.linear1_in = self.input_size + n_ch*int(n_ch*rate)
        
        # 1280 = 5*16*16
        self.structure = [self.linear1_in, self.linear1_in//32, self.output_size]
        self.linear1 = nn.Linear(self.structure[0], self.structure[1])
        self.linear2 = nn.Linear(self.structure[1], self.structure[2], bias=False)
        # self.linear3 = nn.Linear(self.structure[2], self.structure[3])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        
        self.rate = rate

    def forward(self, x, chs):
        # need to change it to 0-1
        # flatten the input first
        out = self.flatten(x)
        emb = self.label_embedding(chs)
        emb = self.flatten(emb)
        out = torch.cat([out, emb], dim=-1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out
    
class GateMLP_POS(nn.Module):
    def __init__(self, i_size, width, height, o_size=1, n_ch=32, rate=0.5):
        super().__init__()
        

        self.input_size = i_size * width * height # 8, 32, 32 - 24, 32, 32
        self.output_size = o_size
        self.label_embedding = nn.Embedding(n_ch, n_ch)
        self.linear1_in = self.input_size + n_ch*int(n_ch*rate)
        
        # 1280 = 5*16*16
        self.structure = [self.linear1_in, self.linear1_in//32, self.output_size]
        self.linear1 = nn.Linear(self.structure[0], self.structure[1])
        self.linear2 = nn.Linear(self.structure[1], self.structure[2], bias=False)
        # self.linear3 = nn.Linear(self.structure[2], self.structure[3])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        
        self.rate = rate

    def forward(self, x, chs):
        # need to change it to 0-1
        # flatten the input first
        out = self.flatten(x)
        emb = self.label_embedding(chs)
        emb = self.flatten(emb)
        out = torch.cat([out, emb], dim=-1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out