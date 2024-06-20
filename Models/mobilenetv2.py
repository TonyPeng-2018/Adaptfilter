'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''

# changed F.relu to nn.relu
# from https://github.com/kuangliu/pytorch-cifar
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layers(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        # out = out.view(out.size(0), -1) # keep the first, use flatten instead
        out = self.flatten(out)
        out = self.linear(out)
        return out

def flatten_model(initial_model, new_model, root_name = ''):
    # recurrsively flatten the model
    # give a model, return two flatten models
    # something like shortcut is static (not dynamic? can be changed dynamically?)
    # this is after loading weights, so we don't need to worry about the weight keys
    
    for sub_name, sub_module in initial_model.named_children():
        if len(list(sub_module.children())) == 0:
            new_model.add_module(root_name + sub_name, sub_module)
        else:
            # if the previous layer a.b.c, now it is a_b_c
            new_model = flatten_model(sub_module, new_model, root_name + sub_name + '_')
    return new_model

# split the mobilenetV2 to client and server part. 
def model_splitter(num_classes: int = 10, partioning_point:int = 3, weight_path: str = ''):
    # this is not only for cifar, so we can also change the last layer
    # num_classes: the number of classes in the dataset
    # slice_granularity: the granularity of the slice
    # weight_path: the path of the pretrained model

    # initial variables here
    model = None

    # create a mobilenetv2 model, load the weight
    if weight_path == '':
        model = MobileNetV2(num_classes=num_classes)
    else:
        model = MobileNetV2(num_classes=num_classes)
        model.load_state_dict(torch.load(weight_path)['net'])
    
    # let's skip the granularity for now
    # slice the model into a client and server part
    # use childrens or modules(fake layer splitter)
    client_model = nn.Sequential()
    server_model = nn.Sequential()

    # # flatten the model
    # new_model = flatten_model(model, client_model)
    # client_model = new_model[:partioning_point]
    # server_model = new_model[partioning_point:]

    # print(len(client_model))
    # print(len(server_model))
    # for ind, sub_module in enumerate(model.named_modules()):
    #     name, module = sub_module
    #     if ind < partioning_point:
    #         client_model.add_module(name, module)

    # return the client and server model

    # split useing named_children, children doesn't derive the previous forward. 

    for ind, (name, module) in enumerate(model.named_children()):
        if ind < partioning_point:
            client_model.add_module(name, module)
        else:
            server_model.add_module(name, module)

    return client_model, server_model

class MobileNetV2_Client(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2_Client, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class MobileNetV2_Server(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2_Server, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        # out = out.view(out.size(0), -1) # keep the first, use flatten instead
        out = self.flatten(out)
        out = self.linear(out)
        return out

# split the mobilenetV2 to client and server part. 
def stupid_model_splitter(num_classes: int = 10, partioning_point:int = 3, weight_path: str = '', device = 'cuda:0'):
    assert partioning_point ==3, 'The stupid model splitter only works for partioning_point = 3'
    # create client, filter weights and load the server
    client_model = MobileNetV2_Client(num_classes=num_classes)
    if weight_path != '':
        client_weights = torch.load(weight_path, map_location=device)['net']
        client_model_keys = client_model.state_dict().keys()
        client_weights = {k: v for k, v in client_weights.items() if k in client_model_keys}
        client_model.load_state_dict(client_weights)

        # create server, load the weights and filter the client
    server_model = MobileNetV2_Server(num_classes=num_classes) 
    if weight_path != '':
        server_weights = torch.load(weight_path, map_location=device)['net']
        server_model_keys = server_model.state_dict().keys()
        server_weights = {k: v for k, v in server_weights.items() if k in server_model_keys}
        server_model.load_state_dict(server_weights)
        
    return client_model, server_model


# test the performance of the model
if __name__ == '__main__':
    net = MobileNetV2()# test of dimension chagne
    # test model_splitter
    # client_model, server_model = model_splitter(weight_path = './otherwork/pytorchcifar/checkpoint/MobileNetV2.pth')
    # read a image
    # img = torch.randn(10, 3, 32, 32)
    # print(client_model)
    # print(server_model)
    # print(net)
    # out1 = net(img)
    # print(out1.size())
    # for i in range(len(client_model)):
    #     print(i)
    #     print(client_model[i])
    #     if i == 0:
    #         out2 = client_model[i](img)
    #     else:
    #         out2 = client_model[i](out2)
    #     print(out2.size())
    
    # for i in range(len(server_model)):
    #     print(i)
    #     print(server_model[i])
    #     out2 = server_model[i](out2)
    #     print(out2.size())
    # print(out2.size())

    # test the stupid function
    client_model, server_model = stupid_model_splitter(
        weight_path='../otherwork/pytorchcifar/checkpoint/MobileNetV2.pth')
    img = torch.randn(10, 3, 32, 32)
    out1 = net(img)
    print(out1.size())
    out2 = client_model(img)
    print(out2.size())
    out2 = server_model(out2)
    print(out2.size())