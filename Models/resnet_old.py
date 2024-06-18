# similar to mobilenetv2, we can implement resnet18, resnet34, resnet50, resnet101, resnet152

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
# import utils

def flatten_model(initial_model, model_slices, index):
    # recurrsively flatten the model
    for sub_module in initial_model.children():
        if len(list(sub_module.children())) == 0:
            model_slices.append(sub_module)
        else:
            flatten_model(sub_module, model_slices, index)
    return model_slices

class Resnet:
    def __init__(self, num_classifier : int = 1000, num_layer: int = 18, partition_point: int = 3, 
                 pretrained: bool = True, slice_granularity: str = 'coarse grained', ) -> None:
        # create a resnet model
        self.num_layer = num_layer
        self.pretrained = pretrained
        self.partition_point = partition_point
        self.slice_granularity = slice_granularity
        self.num_classifier = num_classifier

        if self.num_layer == 18:
            self.model = torchvision.models.resnet18(pretrained=self.pretrained)
        elif self.num_layer == 34:
            self.model = torchvision.models.resnet34(pretrained=self.pretrained)
        elif self.num_layer == 50:
            self.model = torchvision.models.resnet50(pretrained=self.pretrained)
        elif self.num_layer == 101:
            self.model = torchvision.models.resnet101(pretrained=self.pretrained)
        elif self.num_layer == 152:
            self.model = torchvision.models.resnet152(pretrained=self.pretrained)
        # flatten all layers
        # comment because the downsampler is not able to be flatten easily
        # if self.slice_granularity == 'fine grained':
        #     self.flatten_model = utils.flatten_model(self.model, [], 0)
        #     self.client_model = nn.Sequential(*self.flatten_model[:self.partition_point])
        #     self.server_model = nn.Sequential(*self.flatten_model[self.partition_point:])
        #     # change the last layer to the number of classes
        #     if self. num_classifier!= 1000:
        #         self.server_model[-1] = nn.Linear(512, self.num_classifier, bias=True)
        # slice the model into a client and server part, split layer 1 to layer 4
        if self.slice_granularity == 'coarse grained':
            self.flatten_model = nn.Sequential()
            ind = 0
            self.layer1 = self.model.layer1.children()
            self.layer2 = self.model.layer2.children()
            self.layer3 = self.model.layer3.children()
            self.layer4 = self.model.layer4.children()
            self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            # add model head
            self.flatten_model.add_module(str(ind), nn.Sequential(self.model.conv1, 
                                                                  self.model.bn1, 
                                                                  self.model.relu, 
                                                                  self.model.maxpool))
            ind += 1
            for layer in self.layers:
                for sub_layer in layer:
                    self.flatten_model.add_module(str(ind), sub_layer)
                    ind += 1
            # add output layer
            self.flatten_model.add_module(str(ind), nn.Sequential(self.model.avgpool, 
                                                                  nn.Flatten(1),
                                                                  self.model.fc))
            # slice the model
            self.client_model = nn.Sequential(*list(self.flatten_model.children())[:self.partition_point])
            self.server_model = nn.Sequential(*list(self.flatten_model.children())[self.partition_point:])
            # change the last layer to the number of classes
            if self.num_classifier != 1000:
                self.server_model[-1] = nn.Linear(512, self.num_classifier, bias=True)

        else:
            raise ValueError('Invalid slice granularity')
        
        # create the client and server part
    def get_client_model(self):
        return self.client_model
    
    def get_server_model(self):
        return self.server_model

if __name__ == '__main__':
    model = Resnet(num_layer=18, partition_point=3, slice_granularity='coarse grained')
    client_model = model.get_client_model()
    server_model = model.get_server_model()
    # test the model
    x = torch.randn(32, 3, 224, 224)
    y = torch.randn(32, 64, 112, 112)
    client_model.eval()
    server_model.eval()
    with torch.no_grad():
        client_output = client_model(x)
        print(client_output.shape)
        server_output = server_model(y)
        print(server_output.shape)
