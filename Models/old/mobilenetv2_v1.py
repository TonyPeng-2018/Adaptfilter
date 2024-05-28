# this file is a mobilenet V2 model. 
# we have a partitioning point, layers are sliced into a client and server part

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

class MobileNetV2(nn.Module):
    # import the mobilenetv2 model from torchvision
    # num_classesm: the number of classes in the dataset
    # pretrained: if the model is pretrained (if we need to retrain the model)
    def __init__(self, num_classes: int = 10, partioning_point:int = 3, pretrained: bool = True,
                 slice_granularity: str = 'coarse grained', weight_path: str = '') -> None:
        super(MobileNetV2, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.partioning_point = partioning_point
        self.slice_granularity = slice_granularity
        self.weight_path = weight_path

        # create a mobilenetv2 model, this is pretrained
        if weight_path == '':
            self.model = torchvision.models.mobilenet_v2(weights=self.pretrained)
        else:
            self.model = torchvision.models.mobilenet_v2()
            self.model = nn.DataParallel(self.model)
            print('loading model from {}'.format(weight_path))
            print(torch.load(weight_path)['net'])
            self.model.load_state_dict(torch.load(weight_path)['net'])
        # change it to nn.modules type
        self.feature = nn.Sequential(*list(self.model.features.children())) # not every layer
        self.connection = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Flatten(1))
        self.classifier = nn.Sequential(*list(self.model.classifier.children()))

        # slice the model into a client and server part
        # fine-grain slices, split every layer
        self.model = nn.Sequential(self.feature, self.connection, self.classifier)
        if self.slice_granularity == 'fine grained':
            # flatten the whole model
            self.flatten_model = flatten_model(self.model, [], 0)
            self.client_model = nn.Sequential(*self.flatten_model[:self.partioning_point])
            self.server_model = nn.Sequential(*self.flatten_model[self.partioning_point:])
            # change the last layer to the number of classes
            if self.num_classes != 1000:
                self.server_model[-1] = nn.Linear(1280, self.num_classes, bias=True)

        elif self.slice_granularity == 'coarse grained':
            # flatten the model into layers
            self.flatten_model = nn.Sequential()
            ind = 0
            for feature in self.feature.children():
                self.flatten_model.add_module(str(ind), feature)
                ind += 1
            # add connection and classifier
            self.flatten_model.add_module(str(ind), self.connection)
            self.flatten_model.add_module(str(ind+1), self.classifier)
            # slice the model
            self.client_model = nn.Sequential(*list(self.flatten_model.children())[:self.partioning_point])
            self.server_model = nn.Sequential(*list(self.flatten_model.children())[self.partioning_point:])
            # change the last layer to the number of classes
            if self.num_classes != 1000:
                self.server_model[-1] = nn.Linear(1280, self.num_classes, bias=True)
        else:
            raise ValueError('Invalid slice granularity')
        # create the client and server part
        # https://blog.csdn.net/leviopku/article/details/82150990
    def get_client_model(self):
        return self.client_model
    
    def get_server_model(self):
        return self.server_model

# test the model
if __name__ == '__main__':
    model = MobileNetV2(pretrained=True, partioning_point=3, slice_granularity='fine grained')
    client_model = model.get_client_model()
    server_model = model.get_server_model()
    # test the model
    x = torch.randn(32, 3, 224, 224)
    client_model.eval()
    server_model.eval()
    with torch.no_grad():
        client_output = client_model(x)
        server_output = server_model(client_output)


