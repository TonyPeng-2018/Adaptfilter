# this file is a mobilenet V2 model. 
# we have a partitioning point, layers are sliced into a client and server part

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision

class MobileNetV2(nn.Module):
    # import the mobilenetv2 model from torchvision
    # num_classesm: the number of classes in the dataset
    # pretrained: if the model is pretrained (if we need to retrain the model)
    def __init__(self, num_classesm: int = 1000, pretrained: bool = True, partioning_point:int = 3,
                 slice_granularity: str = 'coarse grained') -> None:
        super(MobileNetV2, self).__init__()
        # create a mobilenetv2 model, this is pretrained
        self.model = torchvision.models.mobilenet_v2(pretrained=pretrained)

        # slice the model into a client and server part
        self.partioning_point = partioning_point
        # fine-grain slices
        if slice_granularity == 'fine grained':
            model_slices = [x for x in self.model.modules() if not isinstance(x, nn.Sequential)]
            model_slices.pop(0) # remove the first summary layer
            if num_classesm != 1000:
                model_slices.pop(-1) # remove the last classifier layer
                model_slices.pop(-1) # remove the last dropout layer
        elif slice_granularity == 'coarse grained':
            model_slices = [x for x in self.model.features if not isinstance(x, nn.Sequential)]
            # this doesn't include the classifier layer, splited by submodules
        else:
            raise ValueError('Invalid slice granularity')

        # create the client and server part
        self.client_model = nn.Sequential(*model_slices[:self.partioning_point])
        self.server_model = nn.Sequential(*model_slices[self.partioning_point:])
        
        # add the dropout and classifier layer to the server model
        if num_classesm != 1000:
            self.server_model.add_module('dropout', nn.Dropout(p=0.2, inplace=False))
            self.server_model.add_module('classifier', nn.Linear(1280, num_classesm))
    def get_client_model(self):
        return self.client_model
    
    def get_server_model(self):
        return self.server_model

    