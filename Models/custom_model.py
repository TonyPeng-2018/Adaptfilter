# this is define a self_defined model

import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_modules) -> None:
        super().__init__()
        # use the imported nn.sequence to create the client and server model
        # the input_modules is the list of models
        self.model = input_modules

    def forward(self, x):
        for module in self.model:
            x = module(x)