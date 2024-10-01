# this layer is created for the last classifier 
import torch
import torch.nn as nn

class last_layer_classifier(nn.Module):
    def __init__(
        self,
        in_channel = 1000,
        out_channel = 20
    ) -> None:
        """
        We create a linear layer for transfering the output of the last layer to the output of the model

        """
        super().__init__()

        # create a linear layer
        # bias is needed? 
        self.linear = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        x = self.linear(x)
        return x