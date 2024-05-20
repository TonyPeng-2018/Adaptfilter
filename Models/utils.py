import torch
from torch import nn
import torch.nn.functional as F

def flatten_model(initial_model, model_slices, index):
    # recurrsively flatten the model
    for sub_module in initial_model.children():
        if len(list(sub_module.children())) == 0:
            model_slices.append(sub_module)
        else:
            flatten_model(sub_module, model_slices, index)
    return model_slices