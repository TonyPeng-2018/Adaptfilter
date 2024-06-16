import numpy as np
import scipy.stats 
import os 
import torch.nn as nn
import torch

class rankerEntropy():
    def __init__(self):
        pass # we dont need to do anything here

    def calculate_entropy(self, embs, percentage):
        # embs are torch tensors
        # percentage shows the number of embeddings to select
        embs_entropy = []
        for i in range(embs.size(len(embs.size())-3)):
            embs_entropy.append(scipy.stats.entropy(embs[:, i].reshape(-1))) # b, c
        # get the top percentage of the channels
        embs_entropy = np.array(embs_entropy)
        indices = np.argsort(embs_entropy)
        num_selected = int(embs.size(1) * percentage)
        selected_indices = indices[:num_selected]
        return selected_indices
        # out b, c*p, h, w or c*p, h, w

    def ranker_entropy(self, embs, percentage):
        # calculate the entropy of the embeddings
        selected_indices = self.calculate_entropy(embs, percentage)
        # select indice from test embeddings
        selected_embeddings = embs[:, selected_indices] # b, c*p, h, w
        return selected_embeddings, selected_indices

class rankerCNN1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(rankerCNN1, self).__init__()
        # make a simple CNN here (we don't need to activate it)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, 1, 1) # what kernel size should we use?
    def forward(self, x):
        return self.conv1(x)
    
class rankerCNN3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(rankerCNN3, self).__init__()
        # make a simple CNN here (we don't need to activate it)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
    def forward(self, x):
        return self.conv1(x)
    
model_list = {'rankerCNN1': rankerCNN1, 'rankerCNN3': rankerCNN3, 'rankerEntropy': rankerEntropy}

# test 
if __name__ == '__main__':
    import torch
    embs = torch.randn(2, 3, 32, 32)
    selected_embs, selected_indices = rankerEntropy().ranker_entropy(embs, 0.25)
    print(selected_embs.size(), selected_indices)
