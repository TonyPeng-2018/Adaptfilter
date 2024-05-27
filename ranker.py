# first, load the embeddings
# stored in torch tensors

import torch
import numpy as np
import os
import sys
import json
import argparse
import time
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import scipy.stats 

embeddings_folder = '../data/cifar-10-embedding-3/embeddings/' 
embeddings_files = sorted(os.listdir(embeddings_folder))
test_embeddings = torch.load(embeddings_folder + embeddings_files[0]) # load the first file
print(test_embeddings.size()) # 128, 32, 32, 32

# calculate the entropy of the embeddings, select 1%, 5%, 10%, 20%, 50% of the embeddings
def calculate_entropy(embs, percentage):
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

def ranker_entropy(embs, percentage):
    # calculate the entropy of the embeddings
    selected_indices = calculate_entropy(embs, percentage)
    # select indice from test embeddings
    selected_embeddings = embs[:, selected_indices] # b, c*p, h, w
    return selected_embeddings

# get the selected embeddings
selected_embeddings = ranker_entropy(test_embeddings, 0.1)
print(selected_embeddings.size())
