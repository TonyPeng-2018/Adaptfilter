import sys
import json
import argparse
import time
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import scipy.stats
import torch.utils 
import os
import numpy as np
import time
from tqdm import tqdm
# import the server model

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


class GatedRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GatedRegression, self).__init__()
        # think about it 3*32*32 -> 1
        # think about the classification of the mobile net
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = input_size
        self.conv1 = nn.Conv2d(input_size, 2*self.hidden_size, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2*self.hidden_size)
        self.conv2 = nn.Conv2d(2*self.hidden_size, 4*self.hidden_size, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*self.hidden_size)
        self.conv3 = nn.Conv2d(4*self.hidden_size, 8*self.hidden_size, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(8*self.hidden_size)
        self.linear = nn.Linear(8*self.hidden_size, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        # need to change 32 -> 4
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out
    
# load the dataset
class gated_dataset(torch.utils.data.Dataset):
    # load the dataset
    def __init__(self, embeddings_folder):
        self.embeddings_folder = embeddings_folder
        self.embeddings_files = sorted(os.listdir(embeddings_folder+'/embeddings/'))
        self.labels_files = sorted(os.listdir(embeddings_folder+'/labels/'))

    def __len__(self):
        return self.embeddings_files.__len__()

    def __getitem__(self, idx):
        self.embeddings = torch.load(self.embeddings_folder+'/embeddings/' + self.embeddings_files[idx])
        self.labels = torch.load(self.embeddings_folder+'/labels/' + self.labels_files[idx])
        return self.embeddings, self.labels
    
# load the dataset
embeddings_folder = '../data/cifar-10-embedding-3/'
dataset = gated_dataset(embeddings_folder)
# print(gated_dataset.__len__(dataset)) # 391 batches                                             

# load the data loader, train no test
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# load the model
Gated = GatedRegression(input_size=3, num_classes=10)
Gated = Gated.cuda()

# load the optimizer
optimizer = optim.Adam(Gated.parameters(), lr=0.001)

# load the loss function
criterion = nn.CrossEntropyLoss()

# train the model
for epoch in range(100):
    for i, data in tqdm(enumerate(train_loader)):
        embeddings, labels = data
        # embeddings = Variable(embeddings) # what are they?
        # labels = Variable(labels) # what are they?
        embeddings = embeddings.squeeze(0)
        labels = labels.squeeze(0)
        time0 = time.time()
        selected_embeddings = ranker_entropy(embeddings, 0.1)
        optimizer.zero_grad()
        selected_embeddings = selected_embeddings.cuda()
        labels = labels.cuda()
        outputs = Gated(selected_embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, 100, loss.item()))