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
from Models import mobilenetv2
from tqdm import tqdm
import numpy as np

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
    return selected_embeddings, selected_indices

# import the server model
run_device = 'server'
# This is the version 1 gated regression model, we need a simpler version
# class GatedRegression(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(GatedRegression, self).__init__()
#         # think about it 3*32*32 -> 1
#         # think about the classification of the mobile net
#         self.input_size = input_size
#         self.num_classes = num_classes
#         self.hidden_size = input_size
#         self.conv1 = nn.Conv2d(input_size, 2*self.hidden_size, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(2*self.hidden_size)
#         self.conv2 = nn.Conv2d(2*self.hidden_size, 4*self.hidden_size, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(4*self.hidden_size)
#         self.conv3 = nn.Conv2d(4*self.hidden_size, 8*self.hidden_size, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(8*self.hidden_size)
#         self.linear = nn.Linear(8*self.hidden_size, num_classes)
#         self.relu = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(4)
#         self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

#     def forward(self, x):
#         # need to change 32 -> 4
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.conv3(out)
#         out = self.bn3(out)
#         out = self.relu(out)
#         out = self.avgpool(out)
#         out = self.flatten(out)
#         out = self.linear(out)
#         return out

class GatedRegression(nn.Module):
    def __init__(self, input_size, weight, height, output_size=10):
        super(GatedRegression, self).__init__()
        # think about it 3*32*32 -> 1
        # think about the classification of the mobile net
        # the input size is b, c*p, h, w, the output size is b 
        # how to make sure more features help the server model?

        self.input_size = input_size * weight * height # 8, 32, 32 - 24, 32, 32
        self.output_size = output_size
        # 1280 = 5*16*16
        self.structure = [self.input_size, self.input_size//16, self.output_size]
        self.linear1 = nn.Linear(self.structure[0], self.structure[1])
        self.linear2 = nn.Linear(self.structure[1], self.structure[2])
        # self.linear3 = nn.Linear(self.structure[2], self.structure[3])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        # need to change it to 0-1
        # flatten the input first
        out = self.flatten(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out

# load the dataset
class gated_dataset(torch.utils.data.Dataset):
    # load the dataset
    def __init__(self, embeddings_folder, labels_folder):
        self.embeddings_folder = embeddings_folder
        self.embeddings_files = sorted(os.listdir(embeddings_folder+'embeddings/'))
        self.labels_folder = labels_folder
        self.labels_files = sorted(os.listdir(labels_folder+'embeddings/'))

    def __len__(self):
        return self.embeddings_files.__len__()

    def __getitem__(self, idx):
        self.embeddings = torch.load(self.embeddings_folder+'embeddings/' + self.embeddings_files[idx])
        self.labels = torch.load(self.labels_folder+'embeddings/' + self.labels_files[idx])
        return self.embeddings, self.labels
    
# load the dataset
if run_device == 'tintin':
    embeddings_folder = '/data/anp407/cifar-10-embedding-3/'
    labels_folder = '/data/anp407/cifar-10-embedding-out/'
if run_device == 'server':
    embeddings_folder = '../data/cifar-10-embedding-3/'
    labels_folder = '../data/cifar-10-embedding-out/'
dataset = gated_dataset(embeddings_folder, labels_folder)
# print(gated_dataset.__len__(dataset)) # 391 batches                                             

# load the data loader, train no test
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# load the model
Gateds = []
input_sizes = [8, 16, 24]
# added Gated to Gateds
for i, input_size in enumerate(input_sizes):
    Gated = GatedRegression(input_size=input_size, weight=32, height=32, output_size=10)
    Gated.load_state_dict(torch.load('./Weights/cifar-10/GatedRegression_%d.pth' % i))
    Gateds.append(Gated)

# print(torch.cuda.is_available())
# model settings
for Gated in Gateds:
    Gated = Gated.cuda()
    Gated.train()

# get the server model
client_model, server_model = mobilenetv2.stupid_model_splitter(weight_path='./Weights/cifar-10/MobileNetV2.pth')
server_model = server_model.cuda()
server_model.eval()
# load the optimizer
optimizer = optim.Adam(Gated.parameters(), lr=0.001)

# load the loss function
criterion = nn.MSELoss()
epochs = 50

# train the model
for epoch in tqdm(range(epochs)):
    for i, data in enumerate(train_loader):
        embs, labels = data
        # for the training, embeddings doesn't need to be on the cuda
        # embs, labels = embs.cuda(), labels.cuda()
        labels = labels.cuda()
        # embeddings = Variable(embeddings) # what are they?
        # labels = Variable(labels) # what are they?
        embs = embs.squeeze(0)
        labels = labels.squeeze(0)
        # for the training, we make 4 gated regression models
        f_rate = [0.25, 0.5, 0.75] # filter rates
        # s_embs, s_inds = [], []
        preds = []
        # get the selected embeddings
        for j, rate in enumerate(f_rate):
            s_emb, s_ind = ranker_entropy(embs, rate)
            # make the embeddings to fit the server model
            # n_emb = torch.zeros(s_emb.size(0), 32, 32, 32)
            # n_emb[:, s_ind] = s_emb
            # n_emb = n_emb.cuda()
            # multiple gated regression models, we use the selected embeddings
            s_emb = s_emb.cuda()
            preds.append(Gateds[j](s_emb))

        # get the output from the server model
        # with torch.no_grad():
        #     embs = embs.cuda()
        #     outputs = server_model(embs)
        # get the save result 


        # loss backpropagation
        losses = [0] * len(preds)
        for j, pred in enumerate(preds):
            loss = criterion(pred, labels)
            losses[j] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
    for j, loss in enumerate(losses):
        print('Epoch [%d/%d], Loss: %.4f, Gated: %d' % (epoch+1, epochs, loss, j))
    # save the pth
    for j, Gated in enumerate(Gateds):
        torch.save(Gated.state_dict(), './Weights/cifar-10/GatedRegression_%d.pth' % (j+50))