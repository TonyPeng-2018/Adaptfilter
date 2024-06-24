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
from Adaptfilter.Debuggers import mobilenetv2_revised
from Models import gatedmodel
from tqdm import tqdm
import numpy as np
import Utils.utils as utils
from Dataloaders.dataloader_gate import dataloader_gate
from Config.external_path import pre_path
import datetime


run_device = 'home'
start_time = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
run_path = pre_path[run_device]

# load the dataset
emb_folder = run_path['cifar10'] + run_path['client']
labels_folder = run_path['cifar10'] + run_path['home']
dataset = dataloader_gate(emb_folder, labels_folder)
logger = utils.APLogger('./Logs/cifar-10/gate_train' + start_time + '.log')
# print(gated_dataset.__len__(dataset)) # 391 batches                                             

# load the data loader, train no test
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

gate_name = 'GateMLP' # GatedRegression or GateCNN or GateCNN2
resume_train = False
# load the model
gates = []
input_sizes = [8, 16, 24]
# added gated to gateds
for i, input_size in enumerate(input_sizes):
    gate = gatedmodel.model_list[gate_name](input_size=input_size, width=32, height=32, output_size=10)
    if resume_train:
        gate.load_state_dict(torch.load('./Weights/cifar-10/'+gate_name+'_%d.pth' % i))
    gates.append(gate)

# print(torch.cuda.is_available())
# model settings
for gate in gates:
    gate = gate.cuda()
    gate.train()

# get the server model
client_model, server_model = mobilenetv2_revised.stupid_model_splitter(weight_path='./Weights/cifar-10/model/MobileNetV2.pth')
server_model = server_model.cuda()
server_model.eval()

# load the optimizer
optimizers = []
for gate in gates:
    optimizer = optim.Adam(gate.parameters(), lr=0.001)
    optimizers.append(optimizer)

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
            s_emb, s_ind = utils.ranker_entropy(embs, rate)
            # make the embeddings to fit the server model
            # n_emb = torch.zeros(s_emb.size(0), 32, 32, 32)
            # n_emb[:, s_ind] = s_emb
            # n_emb = n_emb.cuda()
            # multiple gated regression models, we use the selected embeddings
            s_emb = s_emb.cuda()
            preds.append(gates[j](s_emb))

        # get the output from the server model
        # with torch.no_grad():
        #     embs = embs.cuda()
        #     outputs = server_model(embs)
        # get the save result 


        # loss backpropagation
        losses = [0] * len(preds)

        # get the sigmoid value
        labels = torch.sigmoid(labels)
        for j, pred in enumerate(preds):
            loss = criterion(pred, labels)
            losses[j] = loss.item()
            optimizers[j].zero_grad()
            loss.backward()
            optimizers[j].step()

    # print the loss
        for j, loss in enumerate(losses):
            msg = 'Epoch [%d/%d], Batch: %d Loss: %.4f, gated: %d\n' % (epoch+1, epochs, i, loss, j)
            logger.write(msg)
    for j, loss in enumerate(losses):
        msg = 'Epoch [%d/%d], Loss: %.4f, gated: %d\n' % (epoch+1, epochs, loss, j)
        print(msg)
        logger.write(msg)
    # save the pth
    for j, gate in enumerate(gates):
        torch.save(gate.state_dict(), './Weights/cifar-10/gate/'+gate_name+'_%d'% (j) + start_time + '.pth' )