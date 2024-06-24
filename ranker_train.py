import sys
import json
import argparse
import time
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils 
import os
from Adaptfilter.Debuggers import mobilenetv2_revised
from Models import ranker
from tqdm import tqdm
import numpy as np
import Utils.utils as utils
from Config.external_path import pre_path
import datetime
from Dataloaders.dataloader_cifar10 import Dataloader_cifar10


run_device = 'home'
start_time = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
run_path = pre_path[run_device]

# load the dataset
# load the training data of the 
logger = utils.APLogger('./Logs/cifar-10/ranker_train' + start_time + '.log')
# print(gated_dataset.__len__(dataset)) # 391 batches                                             
train, test, classes = Dataloader_cifar10(train_batch=128, test_batch=100, seed=2024)

ranker_name = 'rankerCNN1' # GatedRegression or GateCNN or GateCNN2
resume_train = False
# load the model
rankers = []
outputsize = [8, 16, 24]
# added gated to gateds
for i, input_size in enumerate(outputsize):
    rank = ranker.model_list[ranker_name](in_ch=32, out_ch=8)
# model settings
for rank in rankers:
    rank = rankers.cuda()
    rank.train()
# get the server model
client_model, server_model = mobilenetv2_revised.stupid_model_splitter(weight_path='./Weights/cifar-10/model/MobileNetV2.pth')
client_model = client_model.cuda()
server_model = server_model.cuda()
client_model.eval()
server_model.eval()

# load the optimizer
optimizers = []
for rank in rankers:
    optimizer = optim.Adam(rank.parameters(), lr=0.001)
    optimizers.append(optimizer)

# load the loss function
criterion = nn.MSELoss()
epochs = 50

# train the model
for epoch in tqdm(range(epochs)):
    for i, data in enumerate(train):
        embs, labels = data
        embs, labels = embs.cuda(), labels.cuda() # b, c, h, w; b

        embs = client_model(embs) # b, e, h, w

        f_rate = [0.25, 0.5, 0.75] # filter rates
        n_embs = []
        # get the selected embeddings
        for j, rate in enumerate(f_rate):
            n_emb = rankers[j](embs) # b,e,h,w -> b,e',h,w

        # fill in zeros
            n_emb = utils.fill_zeros(n_emb, embs.size(), selected_indices)
            n_embs.append(n_emb)

        losses = [0] * len(preds)
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