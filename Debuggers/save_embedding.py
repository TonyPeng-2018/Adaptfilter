# use the model in Model, and dataloader in Dataloaders
# to train the model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Adaptfilter.Debuggers import mobilenetv2_revised
from Dataloaders import dataloader_cifar10
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
# get the dataset
train, test, labels = dataloader_cifar10.Dataloader_cifar10(train_batch=128, test_batch=100, seed=2024)

# 2. transfer the dataset to fit the model, for the training, client and server model are all on the server
client_model, server_model = mobilenetv2_revised.stupid_model_splitter(weight_path='./Weights/cifar-10/MobileNetV2.pth')

# gating = some_gating_function()

# reducer = some_reducer_funtion()

# generator = some_generator_funtion()

client_model = client_model.cuda()

# infer the model
client_model.eval()

# create a file path
embeddings_path = '../data/cifar-10-embedding-3/'
embedding_tensor = None
labels_tensor = None
# the embedding of the model and store
if not os.path.exists(embeddings_path):
    os.makedirs(embeddings_path)
if not os.path.exists(embeddings_path+ 'embeddings/'):
    os.makedirs(embeddings_path+ 'embeddings/')
if not os.path.exists(embeddings_path+ 'labels/'):
    os.makedirs(embeddings_path+ 'labels/')
with torch.no_grad():
    for ind, data in tqdm(enumerate(train)):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = client_model(inputs)

        outputs_cpu = outputs.cpu()
        # if embedding_tensor is None:
        #     embedding_tensor = outputs_cpu
        #     labels_tensor = labels
        # else:
        #     embedding_tensor = torch.cat((embedding_tensor, outputs_cpu), dim=0)
        #     labels_tensor = torch.cat((labels_tensor, labels), dim=0)
        embedding_tensor = outputs_cpu
        labels_tensor = labels
        
        # store the embedding

        torch.save(embedding_tensor, embeddings_path + 'embeddings/' + str(ind) + '.pth')
        torch.save(labels_tensor, embeddings_path + 'labels/' + str(ind) + '.pth')
