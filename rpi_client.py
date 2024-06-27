# this file is for device on the client side

# load the dataset

# This file is for trainning
# Run this on the server, or as we called offline. 

import argparse
from Dataloaders import dataloader_cifar10, dataloader_cifar100
import datetime
from Models import mobilenetv2, mobilenetv3, resnet
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from Utils import utils

def main(args):
    p_start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    logger = utils.APLogger('./Logs/' + args.dataset + '/client_' + p_start_time + '.log\n')

    # write the logger with all args parameters
    logger.write('model: %s, dataset: %s\n' % (args.model, args.dataset))
    logger.write('batch: %d, compressor: %s, ranker: %s\n' % (args.batch, args.compressor, args.ranker))
    logger.write('weight: %s\n' % (args.weight))

    # 1. load the dataset
    weight_path = './Weights/' + args.dataset + '/client/' + args.model + '.pth'
    weight_root = './Weights/' + args.dataset + '/'
    if args.dataset == 'cifar-10':
        num_classes = 10
    elif args.dataset == 'cifar-100':
        num_classes = 100
    elif args.dataset == 'imagenet-mini':
        num_classes = 100
    elif args.dataset == 'imagenet-tiny':
        num_classes = 200
    elif args.dataset == 'imagenet':
        num_classes = 1000
    
    if args.model == 'mobilenetV2':
        c_model = mobilenetv2.mobilenetv2_splitter_client(num_classes = num_classes, weight_root=weight_root, device='cpu')

    elif args.model == 'mobilenetV3':
        c_model = mobilenetv3.mobilenetv3_splitter_client(num_classes = num_classes, weight_root=weight_root, device='cpu')

    elif args.model == 'resnet':
        c_model = resnet.resnet_splitter_client(num_classes = num_classes, weight_root=weight_root, device='cpu', layers = args.resnetsize)
    
    for epoch in tqdm(range(50)):
        model.train()
        for i, data in tqdm(enumerate(train)):
            inputs, labels = data
            inputs, labels = inputs.cuda(args.cuda), labels.cuda(args.cuda)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, 
                             )
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logger.write('epoch: %d, batch: %d, loss: %.3f\n' % (epoch, i, loss.item()))
        # 5. save the model
        logger.write('Train epoch: %d, loss: %.3f\n' % (epoch, loss.item()))
        
        # use the val test to test the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(val):
                inputs, labels = data
                inputs, labels = inputs.cuda(args.cuda), labels.cuda(args.cuda)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        logger.write('Val epoch: %d, accuracy: %.3f\n' % (epoch, correct/total))
        if correct/total > min_loss:
            min_loss = correct/total
            torch.save(model.state_dict(), weightfolder+start_time+'/'+args.model+'_'+str(epoch)+'_%.3f'%(correct/total)+'.pth')


if __name__ == '__main__':
    print('enter')
    parser = argparse.ArgumentParser()
    # we need the name of model, the name of dataset
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--compressor', type=str, default='home', help='compressor name')
    parser.add_argument('--dataset', type=str, default='cifar-10', help='dataset name')
    parser.add_argument('--mobilev3size', type=str, default='small', help='the size of the mobilev3')
    parser.add_argument('--model', type=str, default='mobilenetnetv2', help='model name')
    parser.add_argument('--ranker', type=str, default='zeros', help='ranker name')
    parser.add_argument('--resnetsize', type=int, default=18, help='resnet layers')
    parser.add_argument('--weight', type=str, default='./Weight/cifar-10/', help='weight path')
    args = parser.parse_args()
    print(args)
    main(args)
    
