# This file is for trainning
# Run this on the server, or as we called offline. 

import argparse
import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from Adaptfilter.Debuggers import mobilenetv2_revised
from Dataloaders import dataloader_cifar10, dataloader_imagenet
from Models import mobilenetv3, resnet, mobilenetv2_original
import datetime
from Utils import utils
from tqdm import tqdm

def main(args):
    start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    logger = utils.APLogger('./Logs/' + args.dataset + '/model_train_' + start_time + '.log\n')
    logger.write('model: %s, dataset: %s\n' % (args.model, args.dataset))
    weightfolder = './Weights/' + args.dataset + '/model/'
    if not os.path.exists(weightfolder+start_time+'/'):
        os.makedirs(weightfolder+start_time+'/')

    if args.dataset == 'cifar10':
        # return train, test, val, labels, these are all dataloaders
        train, _, val, _ = dataloader_cifar10.Dataloader_cifar10(train_batch=args.batch, test_batch=100, random_seed=2024)
        num_classes = 10
    elif args.dataset == 'imagenet':
        imageset = dataloader_imagenet.Dataset_imagenet(args.device)
        train_set, _, val_set = imageset.return_sampler()
        tr_dict, _, v_dict = imageset.return_dict()
        class_index = imageset.return_class_index()
        train = dataloader_imagenet.Dataloader_imagenet(train_set, tr_dict, transform=True)
        val = dataloader_imagenet.Dataloader_imagenet(val_set, v_dict, transform=True)
        train = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=4)
        val = torch.utils.data.DataLoader(val, batch_size=args.batch, shuffle=True, num_workers=4)
        num_classes = 1000
    # 2. transfer the dataset to fit the model, for the training, client and server model are all on the server
    if args.model == 'mobilenetV2':
        if args.dataset == 'cifar10':
            model = mobilenetv2_revised.MobileNetV2(num_classes = num_classes)
        elif args.dataset == 'imagenet':
            model = mobilenetv2_original.MobileNetV2(num_classes = num_classes)
    elif args.model == 'mobilenetV3':
        model = mobilenetv3.mobilenetV3(num_classes = num_classes, model_size=args.mobilev3size)
    elif args.model == 'resnet':
        model = resnet.resnet50(num_classes = num_classes)
    
    if args.resume:
        weightpath  = weightfolder + args.weighttime + '/' + args.weightname + '.pth'
        model.load_state_dict(torch.load(weightpath), map_location='cuda:'+str(args.cuda))
    model = model.cuda(args.cuda)

    # 3. define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    min_loss = -1
    # 4. start the training
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
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
    # parser.add_argument('--iot_model', type=str, default='mobilenetV2', help='name of the model on the iot')
    parser.add_argument('--reducer', type=str, default='entrophy', help='name of the reducer')
    parser.add_argument('--client', type=str, default='LTE', help='name of the network condition on the client side')
    parser.add_argument('--server', type=str, default='LTE', help='name of the network condition on the server side')
    parser.add_argument('--generator', type=str, default='None', help='name of the generator')
    # parser.add_argument('--server_model', type=str, default='mobilenetV2', help='name of the model on the server, should be the same as it on the iot')
    parser.add_argument('--device', type=str, default='home', help='run on which device, home, tintin, rpi, pico, jetson?')
    parser.add_argument('--model', type=str, default='mobilenetV2', help='name of the model')
    parser.add_argument('--cuda', type=int, default=0, help='gpu id')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--resume', type=bool, default=False, help='resume from the checkpoint')
    parser.add_argument('--weighttime', type=str, default='', help='the time of the weight')
    parser.add_argument('--weightname', type=str, default='', help='the name of the weight')
    parser.add_argument('--mobilev3size', type=str, default='small', help='the size of the mobilev3')
    args = parser.parse_args()
    print(args)
    main(args)
    
