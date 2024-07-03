# this file is for device on the client side

# load the dataset

# This file is for trainning
# Run this on the server, or as we called offline. 

import argparse
import base64
import cv2
import datetime
from Dataloaders import dataloader_cifar10, dataloader_cifar100, dataloader_imagenet
from Models import gatedmodel,mobilenetv2, mobilenetv3, resnet
import numpy as np
import os
import PIL
import sys
import time
import torch
from tqdm import tqdm
from Utils import utils, encoder

def main(args):
    p_start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    logger = utils.APLogger('./Logs/' + args.dataset + '/train_gate_' + p_start_time + '.log\n')

    # write the logger with all args parameters
    logger.write('model: %s, dataset: %s\n' % (args.model, args.dataset))
    logger.write('batch: %d, compressor: %s, ranker: %s\n' % (args.batch, args.compressor, args.ranker))
    logger.write('weight: %s\n' % (args.weight))

    # 1. load the dataset
    weight_path = './Weights/' + args.dataset + '/client/' + args.model + '.pth'
    weight_root = './Weights/' + args.dataset + '/'

    if args.dataset == 'cifar-10':
        num_classes = 10
        train, _, val, _ = dataloader_cifar10.Dataloader_cifar10_val(train_batch=128, test_batch=1, seed=2024)
        width, height = 32//2, 32//2
    elif args.dataset == 'cifar-100':
        num_classes = 100
        train, _, val = dataloader_cifar100.Dataloader_cifar100_val(train_batch=128, test_batch=100, seed=2024)
        width, height = 32//2, 32//2
    elif args.dataset == 'imagenet':
        num_classes = 1000
        train, _, val, _ = dataloader_imagenet.Dataloader_imagenet_integrated(seed=2024)
        width, height = 224//2, 224//2

    if args.model == 'mobilenetV2':
        c_model, s_model = mobilenetv2.mobilenetv2_splitter(num_classes = num_classes, weight_root=weight_root, partition=-1)

    elif args.model == 'mobilenetV3':
        c_model, s_model = mobilenetv3.mobilenetv3_splitter(num_classes = num_classes, weight_root=weight_root, partition=-1)

    elif args.model == 'resnet':
        c_model, s_model = resnet.resnet_splitter(num_classes = num_classes, weight_root=weight_root, layers = args.resnetsize, partition=-1)
    
    c_model.eval()
    s_model.eval()
    c_model = c_model.to('cuda:0')
    s_model = s_model.to('cuda:0')

    gates = []
    # create gate for 10, 20, 30, 50
    g_rate = [0.2]
    logger.write('g_rate: %s\n' % (g_rate))
    for i in range(len(g_rate)):
        gates.append(gatedmodel.GateCNN_POS(int(32*g_rate[i]), width, height, o_size=1, n_ch=32, rate=g_rate[i]))
        gates[i] = gates[i].to('cuda:0')
        gates[i].train()

    optimizers = []
    for i in range(len(g_rate)):
        optimizers.append(torch.optim.Adam(gates[i].parameters(), lr=0.001))

    loss_f = torch.nn.MSELoss()
    softmax = torch.nn.Softmax(dim=1)
    
    val_loss = -1
    for epoch in range(50):
        for ind, data in tqdm(enumerate(train)):
            for i in range(len(g_rate)):
                gates[i].train()
            
            img, label = data
            img = img.to('cuda:0')
            label = label.to('cuda:0') # b, 1
            
            out = c_model(img)
            # logger.write('out shape: %s\n' % (out[0,:,:2,:2]))
            # out is not needed
            outcpu = out.cpu()
            c_rank = utils.ranker_zeros(out, 0.1, 0.5)
            
            nc_embs = []
            for i in range(len(g_rate)):
                nc_emb = utils.remover_zeros(out, c_rank, 32, g_rate[i])
                nc_emb = nc_emb.to('cuda:0')
                nc_embs.append(nc_emb)

            preds = []
            for i in range(len(g_rate)):
                pred = gates[i](nc_embs[i], c_rank[:, :int(outcpu.shape[1]*g_rate[i])])
                preds.append(pred)

            # fill zeros
            # filled_out = []
            # logger.write('nc_embs shape: %s\n' % ((nc_embs[0][0,:,:2,:2])))
            # logger.write('c_rank shape: %s\n' % ((c_rank[0])))
            # for i in range(len(g_rate)):
                # filled_out.append(utils.fill_zeros(nc_embs[i].cpu(), outcpu.shape, c_rank[:, :int(outcpu.shape[1]*g_rate[i])]).to('cuda:0'))
            # logger.write('filled_out shape: %s\n' % ((filled_out[0][0,:,:2,:2])))

            # infer
            # target_pre = []
            # for i in range(len(g_rate)):
            #     target_pre.append(s_model(filled_out[i]).detach())
            
            target_gt = s_model(out).detach() # b, c
            # conf_pre = []
            conf_gt = []
            for i in range (len(target_gt)):
                # for j in range(len(g_rate)):
                #     conf_pre.append(softmax(target_pre[j])[i][label[i]])
                conf_gt.append(softmax(target_gt)[i][label[i]])
            # conf_pre = torch.stack(conf_pre).unsqueeze(1) # b, 1
            conf_gt = torch.stack(conf_gt).unsqueeze(1)

            loss = []
            for i in range(len(g_rate)):
                optimizers[i].zero_grad()
                # loss1 = loss_f(conf_pre, conf_gt)
                loss2 = loss_f(preds[i], conf_gt)
                # loss.append(loss1 + loss2)
                loss.append(loss2)

            for i in range(len(g_rate)):
                loss[i].backward()
                optimizers[i].step()
            if ind % 100 == 0:
                logger.write('epoch: %d ' % (epoch))
                # logger.write('conf pre-gt loss: %f ' % (loss1.item()))
                logger.write('pred-conf gt loss: %f\n' % (loss2.item()))
            

        # val 
        for i in range(len(g_rate)):
            gates[i].eval()
        with torch.no_grad():
            e_val_loss = 0
            for ind, data in tqdm(enumerate(val)):
                img, label = data
                img = img.to('cuda:0')
                label = label.to('cuda:0')
                out = c_model(img)
                outcpu = out.cpu()
                c_rank = utils.ranker_zeros(out, 0.1, 0.5)
                nc_embs = []
                for i in range(len(g_rate)):
                    nc_emb = utils.remover_zeros(out, c_rank, 32, g_rate[i])
                    nc_emb = nc_emb.to('cuda:0')
                    nc_embs.append(nc_emb)
                preds = []
                for i in range(len(g_rate)):
                    pred = gates[i](nc_embs[i], c_rank[:, :int(outcpu.shape[1]*g_rate[i])])
                    preds.append(pred)
                # filled_out = []
                # for i in range(len(g_rate)):
                #     filled_out.append(utils.fill_zeros(nc_embs[i].cpu(), outcpu.shape, c_rank[:, :int(outcpu.shape[1]*g_rate[i])]).to('cuda:0'))
                # target_pre = []
                # for i in range(len(g_rate)):
                #     target_pre.append(s_model(filled_out[i]).detach())
                target_gt = s_model(out).detach()
                # conf_pre = []
                conf_gt = []
                for i in range (len(target_gt)):
                    # for j in range(len(g_rate)):
                    #     conf_pre.append(softmax(target_pre[j])[i][label[i]])
                    conf_gt.append(softmax(target_gt)[i][label[i]])
                # conf_pre = torch.stack(conf_pre).unsqueeze(1)
                conf_gt = torch.stack(conf_gt).unsqueeze(1)
                loss = []
                for i in range(len(g_rate)):
                    loss.append(loss_f(preds[i], conf_gt))
                # for i in range(len(g_rate)):
                #     logger.write('val loss: %f\n' % (loss[i].item()))
                e_val_loss = e_val_loss + loss[0].item()
            if val_loss == -1 or val_loss > e_val_loss:
                val_loss = e_val_loss
                # store weight
                weight_path = './Weights/' + args.dataset + '/gate/' + args.model + '_'+ str(epoch) + '_'+str(int(g_rate[0]*10)) + '.pth'
            for i in range(len(g_rate)):
                torch.save(gates[i].state_dict(), weight_path)
            logger.write('val_loss: %s' % (val_loss))

if __name__ == '__main__':
    print('enter')
    parser = argparse.ArgumentParser()
    # we need the name of model, the name of dataset
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--compressor', type=str, default='home', help='compressor name')
    parser.add_argument('--dataset', type=str, default='cifar-10', help='dataset name')
    parser.add_argument('--mobilev3size', type=str, default='small', help='the size of the mobilev3')
    parser.add_argument('--model', type=str, default='mobilenetV2', help='model name')
    parser.add_argument('--ranker', type=str, default='zeros', help='ranker name')
    parser.add_argument('--resnetsize', type=int, default=18, help='resnet layers')
    parser.add_argument('--weight', type=str, default='./Weight/cifar-10/', help='weight path')
    args = parser.parse_args()
    print(args)
    main(args)
    
