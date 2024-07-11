# this file is for device on the client side

# load the dataset

# This file is for trainning
# Run this on the server, or as we called offline. 

import argparse
import base64
import cv2
import datetime
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
        width, height = 32//2, 32//2
    elif args.dataset == 'cifar-100':
        num_classes = 100
        width, height = 32//2, 32//2
    elif args.dataset == 'imagenet-mini':
        num_classes = 100
    elif args.dataset == 'imagenet-tiny':
        num_classes = 200
    elif args.dataset == 'imagenet':
        num_classes = 1000
        width, height = 224//2, 224//2
    
    if args.model == 'mobilenetV2':
        c_model = mobilenetv2.mobilenetv2_splitter_client(num_classes = num_classes, weight_root=weight_root, device='cpu')

    elif args.model == 'mobilenetV3':
        c_model = mobilenetv3.mobilenetv3_splitter_client(num_classes = num_classes, weight_root=weight_root, device='cpu')

    elif args.model == 'resnet':
        c_model = resnet.resnet_splitter_client(num_classes = num_classes, weight_root=weight_root, device='cpu', layers = args.resnetsize)
    
    c_model.eval()
    c_model = torch.ao.quantization.quantize_dynamic(c_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

    # 2. dataset
    # directly read bmp image from the storage
    data_root = '../data/' + args.dataset + '-client/'
    images_list = os.listdir(data_root)
    # remove the label file
    images_list.remove('labels.txt')
    images_list = sorted(images_list)
    
    ML_time = 0
    ML_inf_time = 0
    ML_gate_time = 0
    ML_encode_time = 0

    image_batch = 1
    image_bucket = []
    image_count = 0

    i_stop = 600

    # gates = []
    # g_rate = [0.1, 0.2, 0.3]

    # g_w_root = '/home/pi302/Workspace/Adaptfilter/Weights/imagenet/gate/'
    # g_weights = [g_w_root+'mobilenetV2_0_1.pth', g_w_root+'mobilenetV2_12_2.pth', g_w_root+'mobilenetV2_0_3.pth']
    # for i in range(len(g_rate)):
    #     gates.append(gatedmodel.GateCNN_POS(i_size=int(32*g_rate[i]),
    #                                         width=width,
    #                                         height=height,
    #                                         o_size=1,
    #                                         n_ch=32,
    #                                         rate = g_rate[i]))
    #     gates[i].load_state_dict(torch.load(g_weights[i], map_location='cpu'))
        
    # for gate in gates:
    #     gate.eval()
    #     gate = torch.ao.quantization.quantize_dynamic(gate, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

    g_thred = [0.8, 0.85, 0.9]
    with torch.no_grad():
        for i, i_path in tqdm(enumerate(images_list)):
            # print(i)
            if i >= i_stop:
                break
            
            image_path = data_root + i_path
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = image.astype(np.float32)/255.0

            image_count += 1
            image_bucket.append(image)
            if image_count < image_batch:
                continue
            else:
                image_count = 0

            e_time = time.time()

            # 4. model
            image_nplist = np.stack(image_bucket)
            image = torch.tensor(image_nplist)
            image = image.permute(0, 3, 1, 2)
            s_time = time.time()
            with torch.no_grad():
                c_embs = c_model(image)
                s2_time = time.time()
                # c_rank, c_cut = utils.ranker_zeros(c_embs, 0.1, 0.5)
                c_embs = c_embs.squeeze(0)
                c_embs = c_embs.detach().numpy()
            s3_time = time.time()
            c_rank = utils.ranker_zeros(c_embs, 0.1, 0.5)
            s4_time = time.time()
            c_embs = torch.tensor(c_embs).unsqueeze(0)
            c_rank = torch.tensor(c_rank).unsqueeze(0)
            # print('c_embs ', c_embs.size())
            # print('c_rank ', c_rank.size())
            
            s5_time = time.time()
            outgate = -1
            # for i in range (len(g_rate)):
            #     nc_emb = utils.remover_zeros(c_embs, c_rank, 32, g_rate[i])
            #     score = gates[i](c_embs, c_rank[:,:int(32*g_rate[i])]).detach()
            #     print('score', score)
            #     if score > g_thred[i]:
            #         outgate = i
            #         break
            #     if i == len(g_rate) - 1:
            #         nc_emb = c_embs
            #         outgate = 3
            # s6_time = time.time()

            # c_encode = base64.b64encode(nc_emb)
            # c2_encode = base64.b64encode(c_rank)
            # s7_time = time.time()
            # store c2_encode
            


            ML_inf_time += s2_time - s_time
            # ML_gate_time += s6_time - s5_time
            # ML_encode_time += s7_time - s6_time
            # ML_time += s2_time - s_time + s4_time - s3_time + s6_time - s5_time + s7_time - s6_time
            # ML_time += s2_time - s_time + s4_time - s3_time + s6_time - s5_time
            ML_time += s2_time - s_time + s4_time - s3_time
            # print('out gate', outgate)
            # encode it 
            # 5. send the data to the server
            # skipped

            image_bucket = []

    ML_time /= i_stop
    ML_encode_time /= i_stop
    ML_inf_time /= i_stop
    
    print('ML_time: %f, ML_inf_time: %f, ML_encode_time: %f\n' % (ML_time, ML_inf_time, ML_encode_time))


if __name__ == '__main__':
    print('enter')
    parser = argparse.ArgumentParser()
    # we need the name of model, the name of dataset
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--compressor', type=str, default='home', help='compressor name')
    parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name')
    parser.add_argument('--mobilev3size', type=str, default='small', help='the size of the mobilev3')
    parser.add_argument('--model', type=str, default='mobilenetV2', help='model name')
    parser.add_argument('--ranker', type=str, default='zeros', help='ranker name')
    parser.add_argument('--resnetsize', type=int, default=18, help='resnet layers')
    parser.add_argument('--weight', type=str, default='./Weight/imagenet/', help='weight path')
    args = parser.parse_args()
    print(args)
    main(args)
    
