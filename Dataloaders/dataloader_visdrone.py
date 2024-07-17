# here we start from the scratch
import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
import sys
import numpy as np
import json
from torch.utils.data import Dataset

class Dataset_visdrone(Dataset):
    def __init__(self, train_batch = 128, test_batch = 100, device = 'home'):
        root_path = '/home/tonypeng/Workspace1/adaptfilter/data/visdrone/'
        trl_path = root_path + 'train/labels.txt'
        tl_path = root_path + 'test/labels.txt'
        vl_path = root_path + 'val/labels.txt'

        self.tri_path = root_path + 'train/images/'
        self.ti_path = root_path + 'test/images/'
        self.vi_path = root_path + 'val/images/'

        # read the labels to dict
        with open(trl_path) as f:
            trl = f.readlines()
        with open(tl_path) as f:
            tl = f.readlines()
        with open(vl_path) as f:
            vl = f.readlines()

        self.trl = {}
        self.tl = {}
        self.vl = {}

        for i in trl:
            i = i.split(',')
            self.trl[i[0]] = int(i[1])
        for i in tl:
            i = i.split(',')
            self.tl[i[0]] = int(i[1])
        for i in vl:
            i = i.split(',')
            self.vl[i[0]] = int(i[1])

    def __len__(self):
        return len(self.trl) + len(self.tl) + len(self.vl)
    
    def get_labels(self):
        return self.trl, self.tl, self.vl
    
    def get_paths(self):
        return self.tri_path, self.ti_path, self.vi_path
    
class Dataloader_visdrone(Dataset):
    def __init__(self, loader, path):
        self.loader = loader
        self.path = path
        # ind to loader key
        self.ind_to_key = {}
        for i, k in enumerate(self.loader.keys()):
            self.ind_to_key[i] = k

    def __len__(self):  
        return len(self.loader)

    def __getitem__(self, idx):
        k = self.ind_to_key[idx]
        img = Image.open(self.path + 'img' + k)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        return img, torch.tensor(self.loader[k])
    
def Dataloader_visdrone_integrated(device = 'home', train_batch = 128, test_batch = 100):
    dataset = Dataset_visdrone()
    trl, tl, vl = dataset.get_labels()
    tri_path, ti_path, vi_path = dataset.get_paths()
    train = Dataloader_visdrone(trl, tri_path)
    test = Dataloader_visdrone(tl, ti_path)
    val = Dataloader_visdrone(vl, vi_path)
    train = torch.utils.data.DataLoader(train, batch_size=train_batch, shuffle=True)
    test = torch.utils.data.DataLoader(test, batch_size=test_batch, shuffle=True)
    val = torch.utils.data.DataLoader(val, batch_size=train_batch, shuffle=True)
    return train, test, val

if __name__ == '__main__':
    train, test, val = Dataloader_visdrone_integrated()
    for i, data in enumerate(train):
        print(data)
        break
