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

class Dataset_ccpd(Dataset):
    def __init__(self, device = 'home', size ='small'):
        if device == 'home':
            if size == 'small':
                self.root_path = '/home/tonypeng/Workspace1/adaptfilter/data/CCPD_small/'
            else:
                self.root_path = '/home/tonypeng/Workspace1/adaptfilter/data/CCPD_large/'
        if device == 'tintin':
            if size == 'small':
                self.root_path = '/data/anp407/CCPD_small/'
            else:
                self.root_path = '/data/anp407/CCPD_large/'

        # load the json file
        with open(self.root_path + 'train_label.json') as f:
            self.trl = json.load(f)
        with open(self.root_path + 'test_label.json') as f:
            self.tl = json.load(f)
        with open(self.root_path + 'val_label.json') as f:
            self.vl = json.load(f)

    def __len__(self):
        return len(self.trl) + len(self.tl) + len(self.vl)
    
    def get_labels(self):
        return self.trl, self.tl, self.vl
    
class Dataloader_ccpd(Dataset):
    def __init__(self, loader, size ='small', dtype = 'train', device='home'):
        self.loader = loader
        self.size = size
        
        if device == 'home':
            if size == 'small':
                self.root_path = '/home/tonypeng/Workspace1/adaptfilter/data/CCPD_small/'+dtype+'/'
            else:
                self.root_path = '/home/tonypeng/Workspace1/adaptfilter/data/CCPD_large/'+dtype+'/'
        if device == 'tintin':
            if size == 'small':
                self.root_path = '/data/anp407/CCPD_small/'+dtype+'/'
            else:
                self.root_path = '/data/anp407/CCPD_large/'+dtype+'/'
                
        # ind to loader key
        self.ind_to_key = {}
        for i, k in enumerate(self.loader.keys()):
            self.ind_to_key[i] = k

    def __len__(self):  
        return len(self.loader)

    def __getitem__(self, idx):
        k = self.ind_to_key[idx]
        img = Image.open(self.root_path + k)
        img = img.convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(self.loader[k])
    
    def transform(self, img):
        img = img.convert('RGB')
        # resize to 224, 224
        if self.size == 'small':
            img = transforms.Resize((224, 224))(img)
        img = transforms.ToTensor()(img)
        return img
    
def Dataloader_ccpd_integrated(device = 'home', train_batch = 128, test_batch = 100, size = 'small'):
    dataset = Dataset_ccpd(size=size, device=device)
    trl, tl, vl = dataset.get_labels()
    train = Dataloader_ccpd(trl, dtype = 'train', size=size, device=device)
    test = Dataloader_ccpd(tl, dtype = 'test', size=size, device=device)
    val = Dataloader_ccpd(vl, dtype = 'val', size=size, device=device)
    train = torch.utils.data.DataLoader(train, batch_size=train_batch, shuffle=True)
    test = torch.utils.data.DataLoader(test, batch_size=test_batch, shuffle=True)
    val = torch.utils.data.DataLoader(val, batch_size=train_batch, shuffle=True)
    return train, test, val

if __name__ == '__main__':
    # train, test, val = Dataloader_visdrone_integrated()
    # for i, data in enumerate(train):
    #     print(data)
    #     break
    a = Dataloader_ccpd_integrated(size='small')
