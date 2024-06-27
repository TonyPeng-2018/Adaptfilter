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

class Dataset_imagenet():
    def __init__(self, device):
        # get the set for train, test and val
        # path to the file, the number of it, 
        np.random.seed(2024)
        if device == 'home':
            self.r_path = '/home/tonypeng/Workspace1/adaptfilter/data/imagenet/ILSVRC/'
        elif device == 'tintin':
            self.r_path = '/data/anp407/imagenet/ILSVRC/'
        self.d_path = self.r_path + 'Data/CLS-LOC/train/'
        self.t_set = self.r_path + 'ImageSets/CLS-LOC/train_cls.txt' # this should be splited into train and val

        # load the class_index
        self.class_index = {}
        with open('./imagenet_class_index.json', 'r') as f:
            self.class_index = json.load(f) # nclass: (name, wclass)
        # label to nclass
        self.c_to_n = {}
        for nclass, (fname, _) in self.class_index.items():
            self.c_to_n[fname] = int(nclass)

        # read the file to get the dict
        self.a_dict = {} # count: (name, label)
        with open(self.t_set, 'r') as f:
            lines = f.readlines() # path, count
            for line in lines:
                line = line.strip().split()
                count = line[1]
                label = line[0].split('/')[0]
                f_path = self.d_path + line[0]
                self.a_dict[count] = (f_path, self.c_to_n[label])
        self.len = len(self.a_dict)
        self.t_v_sampler = np.random.choice(self.len, int(self.len*0.4), replace=False)
        self.tr_sampler = np.delete(np.arange(self.len), self.t_v_sampler)
        self.t_sampler = np.random.choice(len(self.t_v_sampler), int(len(self.t_v_sampler)*0.5), replace=False)
        self.v_sampler = np.delete(self.t_v_sampler, self.t_sampler)
        # # shuffle the dict
        # np.random.shuffle(self.tr_sampler)
        # np.random.shuffle(self.t_sampler)
        # np.random.shuffle(self.v_sampler)
        # simplify the dict
        self.tr_dict = {str(k): self.a_dict[str(k)] for k in self.tr_sampler}
        self.t_dict = {str(k): self.a_dict[str(k)] for k in self.t_sampler}
        self.v_dict = {str(k): self.a_dict[str(k)] for k in self.v_sampler}
    
    def return_sampler(self):
        return self.tr_sampler, self.t_sampler, self.v_sampler
    
    def return_dict(self):
        return self.tr_dict, self.t_dict, self.v_dict
    
    def return_class_index(self):
        return self.class_index

class Dataloader_imagenet(Dataset):
    def __init__(self, sampler, files, transform):
        self.sampler = sampler
        self.files = files
        
        if transform:
            self.trans = self.transform()
        else:
            self.trans = None

    def __len__(self):  
        return len(self.sampler)

    def __getitem__(self, idx):

        count = self.sampler[idx]
        f_path, label = self.files[str(count)]
        img = Image.open(f_path+'.JPEG').convert('RGB')

        if self.trans != None:
            img = self.trans(img)
        return img, torch.tensor(label)

    def transform(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
        return transform
        
