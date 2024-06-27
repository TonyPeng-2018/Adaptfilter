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
            self.r_path = '/home/tonypeng/Workspace1/adaptfilter/data/imagenet-mini/'
        elif device == 'tintin':
            self.r_path = '/data/anp407/imagenet-mini/'
        self.d_path = self.r_path + 'train/'
        
        # load the class_index
        self.class_index = {}
        with open('./imagenet_class_index.json', 'r') as f:
            self.class_index = json.load(f) # nclass: (name, wclass)
        # label to nclass
        self.c_to_n = {}
        for nclass, (fname, _) in self.class_index.items():
            self.c_to_n[fname] = int(nclass)
        # get the name of folders in the train folder
        self.folders = sorted(os.listdir(self.d_path))
        # select the key in the folders
        self.c_to_n = {k: v for k, v in self.c_to_n.items() if k in self.folders}
        # change the v from 0 - 200
        count = 0
        for k in self.c_to_n.keys():
            self.c_to_n[k] = count # name, label
            count += 1

        # read the file to get the dict
        self.a_dict = {} # count: (name, label)
        count = 0
        self.c_dict = {} # crop
        for folder in self.folders:
            crop_info = {}
            crop_file = self.d_path + folder + '/' + folder + '_boxes.txt'
            with open(crop_file, 'r') as f:
                for line in f:
                    line = line.split('\t')
                    crop_info[line[0]] = (int(line[1]), int(line[2]), int(line[3]), int(line[4]))

            for file in os.listdir(self.d_path + folder + '/images/'):
                s_info = (self.d_path + folder + '/images/' + file, self.c_to_n[folder], crop_info[file])
                self.a_dict[count] = s_info
                count += 1
        # print(self.a_dict[0]) # path, label, crop
        self.len = len(self.a_dict)

        self.v_sampler = np.random.choice(self.len, int(self.len*0.2), replace=False)
        self.tr_sampler = np.delete(np.arange(self.len), self.v_sampler)
        self.tr_dict = {str(k): self.a_dict[k] for k in self.tr_sampler}
        self.v_dict = {str(k): self.a_dict[k] for k in self.v_sampler}
        print(self.v_sampler[0])
        print(self.v_dict[0])

        
        # test
        testpath = self.r_path + 'val/'
        test_ann = testpath + 'val_annotations.txt'
        self.t_dict = {}
        count = 0
        with open(test_ann, 'r') as f:
            for line in f:
                line = line.split('\t') # val_0.JPEG	n03444034	0	32	44	62
                self.t_dict[count] = (testpath + 'images/' + line[0], self.c_to_n[line[1]], (int(line[2]), int(line[3]), int(line[4]), int(line[5])))
                count += 1

if __name__ == '__main__':
    Dataset_imagenet('home')
               
    #     self.t_v_sampler = np.random.choice(self.len, int(self.len*0.4), replace=False)
    #     self.tr_sampler = np.delete(np.arange(self.len), self.t_v_sampler)
    #     self.t_sampler = np.random.choice(len(self.t_v_sampler), int(len(self.t_v_sampler)*0.5), replace=False)
    #     self.v_sampler = np.delete(self.t_v_sampler, self.t_sampler)
    #     # # shuffle the dict
    #     # np.random.shuffle(self.tr_sampler)
    #     # np.random.shuffle(self.t_sampler)
    #     # np.random.shuffle(self.v_sampler)
    #     # simplify the dict
    #     self.tr_dict = {str(k): self.a_dict[str(k)] for k in self.tr_sampler}
    #     self.t_dict = {str(k): self.a_dict[str(k)] for k in self.t_sampler}
    #     self.v_dict = {str(k): self.a_dict[str(k)] for k in self.v_sampler}
    
    # def return_sampler(self):
    #     return self.tr_sampler, self.t_sampler, self.v_sampler
    
    # def return_dict(self):
    #     return self.tr_dict, self.t_dict, self.v_dict
    
    # def return_class_index(self):
    #     return self.class_index

# class Dataloader_imagenet(Dataset):
#     def __init__(self, sampler, files, transform):
#         self.sampler = sampler
#         self.files = files
        
#         if transform:
#             self.trans = self.transform()
#         else:
#             self.trans = None

#     def __len__(self):  
#         return len(self.sampler)

#     def __getitem__(self, idx):

#         count = self.sampler[idx]
#         f_path, label = self.files[str(count)]
#         img = Image.open(f_path+'.JPEG').convert('RGB')

#         if self.trans != None:
#             img = self.trans(img)
#         return img, torch.tensor(label)

#     def transform(self):
#         mean = (0.485, 0.456, 0.406)
#         std = (0.229, 0.224, 0.225)
#         transform = transforms.Compose(
#                     [
#                         transforms.Resize(256),
#                         transforms.CenterCrop(224),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean, std),
#                     ]
#                 )
#         return transform
        
