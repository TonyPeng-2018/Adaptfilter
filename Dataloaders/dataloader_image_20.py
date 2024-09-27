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

class Dataset_imagenet_20():
    def __init__(self, device):
        # get the set for train, test and val
        # path to the file, the number of it, 
        np.random.seed(2024)
        if device == 'home':
            self.r_path = '/home/tonypeng/Workspace1/adaptfilter/data/imagenet-20/'
        elif device == 'tintin':
            self.r_path = '/data/anp407/imagenet-20/'
        self.d_path = self.r_path
        
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
        self.new_c_to_n = self.c_to_n.copy()
        # change the v from 0 - 200
        count = 0
        for k in self.c_to_n.keys():
            self.new_c_to_n[k] = count # name, label
            count += 1

        # read the file to get the dict
        self.a_dict = {} # count: (name, label)
        count = 0
        # self.c_dict = {} # crop
        # for folder in self.folders:
        #     crop_info = {}
        #     crop_file = self.d_path + folder + '/' + folder + '_boxes.txt'
        #     with open(crop_file, 'r') as f:
        #         for line in f:
        #             line = line.split('\t')
        #             crop_info[line[0]] = (int(line[1]), int(line[2]), int(line[3]), int(line[4]))
        for folder in self.folders:
            for file in os.listdir(self.d_path + folder):
                s_info = (self.d_path + folder+'/' + file, self.c_to_n[folder], self.new_c_to_n[folder])
                self.a_dict[count] = s_info
                count += 1
        # print(self.a_dict[0]) # path, label, crop
        self.len = len(self.a_dict)

        self.v_sampler = np.random.choice(self.len, int(self.len*0.2), replace=False)
        self.tr_sampler = np.delete(np.arange(self.len), self.v_sampler)
        self.t_sampler = np.random.choice(len(self.tr_sampler), int(len(self.tr_sampler)*0.25), replace=False)
        self.temp_sampler = self.tr_sampler[self.t_sampler]
        self.tr_sampler = np.delete(self.tr_sampler, self.t_sampler)
        self.t_sampler = self.temp_sampler
        self.tr_dict = {str(k): self.a_dict[k] for k in self.tr_sampler}
        self.v_dict = {str(k): self.a_dict[k] for k in self.v_sampler}
        self.t_dict = {str(k): self.a_dict[k] for k in self.t_sampler}

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
            self.trans = self.default_transform()

    def __len__(self):  
        return len(self.files)

    def __getitem__(self, idx):

        count = self.sampler[idx]
        f_path, label, new_label = self.files[str(count)]
        img = Image.open(f_path).convert('RGB')

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

    def default_transform(self):
        transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )
        return transform

# input train_batch, test_batch, device, seed
# output train, test, val
def Dataloader_imagenet_20_integrated(train_batch = 128, test_batch = 100, device='home', seed=2024, transform=True):
    # input train_batch, test_batch, device, seed
    # output train, test, val
    dataset = Dataset_imagenet_20(device=device)
    tr_sampler, t_sampler, v_sampler = dataset.return_sampler()
    tr_dict, t_dict, v_dict = dataset.return_dict()
    class_index = dataset.return_class_index()
    train = Dataloader_imagenet(tr_sampler, tr_dict, transform=transform)
    test = Dataloader_imagenet(t_sampler, t_dict, transform=transform)
    val = Dataloader_imagenet(v_sampler, v_dict, transform=transform)
    train = torch.utils.data.DataLoader(train, batch_size=train_batch, shuffle=True)
    test = torch.utils.data.DataLoader(test, batch_size=test_batch, shuffle=False)
    val = torch.utils.data.DataLoader(val, batch_size=train_batch, shuffle=True)
    return train, test, val

if __name__ == '__main__':
    train, test, val, classes = Dataloader_imagenet_20_integrated()
    # for i, data in enumerate(train):
    #     inputs, labels, new_labels = data
    #     print(inputs.size(), labels.size(), new_labels.size())
    #     # show image 0 and labels

    #     break
    # import matplotlib.pyplot as plt
    # import numpy as np
    # # functions to show an image
    # def imshow(img):
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    # print(classes[str(labels[0].item())], new_labels[0])
    # imshow(inputs[0])
               
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

'''
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
'''        