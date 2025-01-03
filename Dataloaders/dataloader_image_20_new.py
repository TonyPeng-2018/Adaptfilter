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
    def __init__(self, path):
        np.random.seed(2024)
        self.folder_path = path
        # load the class_index
        self.class_index = {}
        with open('./imagenet_class_index.json', 'r') as f:
            self.class_index = json.load(f) # nclass: (name, wclass)
        # label to nclass
        self.str_2_all_class = {}
        self.str_2_class_name = {}
        for nclass, (fname, cname) in self.class_index.items():
            self.str_2_all_class[fname] = int(nclass) # nxx: label(1k)
            self.str_2_class_name[fname] = cname # nxx: name

        train_classes = sorted(os.listdir(self.folder_path))
        self.str_2_train_class = {k: self.str_2_all_class[k] for k in train_classes}
        self.str_2_train_label = {k: i for i, k in enumerate(train_classes)}
        
        # reverse the dict
        self.train_label_2_str = {v: k for k, v in self.str_2_train_label.items()}
        self.train_class_2_str = {v: k for k, v in self.str_2_train_class.items()}
        self.all_class_2_str = {v: k for k, v in self.str_2_all_class.items()}
        self.all_class_2_name = {v: k for k, v in self.str_2_class_name.items()}

        self.image_path = []
        self.label = []
        self.image_name = []
        self.label_name = []

        for folder in os.listdir(self.folder_path): # nxxx, nxxx ...
            for file in os.listdir(self.folder_path + folder):
                self.image_path.append(self.folder_path + '/' + folder + '/' + file)
                self.label.append(self.str_2_train_label[folder])
                self.label_name.append(self.str_2_train_class[folder])
                self.image_name.append(int(file.split('.')[0].split('_')[-1]))
        # print(self.a_dict[0]) # path, label, crop
        self.len = len(self.image_path)
    
    def __len__(self):
        return self.len
        
    def return_class_index(self):
        return self.class_index

class Dataloader_imagenet_20(Dataset):
    def __init__(self, dataset, transform, quality):
        self.image_path = dataset.image_path
        self.label = dataset.label
        self.image_name = dataset.image_name
        self.label_name = dataset.label_name
        self.len = len(self.image_path)
        self.quality = quality
        self.save_count = 0
        
        if transform:
            self.trans = self.transform()
        else:
            self.trans = self.default_transform()

    def __len__(self):  
        return self.len

    def __getitem__(self, idx):
        f_path = self.image_path[idx]
        label = self.label[idx]
        image_name = self.image_name[idx]
        img = Image.open(f_path)
        img = img.convert('RGB')

        # store the image
        if self.quality != -1 and self.save_count < 60:
            img = img.resize((256, 256))
            img = img.crop((16, 16, 240, 240))
            if not os.path.exists('data/jpeg-224/'):
                os.mkdir('data/jpeg-224/')
            if not os.path.exists(f'data/jpeg-224/{self.quality}'):
                os.mkdir(f'data/jpeg-224/{self.quality}')
            img_path = f'data/jpeg-224/{self.quality}/{idx}.jpg'

            # if not os.path.exists('data/jpeg-uncut/'):
            #     os.mkdir('data/jpeg-uncut/')
            # if not os.path.exists(f'data/jpeg-uncut/{self.quality}'):
            #     os.mkdir(f'data/jpeg-uncut/{self.quality}')
            # img_path = f'data/jpeg-uncut/{self.quality}/{idx}.jpg'
            img.save(img_path, quality=self.quality)
            img = Image.open(img_path)
            img = img.convert('RGB')
            self.save_count += 1

        if self.trans != None:
            img = self.trans(img)

        labels = {
            'label': torch.tensor(label),
            'image_name': torch.tensor(image_name)
        }

        return img, labels

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
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                    ]
                )
        return transform


def Dataloader_imagenet_20_integrated(train_batch = 128, test_batch = 64, transform=True, test_only=False, JPEG=-1):

    if not test_only:
        trainset = Dataset_imagenet_20(path = 'data/imagenet-20-new/train/')
        valset = Dataset_imagenet_20(path = 'data/imagenet-20-new/val/')
        train = Dataloader_imagenet_20(trainset, transform=transform, quality=JPEG)
        val = Dataloader_imagenet_20(valset, transform=transform, quality=JPEG)
        train = torch.utils.data.DataLoader(train, batch_size=train_batch, shuffle=True)
        val = torch.utils.data.DataLoader(val, batch_size=test_batch, shuffle=True)
    testset = Dataset_imagenet_20(path = 'data/imagenet-20-new/test/') 
    test = Dataloader_imagenet_20(testset, transform=transform, quality=JPEG)
    test = torch.utils.data.DataLoader(test, batch_size=test_batch, shuffle=True)
    
    if not test_only:
        return train, test, val
    else:
        return test

if __name__ == '__main__':
    for quality in [1,3,5,7,9,
                    10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        test = Dataloader_imagenet_20_integrated(train_batch=1, test_batch=1, test_only=True, JPEG=quality)
        from tqdm import tqdm
        for i, data in tqdm(enumerate(test)):
            continue
    # for i, data in enumerate(train):
    #     inputs, labels = data
    #     print(inputs.size(), labels['label'].size())
    #     # show image 0 and labels
    #     break
    # Dataset_imagenet_20(path = '/home/tonypeng/Workspace1/adaptfilter/data/imagenet-20-new/train/')
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
