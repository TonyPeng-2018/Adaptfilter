# this dataloader is for gate training 

import torch
import os

class dataloader_gate(torch.utils.data.Dataset):
    # load the dataset
    def __init__(self, embeddings_folder, labels_folder):
        self.embeddings_folder = embeddings_folder
        self.embeddings_files = sorted(os.listdir(embeddings_folder))
        self.labels_folder = labels_folder
        self.labels_files = sorted(os.listdir(labels_folder))

    def __len__(self):
        return self.embeddings_files.__len__()

    def __getitem__(self, idx):
        self.embeddings = torch.load(self.embeddings_folder + self.embeddings_files[idx])
        self.labels = torch.load(self.labels_folder + self.labels_files[idx])
        return self.embeddings, self.labels
    
class dataloader_gate_correct(torch.utils.data.Dataset):
    # inputdata: the embs from the client
    # labeldata: the softmax value of the server embeddings for the correct class
    def __init__(self, emb_folder, emb_out_folder, label_folder):
        self.emb_folder = emb_folder # cifar-10-embedding-client
        self.emb_file = sorted(os.listdir(emb_folder))
        self.emb_out_folder = emb_out_folder # cifar-10-embedding-server
        self.emb_out_file = sorted(os.listdir(emb_out_folder))
        self.label_folder = label_folder #cifar-10-labels
        self.label_files = sorted(os.listdir(label_folder))

    def __len__(self):
        return self.emb_file.__len__()

    def __getitem__(self, idx):
        self.emb = torch.load(self.emb_folder + self.emb_file[idx]) # b,e,h,w
        self.emb_out = torch.load(self.emb_out_folder + self.emb_out_file[idx]) # b,o
        self.labels = torch.load(self.label_folder + self.label_files[idx]) # b,1
        return self.emb, self.emb_out, self.labels