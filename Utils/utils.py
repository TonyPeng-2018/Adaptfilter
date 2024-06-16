import numpy as np
import scipy.stats 
import os 
import torch

def calculate_entropy(embs, percentage):
    # embs are torch tensors
    # percentage shows the number of embeddings to select
    embs_entropy = []
    for i in range(embs.size(len(embs.size())-3)):
        embs_entropy.append(scipy.stats.entropy(embs[:, i].reshape(-1))) # b, c
    # get the top percentage of the channels
    embs_entropy = np.array(embs_entropy)
    indices = np.argsort(embs_entropy)
    num_selected = int(embs.size(1) * percentage)
    selected_indices = indices[:num_selected]
    return selected_indices
    # out b, c*p, h, w or c*p, h, w

def ranker_entropy(embs, percentage):
    # calculate the entropy of the embeddings
    selected_indices = calculate_entropy(embs, percentage)
    # select indice from test embeddings
    selected_embeddings = embs[:, selected_indices] # b, c*p, h, w
    return selected_embeddings, selected_indices

class APLogger():
    def __init__(self, path):
        self.path = path
        # check if path exists
        path_split = path.split('/')
        # create if the path doesn't exist
        for i in range(1, len(path_split)):
            if not os.path.exists('/'.join(path_split[:i])):
                os.mkdir('/'.join(path_split[:i]))
        self.file = open(path, 'w')
    def write(self, text):
        # check the type of text
        if type(text) != str:
            text = str(text)
        self.file.write(text)
    def close(self):
        self.file.close()
    def __del__(self):
        self.close()

def get_latest_weights(mdata, mname, mtype):
    # get the latest weights from the path
    s_weights = os.listdir('./Weights/'+mdata+'/'+mtype+'/')
    s_weights = [i for i in s_weights if mname+'_' in i]
    s_weights.sort()
    s_time = s_weights[-1].split('_')[-6:]
    s_time = '_'.join(s_time)
    return s_time

def fill_zeros(embs, shape, ind = None):
    # the out put size is the same as the input size
    # shape is the rate of the filter
    out = torch.zeros(shape) # b, c, h, w
    out[:,ind,:,:] = embs
    return out


if __name__ == '__main__':
    import torch
    embs = torch.randn(2, 3, 32, 32)
    selected_embs, selected_indices = ranker_entropy(embs, 0.25)
    print(selected_embs.size(), selected_indices)
    # test a loger
    logger = APLogger('./Logs/test.log')
    # test the get_latest_weights
    print(get_latest_weights('cifar-10', 'gate', 'GatedMLP'))