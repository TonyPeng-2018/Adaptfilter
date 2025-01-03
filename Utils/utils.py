import numpy as np
import scipy.stats 
import os 
import torch
import torchvision.transforms as transforms
import time

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
        text = text + '\n'
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
    if type(embs) == torch.Tensor:
        out = torch.zeros(shape) # b, c, h, w
        # ind = b, c'
        for i in range(len(ind)):
            for j in range(len(ind[i])):
                out[i, ind[i,j],:,:] = embs[i,j,:,:]
    else: # embs = c,h,w
        out = np.zeros(shape)
        for i in range(len(ind)):
            out[ind[i],:,:] = embs[j,:,:]
    return out

def ranker_zeros(embs, thred, z_thred):
    # embs # c,h,w torch
    # zero_rate = torch.zeros(embs.shape[:2]) # b,c
    # zero_cutoff = torch.zeros(embs.shape[:1]) # b
    if type(embs) == torch.Tensor:
        # size = embs.shape[2]*embs.shape[3]
        # s_thred = size * z_thred
        zero_rate = torch.count_nonzero(embs <= thred, dim=(2,3))
        zero_rank = torch.argsort(-1*zero_rate, dim=1)
    
    else:
        # size = embs.shape[1]*embs.shape[2]
        # s_thred = size * z_thred
        # for i, emb in enumerate(embs):
            # for j in range(emb.shape[0]): # c
            #     # zeros = torch.where(emb[j,:,:]<=thred, 1, 0)
            #     # zero_rate[i,j] = torch.sum(zeros) / size
            #     t = time.time()
            #     zero_rate[i,j] = torch.count_nonzero(emb[j,:,:] <= thred)
            #     print('time', time.time()-t)
            #     # print('torch.sum(zeros)', torch.sum(zeros))
            #     # print('np.max((np.count_nonzero(zeros), 1) ', np.max((np.count_nonzero(zeros), 1)))
            #     # zero_rate[i,j] = torch.sum(zeros)/np.max((np.count_nonzero(zeros), 1))
            #     # sort the zeros rate
            #     if zero_rate[i,j] >= z_thred:
            #         zero_cutoff[i] += 1 # 0~cutoff are useful
        zero_rate = np.count_nonzero(embs, axis=(1,2)) # spend the most time
        # zero_cutoff = torch.count_nonzero(zero_rate >= s_thred, axis=1) # b
        zero_rank = np.argsort(-1*zero_rate) # first more info, least no info  # b,c
        # return zero_rank, zero_cutoff
    return zero_rank

def remover_zeros(emb, zero_rank, cutoff, per):
    # per is 25%, 50%, 75%
    # embs = c,h,w, zero_rank = c, cutoff = int, per = float
    if type(emb) == np.ndarray:
        chosen = max(int(per*cutoff), 1)
        chosen = zero_rank[:chosen]
        return emb[chosen,:,:]
    else: # embs = b, c,h,w, zero_rank = b, c, cutoff = int, per = float
        n_emb = torch.zeros(emb.shape[0], max(int(per*cutoff), 1), emb.shape[2], emb.shape[3])
        for i in range(emb.shape[0]):
            chosen = max(int(per*cutoff), 1)
            chosen = zero_rank[i,:chosen]
            n_emb[i] = emb[i,chosen,:,:]
        return n_emb

def image_transform(dataset):
    if dataset == 'cifar-10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        return mean, std
    elif dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return mean, std
    elif dataset == 'cifar-100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        return mean, std
    elif dataset == 'ccpd':
        mean = (0, 0, 0)
        std = (1, 1, 1)
        return mean, std

def gate_normal(a, x):
    return a**(3*(1/x**3)) * (1/a)**3

def gate_normal2(a, x):
    return (a*x)**3/(a**3)

def gate_normal3(a, x):
    if type(x) == np.ndarray:
        return 1-(-x+1)**(1/a)
    elif type(x) == torch.Tensor:
        return 1-torch.pow(1-x, 1/a)
    
def gate_renormal3(a, x):
    if type(x) == np.ndarray:
        return -(x-1)**a+1
    elif type(x) == torch.Tensor:
        return 1-torch.pow(x-1, a)

def float_to_uint(x, quant):
    return torch.round(x * quant)

def uint_to_float(x, quant):
    return x / quant

def return_max_min(x):
    return torch.max(x), torch.min(x)

def normalize(x):
    x_min = torch.min(x)
    x_max = torch.max(x)
    return (x - x_min)/(x_max - x_min)

def normalize_return(x):
    x_min = torch.min(x)
    x_max = torch.max(x)
    result = (x - x_min)/(x_max - x_min)
    return result, x_min, x_max

def renormalize(x, x_min, x_max):
    return x * (x_max - x_min) + x_min

if __name__ == '__main__':
    import torch
    # embs = torch.randn(2, 3, 32, 32)
    # selected_embs, selected_indices = ranker_entropy(embs, 0.25)
    # print(selected_embs.size(), selected_indices)
    # # test a loger
    # logger = APLogger('./Logs/test.log')
    # # test the get_latest_weights
    # print(get_latest_weights('cifar-10', 'gate', 'GatedMLP'))
    a = torch.randn(2, 3, 1, 1)
    r = ranker_zeros(a, 0.0)
    s = remover_zeros(a, r, 0.4)
    print(a)
    print(r)
    print(s)