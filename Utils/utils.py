import numpy as np
import scipy.stats 

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
