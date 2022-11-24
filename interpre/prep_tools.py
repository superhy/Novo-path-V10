'''
@author: Yang Hu
'''
import os
import pickle
import random

from sklearn.manifold._t_sne import TSNE

import numpy as np


def store_nd_dict_pkl(_env_vis_store_dir,
                      store_vis_dict, store_pkl_name):
    if not os.path.exists(_env_vis_store_dir):
        os.makedirs(_env_vis_store_dir)
    with open(os.path.join(_env_vis_store_dir, store_pkl_name), 'wb') as f_pkl:
        pickle.dump(store_vis_dict, f_pkl)

def load_vis_pkg_from_pkl(_env_vis_store_dir,
                           package_pkl_name):
    pkl_filepath = os.path.join(_env_vis_store_dir, package_pkl_name)
    with open(pkl_filepath, 'rb') as f_pkl:
        store_vis_dict = pickle.load(f_pkl)
    return store_vis_dict

def safe_random_sample(pickpool, K):
    
    if len(pickpool) > K:
        return random.sample(pickpool, K)
    else:
        return pickpool
    
def fold_id_list(id_list, batchsize):
    '''
    fold a list to several batches
    '''
    batch_list = []
    nb_batches = len(id_list) / batchsize + 1
    for batch in range(nb_batches):
        if (batch + 1) * batchsize < len(id_list):
            this_batch = id_list[batch * batchsize: (batch + 1) * batchsize]
        else:
            this_batch = id_list[batch * batchsize:]
        batch_list.append(this_batch)
    
    return batch_list
    
def tSNE_transform(vectors, output_dim=2):
    '''
    Args:
        vectors: any iterable of encodes
    '''
    encodes_X = np.array(vectors)
    embed_X = TSNE(n_components=output_dim).fit_transform(encodes_X)
    embeds = embed_X.tolist()
    return embeds 


if __name__ == '__main__':
    pass