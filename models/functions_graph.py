'''
@author: Yang Hu
'''

import math
import os
import pickle

import networkx as nx


def load_adj_pkg_from_pkl(graph_store_dir, adjdict_pkl_name):
    pkl_filepath = os.path.join(graph_store_dir, adjdict_pkl_name)
    with open(pkl_filepath, 'rb') as f_pkl:
        adj_pkg = pickle.load(f_pkl)
        
    return adj_pkg


def nx_graph_from_npadj(t_adj_nd):
    '''
    
    Args:
        t_adj_nd: adjacency matrix numpy array for one tile, (q, k), q == k
    
    Return:
        nx_G:
    '''
    nx_G = nx.from_numpy_array(t_adj_nd)
    return nx_G
    

if __name__ == '__main__':
    pass