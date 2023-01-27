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
        nxG:
    '''
    nxG = nx.from_numpy_array(t_adj_nd)
    return nxG

def nx_neb_graph_from_symadj(t_sym_adj_nd, id_pos_dict):
    '''
    Args:
        t_sym_adj_nd: the symm adjacency matrix with original weights of edges
        id_pos_dict: the node_id <-> position on x-y axis
    
    Return:
        
    '''
    canvas_nxG = nx.from_numpy_array(t_sym_adj_nd)
    return canvas_nxG
    
    

if __name__ == '__main__':
    pass