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


def create_nx_graph_from_npadj(t_adj_nd):
    '''
    
    Args:
        t_adj_nd: adjacency matrix numpy array for one tile, (q, k), q == k
    
    Return:
        nx_G:
        positions:
    '''
    (q, k) = t_adj_nd.shape

    nx_G = nx.from_numpy_array(t_adj_nd)
    nodes = nx.function.nodes(nx_G)
    s_nodes = []
    for i, node in enumerate(nodes):
        if len(list(nx.function.all_neighbors(nx_G, node))) > 0:
            s_nodes.append(node)
        
    # load positions from adj mat
    s = int(math.sqrt(q) ) # size of matrix
    positions = {}
    for i in range(q):
        positions[i] = (int(i % s), s - int(i / s) )
        
    return nx_G, positions, s_nodes
    

if __name__ == '__main__':
    pass