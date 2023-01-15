'''
@author: Yang Hu
'''

import networkx as nx
import math


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
    print(nx_G.edges())
    
    # load positions from adj mat
    s = int(math.sqrt(q) ) # size of matrix
    positions = {}
    for i in range(q):
        positions[i] = (int(i / s), int(i % s) )
        
    return nx_G, positions
    

if __name__ == '__main__':
    pass