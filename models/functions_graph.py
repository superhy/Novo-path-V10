'''
@author: Yang Hu
'''

import math
import os
import pickle

import networkx as nx
import numpy as np


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


''' ------------- algorithm for build the self-attention & neighbors based graph -------------- '''

def eud_dist(pos_a, pos_b):
    '''
    calculate the euclidean distance between pos_a and pos_b
    '''
    a = np.array(pos_a)
    b = np.array(pos_b)
    return np.linalg.norm(a-b)

def check_near_pair(node_a, node_b, id_pos_dict, close_thd=2.0):
    '''
    check is the input node pair is close enough
    
    Args:
        node_a, node_b: just the node id
    '''
    return eud_dist(id_pos_dict[node_a], id_pos_dict[node_b]) <= close_thd

def check_far_node(check_node, exist_nodes, id_pos_dict, far_thd=2.0):
    '''
    check if the prepared nodes is enough far from the exist nodes in old graph
    '''
    far_flag = True
    for ex_node in exist_nodes:
        if eud_dist(id_pos_dict[check_node], id_pos_dict[ex_node]) <= far_thd:
            far_flag = False
    return far_flag

def nx_neb_graph_from_symadj(t_sym_adj_nd, id_pos_dict,
                             T_n=20.0, T_e_1=0.2, T_e_2=0.1):
    '''
    Args:
        t_sym_adj_nd: the symm adjacency matrix with original weights of edges
        id_pos_dict: the node_id <-> position on x-y axis
    
    Return:
        
    '''
    canvas_nxG = nx.from_numpy_array(t_sym_adj_nd)
    # print(canvas_nxG)
    neig_nxG = nx.Graph()
    new_old_nodeid_dict, old_new_nodeid_dict, new_roots = {}, {}, []
    new_node_id, new_id_pos_dict = 0, {}
    
    ''' build the root nodes in neighbors-based new graph '''
    for old_node in canvas_nxG.nodes():
        # old_node is the node_id on old graph
        print(canvas_nxG.degree(old_node, weight='weight'))
        if canvas_nxG.degree(old_node, weight='weight') >= T_n:
            # print(old_node, canvas_nxG.degree(old_node, weight='weight'))
            # check if enough far from the exist nodes first
            exist_nodes = new_old_nodeid_dict.values()
            if check_far_node(old_node, exist_nodes, id_pos_dict) is False:
                continue
            # next, can be added to the new graph
            new_old_nodeid_dict[new_node_id] = old_node
            old_new_nodeid_dict[old_node] = new_node_id
            new_id_pos_dict[new_node_id] = id_pos_dict[old_node]
            new_roots.append(new_node_id)
            neig_nxG.add_node(new_node_id)
            new_node_id += 1
    # print(neig_nxG.nodes(), neig_nxG.edges(), new_old_nodeid_dict)
    
    ''' build the 1st round neighb nodes based on the root nodes '''
    for root in new_roots:
        old_root = new_old_nodeid_dict[root]
        for old_neig in canvas_nxG.neighbors(old_root):
            # print(canvas_nxG.get_edge_data(old_neig, old_root)['weight'])
            if check_near_pair(old_neig, old_root, id_pos_dict) and canvas_nxG.get_edge_data(old_neig, old_root)['weight'] >= T_e_1:
                # check if need to add a new node
                if old_neig not in new_old_nodeid_dict.values():
                    new_old_nodeid_dict[new_node_id] = old_neig
                    old_new_nodeid_dict[old_neig] = new_node_id
                    new_id_pos_dict[new_node_id] = id_pos_dict[old_neig]
                    neig_nxG.add_node(new_node_id)
                    new_node_id += 1
                # check if need to add a new edge
                new_neig, new_root = old_new_nodeid_dict[old_neig], old_new_nodeid_dict[old_root]
                if (new_neig, new_root) not in neig_nxG.edges():
                    neig_nxG.add_edge(new_neig, new_root, weight=canvas_nxG.get_edge_data(old_neig, old_root)['weight'])
    # print(neig_nxG.nodes(), neig_nxG.edges(), new_old_nodeid_dict)
                    
    ''' build the 2nd round BFS extension nodes based on  '''
    extend_nodes = list(neig_nxG.nodes())
    # out all root nodes from the search list
    for r_id in new_roots:
        extend_nodes.remove(r_id)
    while len(extend_nodes) > 0:
        for ext_node in extend_nodes:
            old_ext = new_old_nodeid_dict[ext_node]
            for old_ard in canvas_nxG.neighbors(old_ext):
                # print(canvas_nxG.get_edge_data(old_ard, old_ext)['weight'])
                if check_near_pair(old_ard, old_ext, id_pos_dict) and canvas_nxG.get_edge_data(old_ard, old_ext)['weight'] >= T_e_2:
                    # check if need to add a new node
                    if old_ard not in new_old_nodeid_dict.values():
                        new_old_nodeid_dict[new_node_id] = old_ard
                        old_new_nodeid_dict[old_ard] = new_node_id
                        new_id_pos_dict[new_node_id] = id_pos_dict[old_ard]
                        neig_nxG.add_node(new_node_id)
                        extend_nodes.append(new_node_id)
                        new_node_id += 1
                    # check if need to add a new edge
                    new_ard, new_ext = old_new_nodeid_dict[old_ard], old_new_nodeid_dict[old_ext]
                    if (new_ard, new_ext) not in neig_nxG.edges():
                        neig_nxG.add_edge(new_ard, new_ext, weight=canvas_nxG.get_edge_data(old_ard, old_ext)['weight'])
            extend_nodes.remove(ext_node)
    # print(neig_nxG.nodes(), neig_nxG.edges(), new_old_nodeid_dict)
    
    return neig_nxG, new_id_pos_dict
    
    
if __name__ == '__main__':
    pass