'''
Created on 7 Oct 2022

@author: laengs2304
'''

import os
import warnings

from einops.einops import rearrange
import torch
from vit_pytorch.vit import ViT

import matplotlib.pyplot as plt
from models.functions_graph import nx_graph_from_npadj, nx_neb_graph_from_symadj
from models.functions_vit_ext import symm_adjmats, gen_edge_adjmats, \
    filter_node_pos_t_adjmat, node_pos_t_adjmat
import networkx as nx
import networkx as nx
import numpy as np
from support.tools import normalization
from wsi.filter_tools import apply_image_filters_he, apply_image_filters_psr, \
    apply_image_filters_cd45
from wsi.image_tools import np_to_pil
from wsi.slide_tools import original_slide_and_scaled_pil_image, \
    slide_to_scaled_np_image


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

    



def test_filter_slide_img():
    # slide_filepath = 'D:/FLINC_dataset/slides/yang_psr_fib/tissues/23910-157_Sl040-C24-PSR.ndpi'
    # slide_filepath = 'D:/FLINC_dataset/slides/yang_he_stea/tissues/23910-157_Sl049-C32-HE.ndpi'
    slide_filepath = 'D:/FLINC_dataset/slides/yang_cd45_u/tissues/23910-158_Sl251-C2-CD45.ndpi'
    
#     slide_filepath = 'D:/IBD-Matthias/example_dx/slides/6574_20_l1 - 2022-08-02 13.26.48.ndpi'
    
    np_slide_img, _, _, _, _ = slide_to_scaled_np_image(slide_filepath)
    
#     np_filtered_img = apply_image_filters_he(np_slide_img)
    # np_filtered_img = apply_image_filters_psr(np_slide_img)
    np_filtered_img = apply_image_filters_cd45(np_slide_img)
    
    pil_img = np_to_pil(np_filtered_img)
    print(pil_img)
    pil_img.save('test_slide_filter.jpg')
    pil_img.show()
    
def test_vit_forward():
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 2,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 256, 256)
    
    preds = v(img) # (1, 1000)
    print(preds)
    
def test_networkx():
#     G = nx.random_geometric_graph(200, 0.125)
    G = nx.Graph()
    G.add_nodes_from([
        (0, {'pos': [0.2, 0.2]}),
        (1, {'pos': [0.4, 0.4]}),
        (2, {'pos': [0.4, 0.8]}),
        (3, {'pos': [0.8, 0.8]})
        ])
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (0, 3)])
    print(G.nodes()[0])
    
    subax1 = plt.subplot(111)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    
def test_numpy():
    test_nd = np.random.random((4, 6, 8, 8))
    print(len(test_nd.shape), test_nd.shape, np.shape(test_nd))
    
    test_nd_2 = np.array([[[0.1, 0.8, 0.2], 
                           [0.2, 0.4, 0.9], 
                           [0.9, 0.7, 0.3]],
                          [[0.3, 0.5, 0.6], 
                           [0.4, 0.2, 0.7], 
                           [0.6, 0.3, 0.1]],
                          [[0.8, 0.2, 0.4], 
                           [0.7, 0.1, 0.9], 
                           [0.9, 0.2, 0.5]],
                          [[0.1, 0.2, 0.6], 
                           [0.5, 0.4, 0.9], 
                           [0.6, 0.2, 0.6]]])
    print(test_nd_2[0, :, :])
    # print(type(test_nd_2))
    # print(test_nd_2)
    # test_nd_2[test_nd_2 >= 0.5] = 1
    # test_nd_2[test_nd_2 < 0.5] = 0
    # test_nd_2.astype('int32')
    # print(test_nd_2)
    
    sym_test_nd_2 = symm_adjmats(test_nd_2, True)
    print(sym_test_nd_2)
    
    edge_test_nd_2 = gen_edge_adjmats(sym_test_nd_2, False)
    print(edge_test_nd_2)
    
    list_edge_nd_2 = edge_test_nd_2.tolist()
    print(type(list_edge_nd_2), type(list_edge_nd_2[0]))
    
def test_nx_graph():
    
    test_nd_2 = np.array([[[0.1, 0.8, 0.2, 0.6], 
                           [0.2, 0.4, 0.9, 0.8], 
                           [0.9, 0.7, 0.3, 0.1],
                           [0.5, 0.7, 0.6, 0.4]],
                          [[0.3, 0.5, 0.6, 0.4], 
                           [0.4, 0.2, 0.7, 0.5], 
                           [0.6, 0.3, 0.1, 0.2],
                           [0.5, 0.7, 0.6, 0.7]],
                          [[0.8, 0.2, 0.4, 0.2], 
                           [0.7, 0.1, 0.9, 0.9], 
                           [0.9, 0.2, 0.5, 0.6],
                           [0.3, 0.7, 0.2, 0.4]],
                          [[0.1, 0.2, 0.6, 0.8], 
                           [0.5, 0.4, 0.9, 0.4], 
                           [0.6, 0.2, 0.6, 0.2],
                           [0.5, 0.7, 0.2, 0.4]]])
    (t, q, k) = test_nd_2.shape
    
    symm_test_nd = symm_adjmats(test_nd_2)
    onehot_test_nd = gen_edge_adjmats(symm_test_nd)
#     print(onehot_test_nd[0])
    
#     flat_test_nd = rearrange(test_nd_2[0], 'a b -> (a b)')
#     print(flat_test_nd)
#     reshape_test_nd = rearrange(flat_test_nd, '(a b) -> a b', a=4)
#     print(reshape_test_nd)

    f_onehot_test_nd, f_pos = filter_node_pos_t_adjmat(onehot_test_nd[1])
    print(f_onehot_test_nd)
    
#     nx_G, positions, s_nodes = nx_graph_from_npadj(onehot_test_nd[0])
    nx_G = nx_graph_from_npadj(f_onehot_test_nd)
    print(nx_G.edges())
    spring_pos = nx.spring_layout(nx_G)
    print(f_pos)
    print(spring_pos)
    nx.draw(nx_G, f_pos, with_labels=True)
#     nx.draw(nx_G, spring_pos, with_labels=True)
    plt.show()
    
def test_neb_nx_graph():
    test_nd_2 = np.array([[[0.1, 0.8, 0.2, 0.6], 
                           [0.2, 0.4, 0.9, 0.8], 
                           [0.9, 0.7, 0.3, 0.1],
                           [0.5, 0.7, 0.6, 0.4]],
                          [[0.3, 0.5, 0.6, 0.4], 
                           [0.4, 0.2, 0.7, 0.5], 
                           [0.6, 0.3, 0.1, 0.2],
                           [0.5, 0.7, 0.6, 0.7]],
                          [[0.8, 0.2, 0.4, 0.2], 
                           [0.7, 0.1, 0.9, 0.9], 
                           [0.9, 0.2, 0.5, 0.6],
                           [0.3, 0.7, 0.2, 0.4]],
                          [[0.1, 0.2, 0.6, 0.8], 
                           [0.5, 0.4, 0.9, 0.4], 
                           [0.6, 0.2, 0.6, 0.2],
                           [0.5, 0.7, 0.2, 0.4]]])
    (t, q, k) = test_nd_2.shape
    symm_test_nd = symm_adjmats(test_nd_2)
    
    t_symm_test_nd = symm_test_nd[0]
    print(t_symm_test_nd)
    id_pos_dict = node_pos_t_adjmat(t_symm_test_nd)
    
    canvas_nxG = nx_neb_graph_from_symadj(t_symm_test_nd, id_pos_dict)
    print(canvas_nxG.edges())
    print(canvas_nxG.get_edge_data(1, 2))
    
    
if __name__ == '__main__':
#     test_filter_slide_img() # 1
    # test_vit_forward() # 2
    # test_networkx() # 3

#     test_numpy()
    # test_nx_graph()
    test_neb_nx_graph()






