'''
Created on 7 Oct 2022

@author: laengs2304
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
from vit_pytorch.vit import ViT

from wsi.filter_tools import apply_image_filters_he, apply_image_filters_psr, \
    apply_image_filters_cd45
from wsi.image_tools import np_to_pil
from wsi.slide_tools import original_slide_and_scaled_pil_image, \
    slide_to_scaled_np_image
    
import networkx as nx
import matplotlib.pyplot as plt


def test_filter_slide_img():
    # slide_filepath = 'D:/FLINC_dataset/slides/yang_psr_fib/tissues/23910-157_Sl040-C24-PSR.ndpi'
    # slide_filepath = 'D:/FLINC_dataset/slides/yang_he_stea/tissues/23910-157_Sl049-C32-HE.ndpi'
#     slide_filepath = 'D:/FLINC_dataset/slides/yang_cd45_u/tissues/23910-158_Sl251.ndpi'
    
    slide_filepath = 'D:/IBD-Matthias/example_dx/slides/6574_20_l1 - 2022-08-02 13.26.48.ndpi'
    
    np_slide_img, _, _, _, _ = slide_to_scaled_np_image(slide_filepath)
    
    np_filtered_img = apply_image_filters_he(np_slide_img)
    # np_filtered_img = apply_image_filters_psr(np_slide_img)
#     np_filtered_img = apply_image_filters_cd45(np_slide_img)
    
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
    
    
test_filter_slide_img() # 1
# test_vit_forward() # 2
# test_networkx() # 3


