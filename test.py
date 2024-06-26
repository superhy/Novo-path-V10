'''
Created on 7 Oct 2022

@author: laengs2304
'''

import os
import warnings

from PIL import Image
import anndata
from einops.einops import rearrange
import torch
from vit_pytorch.vit import ViT
import h5py

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from models.functions import optimizer_adam_basic
from models.functions_clustering import assign_label
from models.functions_feat_ext import symm_adjmats, gen_edge_adjmats, \
    filter_node_pos_t_adjmat, node_pos_t_adjmat
from models.functions_graph import nx_graph_from_npadj, nx_neb_graph_from_symadj
from models.networks import ViT_Region_4_6, store_net, reload_net, \
    check_reuse_net
import networkx as nx
import networkx as nx
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
from support.tools import normalization
from wsi import slide_tools
from wsi.filter_tools import apply_image_filters_he, apply_image_filters_psr, \
    apply_image_filters_cd45, apply_image_filters_p62
from wsi.image_tools import np_to_pil
from wsi.slide_tools import original_slide_and_scaled_pil_image, \
    slide_to_scaled_np_image
from wsi import image_tools

import collections
import scipy.sparse as sp_sparse
import tables
import scanpy as sc


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

    



def test_filter_slide_img():
    # slide_filepath = 'D:/FLINC_dataset/slides/yang_psr_fib/tissues/23910-157_Sl040-C24-PSR.ndpi'
    # slide_filepath = 'D:/FLINC_dataset/slides/yang_he_stea/tissues/23910-157_Sl049-C32-HE.ndpi'
    # slide_filepath = 'D:/FLINC_dataset/slides/yang_cd45_u/tissues/23910-158_Sl251-C2-CD45.ndpi'
    # slide_filepath = 'D:/FLINC_dataset/slides/yang_p62_u/tissues/23910-158_Sl006-C5-P62.ndpi'
    # slide_filepath = '23910-157_Sl001-C2-HE.ndpi'
    slide_filepath = 'D:/FLINC_dataset/slides/yang_p62/tissues/23910-158_Sl001-C2-P62.ndpi'
    
#     slide_filepath = 'D:/IBD-Matthias/example_dx/slides/6574_20_l1 - 2022-08-02 13.26.48.ndpi'
    
    np_slide_img, _, _, _, _ = slide_to_scaled_np_image(slide_filepath)
    
    # np_filtered_img = apply_image_filters_he(np_slide_img)
    # np_filtered_img = apply_image_filters_psr(np_slide_img)
    # np_filtered_img = apply_image_filters_cd45(np_slide_img)
    np_filtered_img = apply_image_filters_p62(np_slide_img)
    
    pil_img = np_to_pil(np_filtered_img)
    print(pil_img)
    # pil_img.save('test_slide_filter.jpg')
    pil_img.show()
    
def test_transfer_ihc_dab():
    
    # jpeg_name = '23910-158_Sl149-C146-P62-tile_h137-w993_.jpeg'
    # jpeg_name = '23910-158_Sl004-C4-P62-tile_h69-w746_.jpeg'
    jpeg_name = '23910-158_Sl160-C158-P62-tile_h66-w770_.jpeg'
    pil_img = Image.open(os.path.join('D:\\FLINC_dataset\\slides\\yang_p62\\visualization\\heatmap\\hiera-tiledemo_Kmeans-ResNet18-encode_unsupervised2024-02-20\\selected\\cluster-3_1_1_0_1', jpeg_name) )
    dab_img = image_tools.pil_rgb_2_ihc_dab(pil_img)
    print(dab_img)
    
    dab_img.show()
    
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
    
    canvas_nxG, neig_nxG = nx_neb_graph_from_symadj(t_symm_test_nd, id_pos_dict)
    # print(canvas_nxG.edges(), (1, 0) not in canvas_nxG.edges())
    # print(canvas_nxG.get_edge_data(1, 2), canvas_nxG.get_edge_data(1, 2)['weight'])
    # print(canvas_nxG.nodes(), type(canvas_nxG.nodes()), 1 in canvas_nxG.nodes() )
    # print(canvas_nxG.degree(1, weight='weight'))
    #
    # node_list = list(canvas_nxG.nodes())
    # node_list.pop(1)
    # print(node_list, canvas_nxG.nodes())
        
        

cmaps = {}
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

def plot_color_gradients(category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh), dpi=100)
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.01, right=0.99)
    axs[0].set_title(f'{category} colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        # ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
        #         transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # Save colormap list for later.
    cmaps[category] = cmap_list
    
    plt.show()
    
    
def _test_vit_reuse():
    
    vit_reg_pt = ViT_Region_4_6(144, 16, 3)
    input_1 = torch.randn(3, 144, 144).unsqueeze(0).float()
    opt = optimizer_adam_basic(vit_reg_pt)
    
    output_1 = vit_reg_pt(input_1)
    print(output_1.shape)
    
    for name, param in vit_reg_pt.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    
    vit_pt_path = store_net('./', vit_reg_pt, 'test', opt)
    
    vit_reg_en = ViT_Region_4_6(9, 1, 256)
    for name, param in vit_reg_en.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    
    vit_reg_en, _ = check_reuse_net(vit_reg_en, vit_reg_pt, vit_pt_path)
    
    input_2 = torch.randn(256, 9, 9).unsqueeze(0).float()
    
    output_2 = vit_reg_en(input_2)
    print(output_2.shape)
    


def _test_plot_box():
    queue = [("group1", 0.5, "label1"), 
             ("group1", 0.6, "label2"), 
             ("group2", 0.7, "label1"), 
             ("group2", 0.8, "label2"), 
             ("group1", 0.5, "label1"), 
             ("group1", 0.4, "label2"), 
             ("group2", 0.5, "label1"), 
             ("group2", 0.6, "label2"), 
             ("group1", 0.7, "label1"), 
             ("group1", 0.5, "label2"), 
             ("group2", 0.3, "label1"), 
             ("group2", 0.5, "label2"), 
             ("group1", 0.6, "label1"), 
             ("group1", 0.8, "label2"), 
             ("group2", 0.4, "label1"), 
             ("group2", 0.5, "label2"), 
             ]
    
    df = pd.DataFrame(queue, columns=["group", "percentage", "label"])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="group", y="percentage", hue="label", data=df, palette="Set3")
    
    plt.title("Boxplot of Percentages by Group and Label")
    plt.show()
    
def _test_assign_label():
    values = [0.02, 0.07, 0.12, 0.16, 0.39, 0.78, 1.0]
    for v in values:
        print(assign_label(v))
        
''' - some functions for test on spatial transcriptomics data -'''

def _test_read_tiff_image():
    # slide_filepath = '/Volumes/Extreme SSD/st-data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_tissue_image.tif'
    # slide_filepath = '/Users/huyang/Desktop/Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma_image.tif'
    slide_filepath = 'E:\\PanoPath-Project\\st-data/CytAssist_11mm_FFPE_Human_Colorectal_Cancer_tissue_image.tif'
    # slide_filepath = '/Volumes/Extreme SSD/PanoPath-Project/st-data/CytAssist_11mm_FFPE_Human_Glioblastoma_tissue_image.tif'
    img, slide = slide_tools.original_slide_and_scaled_pil_image(slide_filepath)
    
    print(slide.dimensions)
    print(img)
    img.show()
    
def _test_read_hdf5_data():
    hdf5_filepath = '/Volumes/Extreme SSD/st-data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_filtered_feature_bc_matrix.h5'
    with h5py.File(hdf5_filepath, 'r') as h5_f:
        # print all data list in this H5 file
        h5_f.visit(print)
 
def get_matrix_from_h5(filename):
    CountMatrix = collections.namedtuple('CountMatrix', ['feature_ref', 'barcodes', 'matrix'])
    
    with tables.open_file(filename, 'r') as f:
        mat_group = f.get_node(f.root, 'matrix')
        barcodes = f.get_node(mat_group, 'barcodes').read()
        data = getattr(mat_group, 'data').read()
        indices = getattr(mat_group, 'indices').read()
        indptr = getattr(mat_group, 'indptr').read()
        shape = getattr(mat_group, 'shape').read()
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
         
        feature_ref = {}
        feature_group = f.get_node(mat_group, 'features')
        feature_ids = getattr(feature_group, 'id').read()
        feature_names = getattr(feature_group, 'name').read()
        feature_types = getattr(feature_group, 'feature_type').read()
        feature_ref['id'] = feature_ids
        feature_ref['name'] = feature_names
        feature_ref['feature_type'] = feature_types
        tag_keys = getattr(feature_group, '_all_tag_keys').read()
        for key in tag_keys:
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            feature_ref[key] = getattr(feature_group, key_str).read()
         
        return CountMatrix(feature_ref, barcodes, matrix)
    
def _test_parse_h5():
    # filtered_h5 = "/Volumes/Extreme SSD/st-data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_filtered_feature_bc_matrix.h5"
    # filtered_h5 = "/Volumes/Extreme SSD/PanoPath-Project/st-data/CytAssist_11mm_FFPE_Human_Colorectal_Cancer_filtered_feature_bc_matrix.h5"
    filtered_h5 = "E:\\PanoPath-Project\\st-data/CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_filtered_feature_bc_matrix.h5"
    filtered_matrix_h5 = get_matrix_from_h5(filtered_h5)
    
    matrix = filtered_matrix_h5.matrix
    # feature_ref = filtered_matrix_h5.feature_ref
    dense_matrix = matrix.toarray()
    print(dense_matrix.shape)
    # print(matrix)
    
    barcodes = filtered_matrix_h5.barcodes
    # print(type(barcodes))
    feature_names = filtered_matrix_h5.feature_ref['name']
    df = pd.DataFrame(matrix.toarray().T, index=barcodes, columns=feature_names)
    print(df)
    adata = anndata.AnnData(X=df.values, obs={'barcodes': df.index}, var={'genes': df.columns})
    
    print(adata.obs.all)
    print(adata.var.all)
    
def _count_st_position_coords():
    # df = pd.read_csv('/Volumes/Extreme SSD/st-data/tissue_positions.csv')
    df = pd.read_csv('/Volumes/Extreme SSD/st-data/Spatial-Projection.csv')
    nzero_X = df['X Coordinate'][df['X Coordinate'] > 0]
    nzero_Y = df['Y Coordinate'][df['Y Coordinate'] > 0]
    max_X, min_X = np.max(nzero_X), np.min(nzero_X)
    max_Y, min_Y = np.max(nzero_Y), np.min(nzero_Y)
    print(min_X, max_X )
    print(min_Y, max_Y )
    count_max_X = (df['X Coordinate'] > max_X - 100).sum()
    count_max_Y = (df['Y Coordinate'] > max_Y - 100).sum()
    print(count_max_X, count_max_Y)
    rows_with_max_Y = df[df['Y Coordinate'] > max_Y - 100]
    sorted_rows = rows_with_max_Y.sort_values(by='X Coordinate', ascending=True)
    print(sorted_rows)

        
    # print(feature_ref[b'genome'])
    # print(feature_ref['id'][:10])
    # print(feature_ref['name'][:10])
    
if __name__ == '__main__':
    test_filter_slide_img() # 1
    # test_transfer_ihc_dab()
    # test_vit_forward() # 2
    # test_networkx() # 3

#     test_numpy()
    # test_nx_graph()
    # test_neb_nx_graph()
    
    # plot_color_gradients('Qualitative', ['tab10'])
    
    # _test_vit_reuse()
    
    # _test_plot_box()
    # _test_assign_label()
    
    # _test_read_tiff_image()
    # _test_read_hdf5_data()
    # _test_parse_h5()
    # _count_st_position_coords()






