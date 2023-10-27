'''
Created on 23 Sept 2023

@author: super
'''

import gc
import math
import os
import pickle
import warnings

from PIL import Image
import PIL
import cmapy
import cv2
import torch
from torch.nn.functional import softmax

from interpre.prep_clst_vis import load_clst_res_slide_tile_label, \
    load_clst_res_label_tile_slide, col_pal_cv2_10
from interpre.prep_tools import store_nd_dict_pkl
from models import datasets, functions_attpool, functions_clustering
from models.functions_clustering import select_top_att_tiles, \
    load_clustering_pkg_from_pkl
from models.functions_lcsb import filter_singlesldie_top_attKtiles
from models.networks import BasicResNet18, GatedAttentionPool, AttentionPool, \
    reload_net
import numpy as np
from support.env_flinc_p62 import ENV_FLINC_P62_BALL_BI
from support.metadata import query_task_label_dict_fromcsv
from support.tools import normalization
from wsi import image_tools, slide_tools


# include the reusable function from prep_clst_vis, only from prep_dect_vis -> prep_clst_vis
def load_slide_tiles_att_score(slide_matrix_info_tuple, attpool_net):
    '''
    load the attention score for the tiles in each slide
    
    Args:
        slide_matrix_info_tuple: (slide_id, slide_tiles_len, slide_matrix_path)
            PS: !!! slide_matrix must be generated in original order with slide_tiles_list 
        attpool_net: the att-pool based classification
    '''
    _, slide_tiles_len, slide_matrix_path = slide_matrix_info_tuple
    slide_matrix = np.load(slide_matrix_path)
    
    attpool_net.eval()
    with torch.no_grad():
        X_e, bag_lens = torch.from_numpy(np.array([slide_matrix])), torch.from_numpy(np.array([slide_tiles_len]))
        X_e, bag_lens = X_e.cuda(), bag_lens.cuda()
        print(X_e.shape)
        
        if attpool_net.name == 'CLAM':
            att = attpool_net(X_e, bag_lens, attention_only=True)
        else:
            _, att, _ = attpool_net(X_e, bag_lens)
        att = att.detach().cpu().numpy()[-1]
    att_scores = normalization(att[:slide_tiles_len])
    
    return att_scores

def load_assim_res_tiles(model_store_dir, assimilate_pkl_name):
    '''
    Return:
        {slide_id: [tile ...]}
    '''
    assimilate_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, assimilate_pkl_name)
    
    slide_assim_tiles_dict = {}
    for i, assim_res_tuple in enumerate(assimilate_res_pkg):
        tile, slide_id = assim_res_tuple
        if slide_id not in slide_assim_tiles_dict.keys():
            slide_assim_tiles_dict[slide_id] = []
            slide_assim_tiles_dict[slide_id].append(tile)
        else:
            slide_assim_tiles_dict[slide_id].append(tile)
    
    return slide_assim_tiles_dict

def att_heatmap_single_scaled_slide(ENV_task, slide_info_tuple, attpool_net,
                                    for_train, load_attK=0, img_inter_methods='box',
                                    boost_rate=2.0, grad_col_map=True, cut_left=False):
    """
    Args:
        slide_info_tuple: must contains, 1. slide_id; 2. slide_tiles_len;
            3. os.path.join(slide_matrix_dir, slide_matrix_file)
        attpool_net: 
        for_train: for example, =False
        
        img_inter_methods: Image interpolation method, default: PIL.Image.HAMMING ('hamming')
        cmap: Color map of heatmap, default: cmapy.cmap('bwr') ('bwr')
    """
    
    def apply_mask(heat_med, white_mask):
        new_heat = np.uint32(np.float64(heat_med) + np.float64(white_mask))
        new_heat = np.uint8(np.minimum(new_heat, 255))
        return new_heat
    
    def keep_right_half(heat):
        height, width, _ = heat.shape
        start_col = width // 2
        return heat[:, start_col:]
    
#     label_dict = metadata.query_EMT_label_dict_fromcsv()
    slide_id, _, _ = slide_info_tuple
    slide_tiles_dict = datasets.load_slides_tileslist(ENV_task, for_train=for_train)
    slide_tiles_list = slide_tiles_dict[slide_id]
    
    slide_np, _ = slide_tiles_list[0].get_np_scaled_slide()
    H = round(slide_np.shape[0] * ENV_task.SCALE_FACTOR / ENV_task.TILE_H_SIZE)
    W = round(slide_np.shape[1] * ENV_task.SCALE_FACTOR / ENV_task.TILE_W_SIZE)
#     H = np.array([tile.h_id for tile in slide_tiles_list]).max()
#     W = np.array([tile.w_id for tile in slide_tiles_list]).max()
    heat_cam = np.zeros((H, W, 3), dtype=np.float64)
    heat_med = np.ones((H, W, 3), dtype=np.float64)
    white_mask = np.ones((H, W, 3), dtype=np.float64)
    # the heatmap original tensor with numpy format
    heat_np = np.zeros((H, W), dtype=np.float64)
    
    att_scores = load_slide_tiles_att_score(slide_info_tuple, attpool_net)
#     att_scores = att[:slide_tiles_len]
    for i, att_score in enumerate(att_scores):
        h = slide_tiles_list[i].h_id - 1 
        w = slide_tiles_list[i].w_id - 1
        if h >= H or w >= W or h < 0 or w < 0:
            warnings.warn('Out of range coordinates.')
            continue
        heat_cam[h, w] = att_score * boost_rate if att_score * boost_rate < 1.0 else 1.0 - 1e-6
        heat_med[h, w] = att_score * boost_rate if att_score * boost_rate < 1.0 else 1.0 - 1e-6
        white_mask[h, w] = 0.0
        heat_np[h, w] = att_score
        
    # set the image parameters
    if img_inter_methods == 'box':
        pil_img_type = PIL.Image.BOX
    elif img_inter_methods == 'bi':
        pil_img_type = PIL.Image.BICUBIC
    elif img_inter_methods == 'hamming':
        pil_img_type = PIL.Image.HAMMING
    else:
        pil_img_type = PIL.Image.HAMMING
        
#     c_panel_1 = cmapy.cmap('Spectral_r')
    c_panel_2 = cmapy.cmap('RdBu_r') if boost_rate == 1.0 else cmapy.cmap('bwr')
    c_panel_g = cmapy.cmap('RdYlBu_r')
        
    heat_cam = image_tools.np_to_pil(heat_cam).resize((slide_np.shape[1], slide_np.shape[0]), pil_img_type)
    heat_med = image_tools.np_to_pil(heat_med).resize((slide_np.shape[1], slide_np.shape[0]), pil_img_type)

    white_mask = image_tools.np_to_pil(white_mask).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    heat_cam = np.float64(heat_cam) / 255
    heat_med = np.float64(heat_med) / 255
    heat_med_style = cv2.applyColorMap(np.uint8(255 * heat_med), c_panel_2)
    heat_med_grad = cv2.applyColorMap(np.uint8(255 * heat_cam), c_panel_g) if grad_col_map else None
    org_image, _ = slide_tools.original_slide_and_scaled_pil_image(slide_tiles_list[0].original_slide_filepath,
                                                                   ENV_task.SCALE_FACTOR, print_opening=False)
    org_np_img = image_tools.pil_to_np_rgb(org_image)
    heat_med_style = apply_mask(heat_med_style, white_mask)
    print('generate attention score heatmap (image type: {}) for slide: {}'.format(img_inter_methods, slide_id))
    
    attK_tiles_list = []
    if load_attK > 0:
        tiles_all_list, _, slide_tileidxs_dict = datasets.load_richtileslist_fromfile(ENV_task, for_train)
        attK_tiles_list = filter_singlesldie_top_attKtiles(tiles_all_list=tiles_all_list,
                                                          slide_tileidxs_list=slide_tileidxs_dict[slide_id],
                                                          slide_attscores=att_scores, K=load_attK)
        print('load att {} tile list of this slide.'.format(load_attK))
        
    if cut_left:
        print('--- cut left, only keep right part!')
        org_np_img = keep_right_half(org_np_img)
        heat_med_style = keep_right_half(heat_med_style)
        heat_med_grad = keep_right_half(heat_med_grad)
            
    return org_np_img, heat_np, heat_med_style, heat_med_grad, attK_tiles_list


def topK_att_heatmap_single_scaled_slide(ENV_task, k_slide_tiles_list, k_attscores,
                                         boost_rate=2.0, cut_left=False):
    """
    """
    
    def apply_mask(heat, white_mask):
        new_heat = np.uint32(np.float64(heat) + np.float64(white_mask))
        new_heat = np.uint8(np.minimum(new_heat, 255))
        return new_heat
    
    def keep_right_half(heat):
        height, width, _ = heat.shape
        start_col = width // 2
        return heat[:, start_col:]
    
    slide_np, _ = k_slide_tiles_list[0].get_np_scaled_slide()
    H = round(slide_np.shape[0] * ENV_task.SCALE_FACTOR / ENV_task.TILE_H_SIZE)
    W = round(slide_np.shape[1] * ENV_task.SCALE_FACTOR / ENV_task.TILE_W_SIZE)
    
    heat_hard = np.zeros((H, W, 3), dtype=np.float64)
    heat_soft = np.ones((H, W, 3), dtype=np.float64)
    white_mask = np.ones((H, W, 3), dtype=np.float64)
    # the heatmap original tensor with numpy format
    heat_np = np.zeros((H, W), dtype=np.float64)
    
    for i, att_score in enumerate(k_attscores):
        h = k_slide_tiles_list[i].h_id - 1 
        w = k_slide_tiles_list[i].w_id - 1
        if h >= H or w >= W or h < 0 or w < 0:
            warnings.warn('Out of range coordinates.')
            continue
        heat_hard[h, w] = 1.0 - 1e-6
        heat_soft[h, w] = att_score * boost_rate if att_score * boost_rate < 1.0 else 1.0 - 1e-6
        white_mask[h, w] = 0.0
        heat_np[h, w] = att_score
    print('final highlighted tiles: ', len(k_attscores) )
        
    pil_img_type = PIL.Image.BOX
    c_panel_1 = cmapy.cmap('bwr')
    
    heat_hard = image_tools.np_to_pil(heat_hard).resize((slide_np.shape[1], slide_np.shape[0]), pil_img_type)
    heat_soft = image_tools.np_to_pil(heat_soft).resize((slide_np.shape[1], slide_np.shape[0]), pil_img_type)

    white_mask = image_tools.np_to_pil(white_mask).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    heat_hard = np.float64(heat_hard) / 255
    heat_soft = np.float64(heat_soft) / 255
    heat_hard_cv2 = cv2.applyColorMap(np.uint8(255 * heat_hard), c_panel_1)
    heat_soft_cv2 = cv2.applyColorMap(np.uint8(255 * heat_soft), c_panel_1)
    org_image, _ = slide_tools.original_slide_and_scaled_pil_image(k_slide_tiles_list[0].original_slide_filepath,
                                                                   ENV_task.SCALE_FACTOR, print_opening=False)
    org_np_img = image_tools.pil_to_np_rgb(org_image)
    heat_hard_cv2 = apply_mask(heat_hard_cv2, white_mask)
    heat_soft_cv2 = apply_mask(heat_soft_cv2, white_mask)
    print('generate attention score heatmap (both hard and soft) for slide: {}'.format(k_slide_tiles_list[0].query_slideid()) )
    
    if cut_left:
        print('--- cut left, only keep right part!')
        org_np_img = keep_right_half(org_np_img)
        heat_hard_cv2 = keep_right_half(heat_hard_cv2)
        heat_soft_cv2 = keep_right_half(heat_soft_cv2)
        
    return org_np_img, heat_np, heat_hard_cv2, heat_soft_cv2


def gen_single_slide_sensi_clst_spatial(ENV_task, slide_tile_clst_tuples, slide_assim_tiles_list, slide_id, labels_picked, cut_left=False):
    '''
    generate the clusters spatial map on single slide
    
    Return:
        org_np_img: nd_array of scaled original image of slide
        heat_s_clst_col: clusters spatial map for the slide
    '''
    
    def apply_mask(heat, white_mask):
        new_heat = np.uint32(np.float32(heat) + np.float32(white_mask))
        new_heat = np.uint8(np.minimum(new_heat, 255))
        return new_heat
    
    def keep_right_half(heat):
        height, width, _ = heat.shape
        start_col = width // 2
        return heat[:, start_col:]
    
    slide_np, _ = slide_tile_clst_tuples[0][0].get_np_scaled_slide()
    H = round(slide_np.shape[0] * ENV_task.SCALE_FACTOR / ENV_task.TILE_H_SIZE)
    W = round(slide_np.shape[1] * ENV_task.SCALE_FACTOR / ENV_task.TILE_W_SIZE)
    heat_s_clst = np.ones((H, W, 3), dtype=np.float64)
    white_mask = np.ones((H, W, 3), dtype=np.float64)
    
    for tile, label in slide_tile_clst_tuples:
        if label in labels_picked:
            h = tile.h_id - 1 
            w = tile.w_id - 1
            if h >= H or w >= W or h < 0 or w < 0:
                warnings.warn('Out of range coordinates.')
                continue
            heat_s_clst[h, w] = 1.0 - 1e-3
            white_mask[h, w] = 0.0
    print('checked %d tiles for sensitive clusters' % len(slide_tile_clst_tuples) )
    for tile in slide_assim_tiles_list:
        h = tile.h_id - 1 
        w = tile.w_id - 1
        if h >= H or w >= W or h < 0 or w < 0:
            warnings.warn('Out of range coordinates.')
            continue
        heat_s_clst[h, w] = 0.8
        white_mask[h, w] = 0.0
    print('checked %d tiles assimilated from sensitive clusters.' % len(slide_assim_tiles_list) )
    
    c_panel = cmapy.cmap('coolwarm')
    heat_clst_cv2_col = col_pal_cv2_10(heat_s_clst).astype("uint8")
    heat_s_clst = Image.fromarray(heat_clst_cv2_col).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    white_mask = image_tools.np_to_pil(white_mask).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    heat_s_clst_col = cv2.applyColorMap(np.uint8(heat_s_clst), c_panel)
    heat_s_clst_col = apply_mask(heat_s_clst_col, white_mask)
    
    print('generate cluster-{} and the assimilated tiles\' spatial map for slide: {}'.format(str(labels_picked), slide_id))
    if cut_left:
        print('--- cut left, only keep right part!')
        heat_s_clst_col = keep_right_half(heat_s_clst_col)
    return heat_s_clst_col


def make_topK_attention_heatmap_package(ENV_task, agt_model_filenames, label_dict,
                                        cut_left=True, tile_encoder=None, 
                                        K_ratio=0.3, att_thd=0.25, boost_rate=2.0, fills=[3], pkg_range=None):
    """
    make the top K attention pool visualisation (only 1 round, no milestones), 
    only highlight the picked tiles with the highest attention scores.
        include: 
        1. the attention heatmap for attpool
    """
    
    ''' prepare some parames '''
    _env_heatmap_store_dir = ENV_task.HEATMAP_STORE_DIR
    _env_model_store_dir = ENV_task.MODEL_FOLDER
    
    for_train = False if not ENV_task.DEBUG_MODE else True
    
    if tile_encoder is None:
        tile_encoder = BasicResNet18(output_dim=2)
        tile_encoder = tile_encoder.cuda()
    else:
        tile_encoder = tile_encoder.cuda()
    
    _, slide_k_tiles_atts_dict = select_top_att_tiles(ENV_task, tile_encoder, 
                                                      agt_model_filenames, label_dict,
                                                      K_ratio, att_thd, fills=fills, pkg_range=pkg_range)
    
    slide_topK_att_heatmap_dict = {}
    for slide_id in slide_k_tiles_atts_dict.keys():
        k_slide_tiles_list, k_attscores = slide_k_tiles_atts_dict[slide_id]
        org_image, heat_np, heat_hard_cv2, heat_soft_cv2 = topK_att_heatmap_single_scaled_slide(ENV_task,
                                                                                                k_slide_tiles_list, 
                                                                                                k_attscores,
                                                                                                boost_rate=boost_rate,
                                                                                                cut_left=cut_left)
        slide_topK_att_heatmap_dict[slide_id] = {'original': org_image,
                                                 'heat_np': heat_np,
                                                 'heat_hard_cv2': heat_hard_cv2,
                                                 'heat_soft_cv2': heat_soft_cv2
                                                 }
        print('added topK attention map numpy info set for slide: {}'.format(slide_id))
        
    att_heatmap_pkl_name = agt_model_filenames[0].replace('checkpoint', 'topK_map').replace('.pth', '.pkl')
    store_nd_dict_pkl(_env_heatmap_store_dir,
                      slide_topK_att_heatmap_dict, att_heatmap_pkl_name)
    print('Store topK attention map numpy package as: {}'.format(att_heatmap_pkl_name))
    

def make_spatial_sensi_clusters_assim_on_slides(ENV_task, clustering_pkl_name, assimilate_pkl_name, sp_clsts, cut_left):
    '''
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    
    slide_tile_clst_dict = load_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name)
    slide_id_list = list(datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE).keys())
    print('load the slide_ids we have, on the running client (PC or servers), got %d slides...' % len(slide_id_list))
    if assimilate_pkl_name is not None:
        slide_assim_tiles_dict = load_assim_res_tiles(model_store_dir, assimilate_pkl_name)
        print('load the assimilate slide-tiles dictionary, for %d slides...' % len(slide_assim_tiles_dict))
    else:
        slide_assim_tiles_dict = None
        print('assimilate tiles are not available!')
    
    slide_clst_s_spatmap_dict = {}
    for slide_id in slide_id_list:
        tile_clst_tuples = slide_tile_clst_dict[slide_id]
        assim_tiles_list = slide_assim_tiles_dict[slide_id] if slide_assim_tiles_dict is not None else []
        
        '''
        2 key inputs:
            1. tile_clst_tuples: which contains all clusters in this slide, so need to filter further
            2. assim_tiles_list: which only contain the assimilating results (tiles) for the picked sensitive clusters
        '''
        heat_s_clst_col = gen_single_slide_sensi_clst_spatial(ENV_task, tile_clst_tuples, assim_tiles_list, slide_id, labels_picked=sp_clsts, cut_left=cut_left)
        slide_clst_s_spatmap_dict[slide_id] = heat_s_clst_col
            
    clst_s_spatmap_pkl_name = clustering_pkl_name.replace('clst-res', 'clst-{}-spat'.format(str(sp_clsts)))
    store_nd_dict_pkl(heat_store_dir, slide_clst_s_spatmap_dict, clst_s_spatmap_pkl_name)
    print('Store slides\' sensitive clusters (and assimilated) spatial maps numpy package as: {}'.format(clst_s_spatmap_pkl_name))
    
    
def cnt_tis_pct_sensi_clsts_assim_on_slides(ENV_task, clustering_pkl_name, sensi_clsts, assimilate_pkl_name):
    '''
    visualisation of tissue percentage on 
        1. sensitive clusters; 2. assimilate tiles (based on sensitive clusters); 3. both
        
    Args:
        ENV_task:
        clustering_pkl_name: the file name of clustering results which includes all clusters
        sensi_clsts:
        assimilate_pkl_name: the file name of assimilating results which only based on sensitive clusters
    '''
    
    #TODO:


''' ----------------------------------------------------------------------------------------------------------- '''

def _run_make_attention_heatmap_package(ENV_task, model_filename, tile_encoder=None,
                                        batch_size=16, nb_workers=4, att_top_K=0,
                                        org_tile_img=False, boost_rate=2.0,
                                        part_pick=None, image_type='box', grad_col_map=True):
    """
    make the normal attention pool visualisation (only 1 round, no milestones)
        include: 
        1. the attention heatmap for attpool
    """
    
    ''' prepare some parames '''
    _env_heatmap_store_dir = ENV_task.HEATMAP_STORE_DIR
    _env_model_store_dir = ENV_task.MODEL_STORE_DIR
    
    for_train = False if not ENV_task.DEBUG_MODE else True
    
    if tile_encoder is None:
        tile_encoder = BasicResNet18(output_dim=2)
        tile_encoder = tile_encoder.cuda()
    else:
        tile_encoder = tile_encoder.cuda()
    test_slidemat_file_sets = functions_attpool.check_load_slide_matrix_files(ENV_task=ENV_task,
                                                                              batch_size_ontiles=batch_size,
                                                                              tile_loader_num_workers=nb_workers,
                                                                              encoder_net=tile_encoder.backbone,
                                                                              for_train=for_train,
                                                                              force_refresh=False, #True,
                                                                              print_info=True)
    embedding_dim = np.load(test_slidemat_file_sets[0][2]).shape[-1]
    model_filepath = os.path.join(_env_model_store_dir, model_filename)
    
    if model_filename.find('GatedAttPool') != -1:
        attpool = GatedAttentionPool(embedding_dim=embedding_dim, output_dim=2)
    else:
        attpool = AttentionPool(embedding_dim=embedding_dim, output_dim=2)   
    attpool, _ = reload_net(attpool, model_filepath)
    attpool = attpool.cuda()
    
    # for save storage space, can only select a part of test sample for visualization
    if part_pick == None or part_pick > len(test_slidemat_file_sets):
        test_slidemat_file_sets = test_slidemat_file_sets
    else:
        test_slidemat_file_sets = test_slidemat_file_sets[:part_pick]
        # test_slidemat_file_sets = test_slidemat_file_sets[50:50+part_pick]
    
    slide_att_heatmap_dict = {}
    for i, slidemat_info_tuple in enumerate(test_slidemat_file_sets):
        slide_id = slidemat_info_tuple[0]
        print(attpool.name)
        # discard heat_med_grad
        org_image, heat_np, heat_med_style, _, attK_tiles_list = att_heatmap_single_scaled_slide(ENV_task=ENV_task,
                                                                                                 slide_info_tuple=slidemat_info_tuple,
                                                                                                 attpool_net=attpool,
                                                                                                 for_train=for_train,
                                                                                                 load_attK=att_top_K,
                                                                                                 img_inter_methods=image_type,
                                                                                                 boost_rate=boost_rate,
                                                                                                 grad_col_map=grad_col_map)
        slide_att_heatmap_dict[slide_id] = {'original': org_image,
                                            'heat_np': heat_np,
                                            'heat_med_style': heat_med_style,
                                            'heat_med_grad': None # heat_med_grad,
                                            }
        print('added attention_heatmap numpy info set for slide: {}'.format(slide_id))
        
    att_heatmap_pkl_name = model_filename.replace('checkpoint', 'heatmap').replace('.pth', '.pkl')
    store_nd_dict_pkl(_env_heatmap_store_dir,
                      slide_att_heatmap_dict, att_heatmap_pkl_name)
    print('Store attention_heatmap numpy package as: {}'.format(att_heatmap_pkl_name))
    
    
def _run_make_topK_attention_heatmap_resnet_P62(ENV_task, agt_model_filenames, cut_left,
                                                K_ratio=0.3, att_thd=0.3, boost_rate=2.0, fills=[3], pkg_range=[0, 50]):
    '''
    load the label_dict then call the <make_topK_attention_heatmap_package>
    '''
    ENV_annotation = ENV_FLINC_P62_BALL_BI
    label_dict = query_task_label_dict_fromcsv(ENV_annotation)

    tile_encoder = BasicResNet18(output_dim=2)
    make_topK_attention_heatmap_package(ENV_task, agt_model_filenames, label_dict,
                                        cut_left, tile_encoder,
                                        K_ratio, att_thd, boost_rate, fills, pkg_range)
    
def _run_make_spatial_sensi_clusters_assims(ENV_task, clustering_pkl_name, assimilate_pkl_name, sp_clsts, cut_left=True):
    '''
    make the spatial map for sensitive clusters (and their assimilated tiles)
    '''
    make_spatial_sensi_clusters_assim_on_slides(ENV_task, clustering_pkl_name, assimilate_pkl_name, sp_clsts, cut_left)


if __name__ == '__main__':
    pass
