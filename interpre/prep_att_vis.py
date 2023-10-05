'''
Created on 23 Sept 2023

@author: super
'''

import gc
import math
import os
import pickle
import warnings

import PIL
import cmapy
import cv2
import torch
from torch.nn.functional import softmax

from interpre.prep_tools import store_nd_dict_pkl
from models import datasets, functions_attpool, functions_clustering
from models.functions_clustering import select_top_att_tiles
from models.functions_lcsb import filter_singlesldie_top_attKtiles
from models.networks import BasicResNet18, GatedAttentionPool, AttentionPool, \
    reload_net
import numpy as np
from support.env_flinc_p62 import ENV_FLINC_P62_BALL_BI
from support.metadata import query_task_label_dict_fromcsv
from support.tools import normalization
from wsi import image_tools, slide_tools


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

def att_heatmap_single_scaled_slide(ENV_task, slide_info_tuple, attpool_net,
                                    for_train, load_attK=0, img_inter_methods='box',
                                    boost_rate=2.0, grad_col_map=True):
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
            
    return org_np_img, heat_np, heat_med_style, heat_med_grad, attK_tiles_list


def topK_att_heatmap_single_scaled_slide(ENV_task, k_slide_tiles_list, k_attscores,
                                         boost_rate=2.0):
    """
    
    """
    
    def apply_mask(heat_soft, white_mask):
        new_heat = np.uint32(np.float64(heat_soft) + np.float64(white_mask))
        new_heat = np.uint8(np.minimum(new_heat, 255))
        return new_heat
    
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
    print('highlighted tiles: ', len(k_attscores) )
        
    # check the surrounding states
    nb_fill = 0
    fill_heat_np = np.zeros((H, W), dtype=np.float64)
    for i_h in range(H):
        for i_w in range(W):
            stat, avg_surd = functions_clustering.check_surrounding(heat_np, i_h, i_w)
            if stat:
                nb_fill += 1
                fill_heat_np[i_h, i_w] = avg_surd
    for i_h in range(H):
        for i_w in range(W):
            if fill_heat_np[i_h, i_w] > 0.0:
                heat_hard[i_h, i_w] = 1.0 - 1e-6
                heat_soft[i_h, i_w] = fill_heat_np[i_h, i_w] * boost_rate if fill_heat_np[i_h, i_w] * boost_rate < 1.0 else 1.0 - 1e-6
                white_mask[i_h, i_w] = 0.0
                heat_np[i_h, i_w] = fill_heat_np[i_h, i_w]
    print('fill surrounding tiles: ', nb_fill)
    
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
    
    return org_np_img, heat_np, heat_hard_cv2, heat_soft_cv2


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


def make_topK_attention_heatmap_package(ENV_task, agt_model_filenames, label_dict,
                                        tile_encoder=None, 
                                        K_ratio=0.3, att_thd=0.25, boost_rate=2.0):
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
                                                      K_ratio, att_thd)
    
    slide_topK_att_heatmap_dict = {}
    for slide_id in slide_k_tiles_atts_dict.keys():
        k_slide_tiles_list, k_attscores = slide_k_tiles_atts_dict[slide_id]
        org_image, heat_np, heat_hard_cv2, heat_soft_cv2 = topK_att_heatmap_single_scaled_slide(ENV_task,
                                                                                                k_slide_tiles_list, 
                                                                                                k_attscores,
                                                                                                boost_rate=boost_rate)
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
    
    
def _run_make_topK_attention_heatmap_resnet_P62(ENV_task, agt_model_filenames,
                                                K_ratio=0.3, att_thd=0.25, boost_rate=2.0):
    '''
    load the label_dict then call the <make_topK_attention_heatmap_package>
    '''
    ENV_annotation = ENV_FLINC_P62_BALL_BI
    label_dict = query_task_label_dict_fromcsv(ENV_annotation)

    tile_encoder = BasicResNet18(output_dim=2)
    make_topK_attention_heatmap_package(ENV_task, agt_model_filenames, label_dict,
                                        tile_encoder,
                                        K_ratio, att_thd, boost_rate)


if __name__ == '__main__':
    pass
