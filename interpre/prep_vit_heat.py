'''
@author: Yang Hu
'''

import os
import pickle

import cmapy
import cv2
from einops.einops import reduce, rearrange
import torch

from interpre.prep_tools import store_nd_dict_pkl
from models import functions, networks
from models.datasets import Simple_Tile_Dataset
from models.functions_clustering import refine_sp_cluster_homoneig, \
    load_clustering_pkg_from_pkl
from models.functions_vit_ext import access_att_maps_vit, \
    ext_att_maps_pick_layer, ext_cls_patch_att_maps, norm_exted_maps, \
    access_encodes_vit, gen_ctx_grid_tensor, extra_reg_assoc_key_tile
from models.networks import ViT_D6_H8, ViT_D9_H12, ViT_D3_H4_T, ViT_Region_4_6, \
    reload_net, check_reuse_net
import numpy as np
from support.env_flinc_cd45 import ENV_FLINC_CD45_REG_PT
from support.tools import normalization, Time
from wsi.process import recovery_tiles_list_from_pkl


# fixed discrete color value mapping (with 20 colors) for cv2 color palette
def col_pal_cv2_20(i_nd):
    return 5.0 + (255.0 / 20) * i_nd
# same with above with 10 colors
def col_pal_cv2_10(i_nd):
    return 10.0 + (255.0 / 10) * i_nd

def extra_cls_att_maps(tiles_attns_nd):
    '''
    extract the average attention map from numpy nd tensor
    which is for a tile list
    
    with normalization
    new update in Jan, 2023: by calling the functions in <functions_vit_ext.py>
        
    Return:
        a numpy ndarray
    
        Input shape: (tiles, layers, heads, queries, keys)
        Output shape: (tiles, (map_h, map_w))
            picked specific layer,
            averaged all heads,
            use queries[0] (cls) to pick values to keys[1:]
        
    Args:
        tiles_attns_nd: tiles' attention outcomes from torch tensor,
            transformed to numpy ndarray already
        layer_id: the layer which used for picking the attention maps
    '''    
    l_attns_nd = ext_att_maps_pick_layer(tiles_attns_nd, comb_heads='mean')
    cls_atts_maps = ext_cls_patch_att_maps(l_attns_nd)
    cls_atts_maps = norm_exted_maps(cls_atts_maps, 't q k')
    
    return cls_atts_maps

def extra_heads_att_maps(tiles_attns_nd):
    '''
    extract the attention maps for all heads from numpy nd tensor
    which is for a tile list
    
    with normalization
    new update in Jan, 2023: by calling the functions in <functions_vit_ext.py>
    
    Return:
        a numpy ndarray
    
        Input shape: (tiles, layers, heads, queries, keys)
        Output shape: (tiles, heads, (map_h, map_w))
            picked specific layer, keep all heads
            for each, use queries[0] (cls) to pick values to keys[1:]
                    (tiles, (map_h, map_w))
            picked specific layer, select the max idx from all heads
            give these idxs fix color values.
            
    Args:
        tiles_attns_nd: tiles' attention outcomes from torch tensor,
            transformed to numpy ndarray already
        layer_id: the layer which used for picking the attention maps
    '''
    (t, h, q, k) = tiles_attns_nd.shape # the layer pick has already been done, now is (t h q k)

    heads_att_maps = ext_cls_patch_att_maps(tiles_attns_nd)
    heads_att_maps = norm_exted_maps(heads_att_maps, 't h q k')
    
    flat_heads_att_maps = rearrange(heads_att_maps, 't h a b -> t h (a b)')
    maxi_l_att_nd = flat_heads_att_maps.argmax(axis=1)
    col_cv2_l_att_nd = col_pal_cv2_10(maxi_l_att_nd)
    max_att_maps = rearrange(col_cv2_l_att_nd, 't (a b) -> t a b', a=int(np.sqrt(k - 1)))
    
    return heads_att_maps, max_att_maps
    
def zoom_cv_maps(map_cv, z_times):
    return cv2.resize(map_cv, (map_cv.shape[0] * z_times, map_cv.shape[1] * z_times), interpolation=cv2.INTER_NEAREST)


''' ----------------- functions for tiles ----------------- '''
def vit_map_tiles(ENV_task, tiles, trained_vit, layer_id=-1, zoom=0, map_types=['cls', 'heads']):
    '''
    make the clsmap and headsmap packages for a list of tiles
    
    Args:
        ENV_task:
        tiles: 
        trained_vit: reloaded vit model on cuda
        layer_id: the id of layer from which to pick out the attention map
        map_types=['cls', 'heads']:
            'cls' -> average map on attention values
            'heads' -> all map for heads attention and it max index color (cv2) map
    '''
    tiles_attns_nd = access_att_maps_vit(tiles, trained_vit, 
                                         ENV_task.MINI_BATCH_TILE, 
                                         ENV_task.TILE_DATALOADER_WORKER,
                                         layer_id)
    # the layer pick has been done here to save memory
                
    cls_att_maps = extra_cls_att_maps(tiles_attns_nd) if 'cls' in map_types else None
    heads_att_maps, max_att_maps = extra_heads_att_maps(tiles_attns_nd) if 'heads' in map_types else (None, None)
    
    c_panel_cls = cmapy.cmap('plasma')
    c_panel_heads = cmapy.cmap('Oranges')
    c_panel_max = cmapy.cmap('tab10')
#     c_panel_max = cmapy.cmap('tab20')
    '''
    tiles_cls_map_list:
        list: [
            tuple: (att_map_cv, tile_obj, org_img_nd),
            ...
        ]
    '''
    tiles_cls_map_list, tiles_heads_map_list = [], []
    for i, tile in enumerate(tiles):
        org_img_nd = tile.get_np_tile()
        if cls_att_maps is not None:
            att_map_cv = cv2.applyColorMap(np.uint8(255 * cls_att_maps[i]), c_panel_cls)
            if zoom > 1:
                att_map_cv = zoom_cv_maps(att_map_cv, zoom)
            tiles_cls_map_list.append((att_map_cv, tile, org_img_nd))
        if heads_att_maps is not None and max_att_maps is not None:
            # multiple maps for heads and one map for max index head
            heads_att_maps_cv = [cv2.applyColorMap(np.uint8(255 * heads_att_maps[i, j]), c_panel_heads) for j in range(len(heads_att_maps[i]))]
            max_att_map_cv = cv2.applyColorMap(np.uint8(max_att_maps[i]), c_panel_max)
            if zoom > 1:
                heads_att_maps_cv = [zoom_cv_maps(h_att_maps_cv, zoom) for h_att_maps_cv in heads_att_maps_cv]
                max_att_map_cv = zoom_cv_maps(max_att_map_cv, zoom)
            tiles_heads_map_list.append((heads_att_maps_cv, max_att_map_cv, tile, org_img_nd))
        print('>> generated cls maps for: %d tiles, heads maps for: %d tiles' % (len(tiles_cls_map_list), len(tiles_heads_map_list)))
        
    return tiles_cls_map_list, tiles_heads_map_list


def reg_ass_key_tile(radius, key_encode_tuple, tiles_en_nd, tile_loc_dict, vit_region):
    '''
    '''
    region_ctx_nd, coordinates = gen_ctx_grid_tensor(radius, tiles_en_nd, tile_loc_dict, key_encode_tuple, print_info=True)
    vit_region.eval()
    
    ctx_tensor = torch.from_numpy(region_ctx_nd).to(torch.float)
    ctx_tensor = ctx_tensor.cuda()
    ctx_tensor = torch.unsqueeze(ctx_tensor, 0) # (h, w) -> (1, h, w)
    if vit_region.with_wrapper is False:
        vit_region.deploy_recorder()
    en_ctx, attn_ctx = vit_region.backbone(ctx_tensor)
    ass_vec = extra_reg_assoc_key_tile(attn_ctx, radius)
    ass_vec = ass_vec.cpu().detach().numpy()
    
    return vector_to_matrix(ass_vec, radius)

    
def vector_to_matrix(vector, r):
    attention_matrix = np.zeros((2*r + 1, 2*r + 1))
    center = r

    idx = 0
    for i in range(2*r + 1):
        for j in range(2*r + 1):
            if i == center and j == center:
                continue
            attention_matrix[i][j] = vector[idx]
            idx += 1

    return attention_matrix


''' ----------------- functions for slides ----------------- '''
def make_vit_att_map_slides(ENV_task, vit, vit_model_filepath,
                            sample_num=20, layer_id=-1, zoom=4, map_types=['cls', 'heads']):
    '''
    Args:
        ENV_task: 
        vit: vit model not on cuda
        vit_model_filepath: file path of trained vit model
        sample_num: number of sampled tiles from each slide
    '''
    _env_process_slide_tile_pkl_test_dir = ENV_task.TASK_TILE_PKL_TRAIN_DIR if ENV_task.DEBUG_MODE else ENV_task.TASK_TILE_PKL_TEST_DIR
    vit_model_filename = vit_model_filepath.split(os.sep)[-1]
    
    slides_tiles_pkl_dir = _env_process_slide_tile_pkl_test_dir
    sampled_tiles_list = []
    for slide_tiles_filename in os.listdir(slides_tiles_pkl_dir):
        slide_tiles_list = recovery_tiles_list_from_pkl(os.path.join(slides_tiles_pkl_dir, slide_tiles_filename))
        slide_sampled_tiles_list = slide_tiles_list[:sample_num] if len(slide_tiles_list) > sample_num else slide_tiles_list
        sampled_tiles_list.extend(slide_sampled_tiles_list)
    print('> sampled %d tiles from all slides for visualise the attention maps.' % len(sampled_tiles_list))
        
    vit, _ = networks.reload_net(vit, vit_model_filepath)
    vit = vit.cuda()
    print('> loaded trained vit network from: {}'.format(vit_model_filepath))
    
    tiles_cls_map_list, tiles_heads_map_list = vit_map_tiles(ENV_task, tiles=sampled_tiles_list,
                                                             trained_vit=vit, layer_id=layer_id,
                                                             zoom=zoom, map_types=map_types)
    
    slides_tiles_cls_map_dict, slides_tiles_heads_map_dict = {}, {}
    for tile_cls_map_tuple in tiles_cls_map_list:
        (_, tile, _) = tile_cls_map_tuple
        slide_id = tile.query_slideid()
        if slide_id not in slides_tiles_cls_map_dict.keys():
            slides_tiles_cls_map_dict[slide_id] = [tile_cls_map_tuple]
        else:
            slides_tiles_cls_map_dict[slide_id].append(tile_cls_map_tuple)
    if len(slides_tiles_cls_map_dict) > 0:
        clsmap_pkl_name = vit_model_filename.replace('checkpoint', 'clsmap').replace('.pth', '.pkl')
        store_nd_dict_pkl(ENV_task.HEATMAP_STORE_DIR, slides_tiles_cls_map_dict, clsmap_pkl_name)
        print('Done -> made and prepared clsmap for: %d slides, as: %s' % (len(slides_tiles_cls_map_dict), clsmap_pkl_name))
            
    for tile_heads_map_tuple in tiles_heads_map_list:
        (_, _, tile, _) = tile_heads_map_tuple
        slide_id = tile.query_slideid()
        if slide_id not in slides_tiles_heads_map_dict:
            slides_tiles_heads_map_dict[slide_id] = [tile_heads_map_tuple]
        else:
            slides_tiles_heads_map_dict[slide_id].append(tile_heads_map_tuple)
    if len(slides_tiles_heads_map_dict) > 0:
        headsmap_pkl_name = vit_model_filename.replace('checkpoint', 'headsmap').replace('.pth', '.pkl')
        store_nd_dict_pkl(ENV_task.HEATMAP_STORE_DIR, slides_tiles_heads_map_dict, headsmap_pkl_name)
        print('Done -> made and prepared headsmap for: %d slides, as: %s' % (len(slides_tiles_heads_map_dict), headsmap_pkl_name))
        

def reg_ass_sp_clst_homotiles_slides(ENV_task, clustering_res_pkg, tgt_lbl, iso_thd,
                                     vit_encoder, reg_encoder):
    '''
    create visualisation (graph based) to describe the regional associations for each tile and it's context
    for tiles in specific cluster
    '''
    radius = ENV_task.REG_RADIUS
    vit_encoder = vit_encoder.cuda()
    reg_encoder = reg_encoder.cuda()
    
    # load iso/gath tiles for specific clusters in slides
    slide_tgt_tiles_2_dict = refine_sp_cluster_homoneig(clustering_res_pkg, tgt_lbl, iso_thd, radius)
    # load the encode for all tiles
    slides_tiles_pkl_dir = ENV_task.TASK_TILE_PKL_TRAIN_DIR if ENV_task.DEBUG_MODE else ENV_task.TASK_TILE_PKL_TEST_DIR
    tile_list = []
    for slide_tiles_filename in os.listdir(slides_tiles_pkl_dir):
        slide_tiles_list = recovery_tiles_list_from_pkl(os.path.join(slides_tiles_pkl_dir, slide_tiles_filename))
        tile_list.extend(slide_tiles_list)
        
    ''' >>>> the vit_encoder here only support ViT <for the moment> '''
    tiles_en_nd, tile_loc_dict = access_encodes_vit(tile_list, vit_encoder,
                                                    ENV_task.MINI_BATCH_TILE, ENV_task.TILE_DATALOADER_WORKER)
    
    slide_tile_reg_ass_dict = {}
    for slide_id in slide_tgt_tiles_2_dict.keys():
        # make key_encode_tuple
        slide_sp_clst_tiles = slide_tgt_tiles_2_dict[slide_id]
        
        slide_tile_reg_ass_dict[slide_id] = []
        for tuple in slide_sp_clst_tiles:
            sp_homo_lbl = 'iso' if tuple[2] == 0 else 'gath'
            tile = tuple[3]
            tile_id = '{}-h{}-w{}'.format(slide_id, tile.h_id, tile.w_id)
            tile_en_idx = tile_loc_dict[tile_id][0]
            tile_encode = tiles_en_nd[tile_en_idx]
            key_encode_tuple = (tile_encode, tile, slide_id)
            
            ass_mat = reg_ass_key_tile(radius, key_encode_tuple, tiles_en_nd, tile_loc_dict, reg_encoder)
            slide_tile_reg_ass_dict[slide_id].append((ass_mat, tile, sp_homo_lbl) )
            
    sp_clst_reg_ass_pkl_name = 'sp_clst_homotiles_reg_ass-{}.pkl'.format(str(Time().date))
    store_nd_dict_pkl(ENV_task.HEATMAP_STORE_DIR, slide_tile_reg_ass_dict, sp_clst_reg_ass_pkl_name)
    print('Store specific cluster\'s region association matrices at {}.'.format(sp_clst_reg_ass_pkl_name))
        

''' --------------------- functions for calling --------------------- '''

def _run_vit_d6_h8_cls_map_slides(ENV_task, vit_model_filename):
    vit = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                    patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    make_vit_att_map_slides(ENV_task=ENV_task, vit=vit, 
                            vit_model_filepath=os.path.join(ENV_task.MODEL_FOLDER, vit_model_filename),
                            layer_id=-1, zoom=16, map_types=['cls'])

def _run_vit_d6_h8_heads_map_slides(ENV_task, vit_model_filename):
    vit = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                    patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    make_vit_att_map_slides(ENV_task=ENV_task, vit=vit, 
                            vit_model_filepath=os.path.join(ENV_task.MODEL_FOLDER, vit_model_filename),
                            layer_id=-1, zoom=16, map_types=['heads'])
    
def _run_vit_d6_h8_cls_heads_map_slides(ENV_task, vit_model_filename):
    vit = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                    patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    make_vit_att_map_slides(ENV_task=ENV_task, vit=vit, 
                            vit_model_filepath=os.path.join(ENV_task.MODEL_FOLDER, vit_model_filename),
                            layer_id=-1, zoom=16, map_types=['cls', 'heads'])
    
def _run_reg_ass_sp_clst_homotiles_slides(ENV_task, clustering_pkl_name, tgt_lbl, iso_thd,
                                          vit_pt_name, reg_vit_pt_name):
    clustering_res_pkg = load_clustering_pkg_from_pkl(ENV_task.MODEL_FOLDER, clustering_pkl_name)
    
    vit_encoder = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                            patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    reg_vit_encoder = ViT_Region_4_6(image_size=2 * ENV_task.REG_RADIUS + 1, patch_size=1,
                                     channels=ENV_task.TRANSFORMS_RESIZE)
    vit_encoder, _ = reload_net(vit_encoder, os.path.join(ENV_task.MODEL_FOLDER, vit_pt_name))
    
    vit_reg_load = ViT_Region_4_6(image_size=144, patch_size=int(144/ENV_FLINC_CD45_REG_PT.VIT_SHAPE), channels=3)
    reg_vit_encoder, _ = check_reuse_net(reg_vit_encoder, vit_reg_load,
                                         os.path.join(ENV_task.MODEL_FOLDER, reg_vit_pt_name))
    del vit_reg_load
    
    reg_ass_sp_clst_homotiles_slides(ENV_task, clustering_res_pkg, tgt_lbl, iso_thd, 
                                     vit_encoder, reg_vit_encoder)


if __name__ == '__main__':
    pass



