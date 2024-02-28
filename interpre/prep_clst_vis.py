'''
Created on 17 Nov 2022

@author: Yang Hu
'''
import csv
import os
import warnings

from PIL import Image
import PIL
import cmapy
import cv2
from tqdm import tqdm

from interpre.prep_tools import safe_random_sample, tSNE_transform, \
    store_nd_dict_pkl, load_vis_pkg_from_pkl
from models import datasets
from models.functions_clustering import load_clustering_pkg_from_pkl, \
    refine_sp_cluster_homoneig, refine_sp_cluster_levels
import numpy as np
from support.files import parse_caseid_from_slideid
from support.metadata import query_task_label_dict_fromcsv
from support.tools import Time
from wsi import image_tools, slide_tools


# fixed discrete color value mapping (with 20 colors) for cv2 color palette
def col_pal_cv2_20(i_nd):
    return 5.0 + (255.0 / 20) * i_nd
# same with above with 10 colors
def col_pal_cv2_10(i_nd):
    return 10.0 + (255.0 / 10) * i_nd

def pick_centrest_encodes(encodes_list, nb_pick):
    '''
    '''
    encodes_array = np.array(encodes_list)
    feature_center = np.mean(encodes_array, axis=0)
    
    distances = np.linalg.norm(encodes_array - feature_center, axis=1)
    sorted_indices = np.argsort(distances)
    closest_indices = sorted_indices[:nb_pick]
    picked_encodes = encodes_array[closest_indices]
    return picked_encodes

def load_clst_res_encode_label(model_store_dir, clustering_pkl_name, r_pick_from_clst=None):
    '''
    Return:
        {clst_label: [encodes]}, [(clst_label, encode)]
        
    can sampling some points for each clst_label, not all
    '''
    clustering_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name)
    
    clst_encode_dict, clst_encode_list = {}, []
    for i, clst_res_tuple in enumerate(clustering_res_pkg):
        label, encode, _, _ = clst_res_tuple
        if label not in clst_encode_dict.keys():
            clst_encode_dict[label] = []
            clst_encode_dict[label].append(encode)
        else:
            clst_encode_dict[label].append(encode)
        
    pick_clst_encode_dict = {}    
    if r_pick_from_clst is not None:
        for label in clst_encode_dict.keys():
            label_encodes_list = clst_encode_dict[label]
            r_pick_from_clst = 0.1 if r_pick_from_clst is None else r_pick_from_clst
            nb_pick = int(len(label_encodes_list) * r_pick_from_clst)
            nb_half = int(len(label_encodes_list) * 0.5)
            half_pick = pick_centrest_encodes(label_encodes_list, nb_half)
            # print(type(half_pick))
            closed_rand_pick = safe_random_sample(list(half_pick), nb_pick)
            pick_clst_encode_dict[label] = closed_rand_pick
            print(f'pick {nb_pick} embeds from cluster: {label}...')
    else:
        pick_clst_encode_dict = clst_encode_dict
            
    for label, encode_list in pick_clst_encode_dict.items():
        for encode in encode_list:
            clst_encode_list.append((label, encode))
            
    return pick_clst_encode_dict, clst_encode_list
    
    
def load_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name):
    '''
    Return:
        {slide_id: [(tile, clst_label) ...]}
    '''
    clustering_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name)
    
    slide_tile_clst_dict = {}
    for i, clst_res_tuple in enumerate(clustering_res_pkg):
        label, _, tile, slide_id = clst_res_tuple
        if slide_id not in slide_tile_clst_dict.keys():
            slide_tile_clst_dict[slide_id] = []
            slide_tile_clst_dict[slide_id].append((tile, label))
        else:
            slide_tile_clst_dict[slide_id].append((tile, label))
    
    return slide_tile_clst_dict

def load_select_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name,
                                          select_labels):
    '''
    @deprecated: not used now
    
    Same with above function, just exclude the not-selected labels
    only left selected cluster labels
    
    Return:
        {slide_id: [(tile, clst_label) ...]}
    '''
    clustering_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name)
    
    select_s_tile_clst_dict = {}
    for i, clst_res_tuple in enumerate(clustering_res_pkg):
        label, _, tile, slide_id = clst_res_tuple
        if label not in select_labels:
            continue
        if slide_id not in select_s_tile_clst_dict.keys():
            select_s_tile_clst_dict[slide_id] = []
            select_s_tile_clst_dict[slide_id].append((tile, label))
        else:
            select_s_tile_clst_dict[slide_id].append((tile, label))
    
    return select_s_tile_clst_dict

def pick_clusters_by_prefix(ENV_task, clustering_pkl_name, cluster_groups):
    '''
    from clustering results pick the selected cluster names
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    clustering_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name)
    
    selected_labels = []
    # enumerate all labels in clustering results
    for label, _, _, _ in clustering_res_pkg:
        if any(label.startswith(prefix) for prefix in cluster_groups):
            selected_labels.append(label)
            
    # filter the repeat labels
    selected_labels = list(set(selected_labels))

    return selected_labels

def load_ref_group_slide_tile_sp_clst(model_store_dir, clustering_pkl_name, sp_clst, iso_thd=0.25, radius=5):
    '''
    Return:
        {slide_id: [(tile, group_label)]}
    '''
    slide_id_list = []
    clustering_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name)
    for i, clst_res_tuple in enumerate(clustering_res_pkg):
        _, _, _, slide_id = clst_res_tuple
        slide_id_list.append(slide_id)
    
    slide_tgt_tiles_dict = refine_sp_cluster_homoneig(clustering_res_pkg, sp_clst, iso_thd, radius)
    
    slide_tile_ref_gp_dict = {}
    for slide_id in slide_id_list:
        if slide_id not in slide_tgt_tiles_dict.keys():
            continue
        ref_tiles_tuples = slide_tgt_tiles_dict[slide_id]
        slide_tile_ref_gp_dict[slide_id] = []
        for tile_tuple in ref_tiles_tuples:
            slide_tile_ref_gp_dict[slide_id].append((tile_tuple[3], tile_tuple[2]))
            
    slide_tis_nb_dict = {}
    for i, clst_res_tuple in enumerate(clustering_res_pkg):
        _, _, _, slide_id = clst_res_tuple
        if slide_id not in slide_tis_nb_dict.keys():
            slide_tis_nb_dict[slide_id] = 1
        else:
            slide_tis_nb_dict[slide_id] += 1
    
    return slide_tile_ref_gp_dict, slide_tis_nb_dict

def load_clst_res_label_tile_slide(model_store_dir, clustering_pkl_name):
    '''
    Return:
        {clst_label: [(tile, slide_id)]}
    '''
    clustering_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name)
    
    clst_tile_slideid_dict = {}
    for i, clst_res_tuple in enumerate(clustering_res_pkg):
        label, _, tile, slide_id = clst_res_tuple
        if label not in clst_tile_slideid_dict.keys():
            clst_tile_slideid_dict[label] = []
            clst_tile_slideid_dict[label].append((tile, slide_id))
        else:
            clst_tile_slideid_dict[label].append((tile, slide_id))
    
    return clst_tile_slideid_dict
    

def clst_encode_redu_tsne(clst_encode_tuples):
    '''
    using tSNE dim-reduction
    
    Return:
        {label: nd_array [embeds] dim-redu encodes}
    '''
    encodes, labels = [], []
    for label, encode in clst_encode_tuples:
        encodes.append(encode)
        labels.append(label)
        
    print('running t-SNE algorithm...')
    time = Time()
    embeds = tSNE_transform(encodes)
    print('finished with time: {}'.format(str(time.elapsed())[:-5]))
    
    clst_redu_en_dict = {}
    for i, embed in enumerate(embeds):
        if labels[i] not in clst_redu_en_dict.keys():
            clst_redu_en_dict[labels[i]] = []
            clst_redu_en_dict[labels[i]].append(embed)
        else:
            clst_redu_en_dict[labels[i]].append(embed)
    for label in clst_redu_en_dict.keys():
        label_embed_list = clst_redu_en_dict[label]
        clst_redu_en_dict[label] = np.array(label_embed_list)
        
    return clst_redu_en_dict

def make_clsuters_space_maps(ENV_task, clustering_pkl_name, r_picked=None):
    '''
    reduce the clusters points to a feature space
    store the clst - dim_redu space in pkl
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    _, clst_encode_list = load_clst_res_encode_label(model_store_dir, clustering_pkl_name, r_picked)
    clst_redu_en_dict = clst_encode_redu_tsne(clst_encode_tuples=clst_encode_list)
    clst_tsne_pkl_name = 'tsne_{}_{}'.format('all' if r_picked is None else str(r_picked), clustering_pkl_name)
    store_nd_dict_pkl(stat_store_dir, clst_redu_en_dict, clst_tsne_pkl_name)
    print('done the clusters dim-reduction and store as: ', clst_tsne_pkl_name)
    

def gen_single_slide_clst_spatial(ENV_task, slide_tile_clst_tuples, slide_id, cut_left=False):
    '''
    generate the clusters spatial map on single slide
    
    Return:
        org_np_img: nd_array of scaled original image of slide
        heat_clst_col: clusters spatial map for the slide
    '''
    
    def apply_mask(heat, white_mask):
        new_heat = np.uint32(np.float32(heat) + np.float32(white_mask))
        new_heat = np.uint8(np.minimum(new_heat, 255))
        return new_heat
    
    def keep_right_half(img):
        height, width, _ = img.shape
        start_col = width // 2
        return img[:, start_col:]
    
    slide_np, _ = slide_tile_clst_tuples[0][0].get_np_scaled_slide()
    H = round(slide_np.shape[0] * ENV_task.SCALE_FACTOR / ENV_task.TILE_H_SIZE)
    W = round(slide_np.shape[1] * ENV_task.SCALE_FACTOR / ENV_task.TILE_W_SIZE)
    heat_clst = np.ones((H, W, 3), dtype=np.float64)
    white_mask = np.ones((H, W, 3), dtype=np.float64)
    
    for tile, label in slide_tile_clst_tuples:
        h = tile.h_id - 1 
        w = tile.w_id - 1
        if h >= H or w >= W or h < 0 or w < 0:
            warnings.warn('Out of range coordinates.')
            continue
        heat_clst[h, w] = label * 1.0
        white_mask[h, w] = 0.0
    
    c_panel = cmapy.cmap('tab10')
    heat_clst_cv2_col = col_pal_cv2_10(heat_clst).astype("uint8")
    heat_clst = Image.fromarray(heat_clst_cv2_col).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    white_mask = image_tools.np_to_pil(white_mask).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    heat_clst_col = cv2.applyColorMap(np.uint8(heat_clst), c_panel)
    heat_clst_col = apply_mask(heat_clst_col, white_mask)
    
    org_image, _ = slide_tools.original_slide_and_scaled_pil_image(slide_tile_clst_tuples[0][0].original_slide_filepath,
                                                                   ENV_task.SCALE_FACTOR, print_opening=False)
    org_np_img = image_tools.pil_to_np_rgb(org_image)
    print('generate cluster spatial map and keep the original image for slide: {}'.format(slide_id))
    if cut_left:
        print('--- cut left, only keep right part!')
        heat_clst_col = keep_right_half(heat_clst_col)
        org_np_img = keep_right_half(org_np_img)
    return org_np_img, heat_clst_col

def gen_single_slide_pick_clst_spatial(ENV_task, slide_tile_clst_tuples, slide_id, label_picked, cut_left=False):
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
    
    def keep_right_half(img):
        height, width, _ = img.shape
        start_col = width // 2
        return img[:, start_col:]
    
    slide_np, _ = slide_tile_clst_tuples[0][0].get_np_scaled_slide()
    H = round(slide_np.shape[0] * ENV_task.SCALE_FACTOR / ENV_task.TILE_H_SIZE)
    W = round(slide_np.shape[1] * ENV_task.SCALE_FACTOR / ENV_task.TILE_W_SIZE)
    heat_s_clst = np.ones((H, W, 3), dtype=np.float64)
    white_mask = np.ones((H, W, 3), dtype=np.float64)
    
    for tile, label in slide_tile_clst_tuples:
        if label == label_picked:
            h = tile.h_id - 1 
            w = tile.w_id - 1
            if h >= H or w >= W or h < 0 or w < 0:
                warnings.warn('Out of range coordinates.')
                continue
            heat_s_clst[h, w] = label * 1.0
            white_mask[h, w] = 0.0
    
    c_panel = cmapy.cmap('tab10')
    heat_clst_cv2_col = col_pal_cv2_10(heat_s_clst).astype("uint8")
    heat_s_clst = Image.fromarray(heat_clst_cv2_col).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    white_mask = image_tools.np_to_pil(white_mask).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    heat_s_clst_col = cv2.applyColorMap(np.uint8(heat_s_clst), c_panel)
    heat_s_clst_col = apply_mask(heat_s_clst_col, white_mask)
    
    print('generate cluster-{}\'s spatial map for slide: {}'.format(label_picked, slide_id))
    if cut_left:
        print('--- cut left, only keep right part!')
        heat_s_clst_col = keep_right_half(heat_s_clst_col)
    return heat_s_clst_col

def gen_single_slide_iso_gath_spatial(ENV_task, slide_tgt_tiles_dict, slide_id):
    '''
    generate the iso-group spatial map for specific cluster on a single slide
    
    Return:
        
    '''
    def apply_mask(heat, white_mask):
        new_heat = np.uint32(np.float32(heat) + np.float32(white_mask))
        new_heat = np.uint8(np.minimum(new_heat, 255))
        return new_heat
    
    # (nb_tgt_lbl, pct_tgt_lbl, 0 if pct_tgt_lbl < iso_thd else 1, tile)
    tgt_tile_tuples = slide_tgt_tiles_dict[slide_id]
    
    slide_np, _ = tgt_tile_tuples[0][3].get_np_scaled_slide()
    H = round(slide_np.shape[0] * ENV_task.SCALE_FACTOR / ENV_task.TILE_H_SIZE)
    W = round(slide_np.shape[1] * ENV_task.SCALE_FACTOR / ENV_task.TILE_W_SIZE)
    heat_s_iso_gath = np.ones((H, W, 3), dtype=np.float64)
    white_mask = np.ones((H, W, 3), dtype=np.float64)
    
    for _, _, iso_label, tile in tgt_tile_tuples:
        if iso_label == 0:
            h, w = tile.h_id - 1, tile.w_id - 1
            if h >= H or w >= W or h < 0 or w < 0:
                warnings.warn('Out of range coordinates.')
                continue
            heat_s_iso_gath[h, w] = 0.8
            white_mask[h, w] = 0.0
        elif iso_label == 1:
            h, w = tile.h_id - 1, tile.w_id - 1
            if h >= H or w >= W or h < 0 or w < 0:
                warnings.warn('Out of range coordinates.')
                continue
            heat_s_iso_gath[h, w] = 0.2
            white_mask[h, w] = 0.0
            
    c_panel = cmapy.cmap('bwr')
    heat_s_iso_gath = image_tools.np_to_pil(heat_s_iso_gath).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    white_mask = image_tools.np_to_pil(white_mask).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    heat_s_iso_col = cv2.applyColorMap(np.uint8(heat_s_iso_gath), c_panel)
    heat_s_iso_col = apply_mask(heat_s_iso_col, white_mask)
    
    print('generate iso-group\'s spatial map for slide: {}'.format(slide_id))
    return heat_s_iso_col

def gen_single_slide_levels_spatial(ENV_task, slide_tgt_tiles_n_dict, bounds, slide_id):
    '''
    generate the iso-group (levels) spatial map for specific cluster on a single slide
    
    Return:
        
    '''
    def apply_mask(heat, white_mask):
        new_heat = np.uint32(np.float32(heat) + np.float32(white_mask))
        new_heat = np.uint8(np.minimum(new_heat, 255))
        return new_heat
    
    # (nb_tgt_lbl, pct_tgt_lbl, 0 if pct_tgt_lbl < iso_thd else 1, tile)
    tgt_tile_tuples = slide_tgt_tiles_n_dict[slide_id]
    
    slide_np, _ = tgt_tile_tuples[0][3].get_np_scaled_slide()
    H = round(slide_np.shape[0] * ENV_task.SCALE_FACTOR / ENV_task.TILE_H_SIZE)
    W = round(slide_np.shape[1] * ENV_task.SCALE_FACTOR / ENV_task.TILE_W_SIZE)
    heat_s_levels = np.ones((H, W, 3), dtype=np.float64)
    white_mask = np.ones((H, W, 3), dtype=np.float64)
    
    color_dict = {}
    for i, b in enumerate(bounds):
        if i == 0:
            color_dict[i] = 1.0 - b * 1.0 / 2
        else:
            color_dict[i] = 1.0 - ((b - bounds[i-1]) * 1.0 / 2 + bounds[i-1])
    
    for _, _, level_label, tile in tgt_tile_tuples:
        h, w = tile.h_id - 1, tile.w_id - 1
        if h >= H or w >= W or h < 0 or w < 0:
            warnings.warn('Out of range coordinates.')
            continue
        heat_s_levels[h, w] = color_dict[level_label]
        white_mask[h, w] = 0.0
        
    c_panel = cmapy.cmap('bwr')
    heat_s_levels = image_tools.np_to_pil(heat_s_levels).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    white_mask = image_tools.np_to_pil(white_mask).resize((slide_np.shape[1], slide_np.shape[0]), PIL.Image.BOX)
    heat_s_levels_col = cv2.applyColorMap(np.uint8(heat_s_levels), c_panel)
    heat_s_levels_col = apply_mask(heat_s_levels_col, white_mask)
    
    print('generate iso-group\'s spatial map for slide: {}'.format(slide_id))
    return heat_s_levels_col
            

def make_spatial_clusters_on_slides(ENV_task, clustering_pkl_name, keep_org_slide=True, cut_left=True):
    '''
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    
    slide_tile_clst_dict = load_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name)
    slide_id_list = list(datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE).keys())
    print('load the slide_ids we have, on the running client (PC or servers), got %d slides...' % len(slide_id_list))
    
    slide_clst_spatmap_dict = {}
    for slide_id in slide_id_list:
        tile_clst_tuples = slide_tile_clst_dict[slide_id]
        org_np_img, heat_clst_col = gen_single_slide_clst_spatial(ENV_task, tile_clst_tuples, 
                                                                  slide_id, cut_left=cut_left)
        
        slide_clst_spatmap_dict[slide_id] = {'original': org_np_img if keep_org_slide else None,
                                             'heat_clst': heat_clst_col}
    
    clst_spatmaps_pkl_name = clustering_pkl_name.replace('clst-res', 'clst-spat')
    store_nd_dict_pkl(heat_store_dir, slide_clst_spatmap_dict, clst_spatmaps_pkl_name)
    print('Store slides clusters spatial maps numpy package as: {}'.format(clst_spatmaps_pkl_name))
    
def make_tiles_demo_clusters(ENV_task, clustering_pkl_name, nb_sample=50):
    '''
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    
    clst_tile_slideid_dict = load_clst_res_label_tile_slide(model_store_dir, clustering_pkl_name)
    
    clst_tile_img_dict = {}
    for label in clst_tile_slideid_dict.keys():
        tile_slideid_tuples = clst_tile_slideid_dict[label]
        picked_tile_slideids = safe_random_sample(tile_slideid_tuples, nb_sample)
        clst_tile_img_dict[label] = []
        
        for tile, slide_id in picked_tile_slideids:
            tile_img = tile.get_np_tile()
            clst_tile_img_dict[label].append((slide_id, tile, tile_img))
        print(f'sampled {nb_sample} tile demos for cluster-{label}' )
            
    clst_tiledemo_pkl_name = clustering_pkl_name.replace('clst-res', 'clst-tiledemo')
    clst_tiledemo_pkl_name = clst_tiledemo_pkl_name.replace('hiera-res', 'hiera-tiledemo')
    store_nd_dict_pkl(heat_store_dir, clst_tile_img_dict, clst_tiledemo_pkl_name)
    print('Store clusters tile demo image numpy package as: {}'.format(clst_tiledemo_pkl_name))
    
def make_tiles_ihcdab_demo_clusters(ENV_task, clustering_pkl_name, nb_sample=50):
    '''
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    
    clst_tile_slideid_dict = load_clst_res_label_tile_slide(model_store_dir, clustering_pkl_name)
    
    clst_t_dab_img_dict = {}
    for label in clst_tile_slideid_dict.keys():
        tile_slideid_tuples = clst_tile_slideid_dict[label]
        picked_tile_slideids = safe_random_sample(tile_slideid_tuples, nb_sample)
        clst_t_dab_img_dict[label] = []
        
        for tile, slide_id in picked_tile_slideids:
            tile_img = tile.get_np_tile()
            tile_dab_img = tile.get_ihc_dab_np_tile()
            clst_t_dab_img_dict[label].append((slide_id, tile, tile_img, tile_dab_img))
        print(f'sampled {nb_sample} tile demos for cluster-{label}' )
            
    clst_t_dab_demo_pkl_name = clustering_pkl_name.replace('clst-res', 'clst-t_dab-demo')
    clst_t_dab_demo_pkl_name = clst_t_dab_demo_pkl_name.replace('hiera-res', 'hiera-t_dab-demo')
    store_nd_dict_pkl(heat_store_dir, clst_t_dab_img_dict, clst_t_dab_demo_pkl_name)
    print('Store clusters tile demo image numpy package as: {}'.format(clst_t_dab_demo_pkl_name))
    
def make_spatial_each_clusters_on_slides(ENV_task, clustering_pkl_name, sp_clst=None):
    '''
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    
    slide_tile_clst_dict = load_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name)
    slide_id_list = list(datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE).keys())
    print('load the slide_ids we have, on the running client (PC or servers), got %d slides...' % len(slide_id_list))
    
    nb_clst = len(load_clst_res_label_tile_slide(model_store_dir, clustering_pkl_name).keys())
    clst_labels = list(range(nb_clst))
    
    slide_clst_s_spatmap_dict = {}
    for slide_id in slide_id_list:
        tile_clst_tuples = slide_tile_clst_dict[slide_id]
        
        label_spatmap_dict = {}
        for label_picked in clst_labels:
            if sp_clst is not None and label_picked != sp_clst:
                continue
            heat_s_clst_col = gen_single_slide_pick_clst_spatial(ENV_task, tile_clst_tuples, slide_id, label_picked)
            label_spatmap_dict[label_picked] = heat_s_clst_col
        slide_clst_s_spatmap_dict[slide_id] = label_spatmap_dict
            
    clst_s_spatmap_pkl_name = clustering_pkl_name.replace('clst-res', 
                                                          'clst-s-spat' if sp_clst != None else 'clst-{}-spat'.format(str(sp_clst)))
    store_nd_dict_pkl(heat_store_dir, slide_clst_s_spatmap_dict, clst_s_spatmap_pkl_name)
    print('Store slides clusters (for each) spatial maps numpy package as: {}'.format(clst_s_spatmap_pkl_name))
    
def make_spatial_iso_gath_on_slides(ENV_task, clustering_pkl_name, sp_clst, iso_thd, radius):
    '''
    make the pkl package with spatial heatmaps of iso group in sp_clst
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    
    clustering_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name)
    slide_tgt_tiles_dict = refine_sp_cluster_homoneig(clustering_res_pkg, sp_clst, iso_thd, radius)
    
    slide_id_list = list(datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE).keys())
    print('load the slide_ids we have, on the running client (PC or servers), got %d slides...' % len(slide_id_list))
    
    slide_iso_s_spatmap_dict = {}
    for slide_id in slide_id_list:
        heat_s_iso_col = gen_single_slide_iso_gath_spatial(ENV_task, slide_tgt_tiles_dict, slide_id)
        slide_iso_s_spatmap_dict[slide_id] = heat_s_iso_col
        
    clst_iso_spatmap_pkl_name = clustering_pkl_name.replace('clst-res', 
                                                            'clst-s-iso' if sp_clst != None else 'clst-{}-iso'.format(str(sp_clst)))
    store_nd_dict_pkl(heat_store_dir, slide_iso_s_spatmap_dict, clst_iso_spatmap_pkl_name)
    print('Store slides iso_group for sp_clst spatial maps numpy package as: {}'.format(clst_iso_spatmap_pkl_name))
    
def make_spatial_levels_on_slides(ENV_task, clustering_pkl_name, sp_clst, radius):
    '''
    make the pkl package with spatial heatmaps of iso group in sp_clst
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    
    clustering_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name)
    slide_tgt_tiles_dict, bounds = refine_sp_cluster_levels(clustering_res_pkg, sp_clst, radius)
    
    slide_id_list = list(datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE).keys())
    print('load the slide_ids we have, on the running client (PC or servers), got %d slides...' % len(slide_id_list))
    
    slide_levels_s_spatmap_dict = {}
    for slide_id in slide_id_list:
        heat_s_levels_col = gen_single_slide_levels_spatial(ENV_task, slide_tgt_tiles_dict, bounds, slide_id)
        slide_levels_s_spatmap_dict[slide_id] = heat_s_levels_col
        
    clst_levels_spatmap_pkl_name = clustering_pkl_name.replace('clst-res', 
                                                               'clst-s-lv' if sp_clst != None else 'clst-{}-lv'.format(str(sp_clst)))
    store_nd_dict_pkl(heat_store_dir, slide_levels_s_spatmap_dict, clst_levels_spatmap_pkl_name)
    print('Store slides levels_group for sp_clst spatial maps numpy package as: {}'.format(clst_levels_spatmap_pkl_name))

 
''' -------------------------------------------------------------------------------------------- ''' 
    
def get_all_clst_labels(slide_tile_clst_dict):
    """
    """
    label_set = set()  # avoid repeat label

    # process bar
    for tile_label_list in tqdm(slide_tile_clst_dict.values(), desc="Search all clst-res:"):
        for _, label in tile_label_list:
            label_set.add(label)

    return list(label_set)    

def cnt_tis_pct_abs_num_clsts_on_slides(ENV_task, clustering_pkl_name, slides_tiles_dict=None):
    '''
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    slide_tile_clst_dict = load_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name)
    clst_labels = get_all_clst_labels(slide_tile_clst_dict)
    
    slide_id_list = list(datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE).keys())
    slide_tis_pct_dict, tis_pct_dict_list = {}, []
    for slide_id in slide_id_list:
        if slide_id not in slide_tile_clst_dict.keys():
            print(f'slide: {slide_id} not in clst results')
            continue
        tile_clst_tuples = slide_tile_clst_dict[slide_id]
        # count tissue percentage
        if slides_tiles_dict is None:
            nb_tiles = None
        else:
            nb_tiles = len(slides_tiles_dict[slide_id])
        tissue_pct_dict, abs_num_dict = tissue_pct_clst_single_slide(tile_clst_tuples, 
                                                                     clst_labels, nb_tiles )
        slide_tis_pct_dict[slide_id] = (tissue_pct_dict, abs_num_dict)
        # tis_pct_dict_list.append(tissue_pct_dict)
    
    # slide_tis_pct_dict['avg'] = avg_tis_pct_clst_on_slides(tis_pct_dict_list)
    clst_prefix = 'hiera-res' if clustering_pkl_name.startswith('hiera-res') else 'clst-res'
    tis_pct_prefix = 'hiera-tis-pct' if clustering_pkl_name.startswith('hiera-res') else 'clst-tis-pct'
    tis_pct_pkl_name = clustering_pkl_name.replace(clst_prefix, tis_pct_prefix)
    store_nd_dict_pkl(heat_store_dir, slide_tis_pct_dict, tis_pct_pkl_name)
    print('Store clusters tissue percentage record as: {}'.format(tis_pct_pkl_name))
    
    
def top_pct_slides_4_sp_clst(ENV_task, tis_pct_pkl_name, lobular_label_fname, sp_clst, nb_top):
    '''
    PS: get both top-k and lowest-k
    '''
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    all_slide_id_list = list(datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE).keys())
    # filter the slide_ids appear in 0-3 bi-labels
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    slide_id_list = []
    for slide_id in all_slide_id_list:
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id in lobular_label_dict.keys():
            slide_id_list.append(slide_id)
    
    slide_sp_clst_tispct = [0.0] * len(slide_id_list)
    for i, slide_id in enumerate(slide_id_list):
        tissue_pct_dict = slide_tis_pct_dict[slide_id]
        tissue_pct_sp_clst = tissue_pct_dict[sp_clst]
        slide_sp_clst_tispct[i] = tissue_pct_sp_clst
    
    order = np.argsort(slide_sp_clst_tispct)
    lowest_ord = order[:nb_top]
    top_ord = order[-nb_top:]
    
    top_slides_ids, lowest_slides_ids = [], []
    for ord in top_ord:
        top_slides_ids.append(slide_id_list[ord])
    for ord in lowest_ord:
        lowest_slides_ids.append(slide_id_list[ord])
        
    return top_slides_ids, lowest_slides_ids

def cnt_prop_slides_ref_homo_sp_clst(ENV_task, clustering_pkl_name, sp_clst, iso_thd, radius):
    '''
    count the population (nb_iso / (nb_iso + nb_gath) ) of refined clusters (into 2 groups by homogeneity in context)
    for each slide
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    clustering_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name)
    
    slide_tgt_tiles_dict = refine_sp_cluster_homoneig(clustering_res_pkg, sp_clst, iso_thd, radius)
    
    slide_iso_gath_nb_dict = {}
    for slide_id in slide_tgt_tiles_dict.keys():
        ref_tiles_tuples = slide_tgt_tiles_dict[slide_id]
        nb_iso, nb_gath = 0, 0
        for tile_tuple in ref_tiles_tuples:
            if tile_tuple[2] == 0:
                nb_iso += 1
            else:
                nb_gath += 1
        pop_iso = nb_iso * 1.0 / len(ref_tiles_tuples)
        pop_gath = nb_gath * 1.0 / len(ref_tiles_tuples)
        slide_iso_gath_nb_dict[slide_id] = (nb_iso, nb_gath, pop_iso, pop_gath)
        print('slide %s has %d/%d iso/gath tiles for cluster %d' % (slide_id,
                                                                    nb_iso, nb_gath, sp_clst))
        
    return slide_iso_gath_nb_dict

def cnt_prop_slides_ref_levels_sp_clst(ENV_task, clustering_pkl_name, sp_clst, radius=5):
    '''
    count the population (nb_iso / (nb_iso + nb_gath) ) of refined clusters (into n groups by proportion of homo-neighbours)
    for each slide
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    clustering_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name)
    
    slide_tgt_tiles_n_dict, bounds = refine_sp_cluster_levels(clustering_res_pkg, sp_clst, radius)
    
    slide_levels_nb_dict = {}
    for slide_id in slide_tgt_tiles_n_dict.keys():
        ref_tiles_tuples = slide_tgt_tiles_n_dict[slide_id]
        
        levels_nb_dict = {}
        for level in range(len(bounds)):
            levels_nb_dict[level] = 0
        for tile_tuple in ref_tiles_tuples:
            levels_nb_dict[tile_tuple[2]] += 1
        
        levels_pop_dict = {}
        for level in range(len(bounds)):
            levels_pop_dict[level] = levels_nb_dict[level] * 1.0 / len(ref_tiles_tuples)
        
        slide_levels_nb_dict[slide_id] = (levels_nb_dict, levels_pop_dict)
        print('slide %s has levels: %s for cluster %d' % (slide_id, str(levels_nb_dict), sp_clst))
        
    return slide_levels_nb_dict, bounds

def cnt_tis_pct_slides_ref_homo_sp_clst(ENV_task, clustering_pkl_name, sp_clst, iso_thd=0.25, radius=5):
    '''
    load the tissue percentage of iso/gath tiles in specific cluster and write them into .csv file
    PS: for other researchers who need raw data
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    # slide_tile_clst_dict = load_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name)
    #
    
    slide_tile_ref_gp_dict, slide_tis_nb_dict = load_ref_group_slide_tile_sp_clst(model_store_dir, clustering_pkl_name,
                                                               sp_clst, iso_thd, radius)
    
    slide_id_list = list(datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE).keys())
    slide_id_list.sort() # sort by string of slide_id
    slide_tis_pct_dict = {}
    for slide_id in slide_id_list:
        if slide_id not in slide_tile_ref_gp_dict.keys():
            continue
        tile_ref_gp_tuples = slide_tile_ref_gp_dict[slide_id]
        tissue_pct_dict, _ = tissue_pct_clst_single_slide(tile_ref_gp_tuples, [0, 1], slide_tis_nb_dict[slide_id])
        slide_tis_pct_dict[slide_id] = tissue_pct_dict
        
    tis_pct_pkl_name = clustering_pkl_name.replace('clst-res', 'clst-gp-tis-pct')
    store_nd_dict_pkl(heat_store_dir, slide_tis_pct_dict, tis_pct_pkl_name)
    print('Store clusters tissue percentage for clst-ref-group as: {}'.format(tis_pct_pkl_name))
    
    tis_pct_csv_name = tis_pct_pkl_name.replace('.pkl', '.csv')
    with open(os.path.join(heat_store_dir, tis_pct_csv_name), 'w', newline='') as csvfile:
        fieldnames = ['slide_id', 'iso_tiles', 'gath_tiles']  
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for slide_id in slide_id_list:
            tissue_pct_dict = slide_tis_pct_dict[slide_id]
            writer.writerow({
                'slide_id': slide_id,
                'iso_tiles': tissue_pct_dict.get(0),
                'gath_tiles': tissue_pct_dict.get(1)
            })
    print('Write clusters tissue percentage csv for clst-ref-group in: {}'.format(tis_pct_csv_name))


def top_pop_slides_4_ref_group(ENV_task, slide_iso_gath_nb_dict, lobular_label_fname, nb_top):
    '''
    find the slide_ids with top-k and lowest-k number of iso/gath refined group in sp_clst
    '''
    all_slide_id_list = list(datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE).keys())
    # filter the slide_ids appear in 0-3 bi-labels
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    slide_id_list = []
    for slide_id in all_slide_id_list:
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id in lobular_label_dict.keys():
            slide_id_list.append(slide_id)
            
    # number of iso/gath tiles for sp_clst
    # slide_nb_iso = [0] * len(slide_id_list)
    # slide_nb_gath = [0] * len(slide_id_list)
    slide_pop_iso = [0.0] * len(slide_id_list)
    slide_pop_gath = [0.0] * len(slide_id_list)
    for i, slide_id in enumerate(slide_id_list):
        nb_iso, nb_gath, pop_iso, pop_gath = slide_iso_gath_nb_dict[slide_id]
        # slide_nb_iso[i] = nb_iso
        # slide_nb_gath[i] = nb_gath
        slide_pop_iso[i] = pop_iso
        slide_pop_gath[i] = pop_gath
        
    # order_iso = np.argsort(slide_nb_iso)
    # order_gath = np.argsort(slide_nb_gath)
    order_iso = np.argsort(slide_pop_iso)
    order_gath = np.argsort(slide_pop_gath)
    lowest_ord_iso, top_ord_iso = order_iso[:nb_top], order_iso[-nb_top:]
    lowest_ord_gath, top_ord_gath = order_gath[:nb_top], order_gath[-nb_top:]
    top_iso_slides_ids, lowest_iso_slides_ids = [], []
    top_gath_slides_ids, lowest_gath_slides_ids = [], []
    for ord in top_ord_iso:
        top_iso_slides_ids.append(slide_id_list[ord])
    for ord in lowest_ord_iso:
        lowest_iso_slides_ids.append(slide_id_list[ord])
    for ord in top_ord_gath:
        top_gath_slides_ids.append(slide_id_list[ord])
    for ord in lowest_ord_gath:
        lowest_gath_slides_ids.append(slide_id_list[ord])
    
    return top_iso_slides_ids, lowest_iso_slides_ids, top_gath_slides_ids, lowest_gath_slides_ids
        
    
''' --------- tissue percentage --------- '''
def tissue_pct_clst_single_slide(slide_tile_clst_tuples, clst_labels, nb_tiles=None):
    '''
    '''
    # initial
    tissue_pct_dict = {}
    abs_num_dict = {}
    if nb_tiles == None:
        nb_tissue = len(slide_tile_clst_tuples)
    else:
        nb_tissue = nb_tiles
        
    # for lbl in range(nb_clst)
    for lbl in clst_labels:
        tissue_pct_dict[lbl] = .0
        abs_num_dict[lbl] = 0
        
    for i, t_l_tuple in enumerate(slide_tile_clst_tuples):
        _, label = t_l_tuple
        tissue_pct_dict[label] += (1.0/nb_tissue)
        abs_num_dict[label] += 1
        
    return tissue_pct_dict, abs_num_dict

def norm_t_pct_clst_single_slide(slide_tis_pct_dict, nb_clst):
    '''
    Args:
        slide_tis_pct_dict: tissue percentage dictionary with slide_id and embedded dictionary with cluster labels
    '''
    new_slide_tis_pct_dict = {}
    for slide_id in slide_tis_pct_dict.keys():
        new_slide_tis_pct_dict[slide_id] = {}
    
    for label in range(nb_clst):
        label_t_pcts = []
        for slide_id in slide_tis_pct_dict.keys():
            label_t_pcts.append(slide_tis_pct_dict[slide_id][label])
        label_max_t_pct = max(label_t_pcts)
        for slide_id in slide_tis_pct_dict.keys():
            org_t_pct = slide_tis_pct_dict[slide_id][label]
            new_slide_tis_pct_dict[slide_id][label] = (org_t_pct + 1e-3) / (label_max_t_pct + 1e-3)
    
    return new_slide_tis_pct_dict

def avg_tis_pct_clst_on_slides(tissue_pct_dict_list):
    '''
    '''
    avg_tis_pct_dict, nb_slides = {}, len(tissue_pct_dict_list)
    for s_tis_pct_dict in tissue_pct_dict_list:
        for label in s_tis_pct_dict.keys():
            if label not in avg_tis_pct_dict.keys():
                avg_tis_pct_dict[label] = .0
            avg_tis_pct_dict[label] += (s_tis_pct_dict[label]/nb_slides)
    
    return avg_tis_pct_dict
    
    
''' ---------------------------------------------------------------------------------- '''

def _run_make_clsuters_space_maps(ENV_task, clustering_pkl_name, r_picked=0.01):
    make_clsuters_space_maps(ENV_task, clustering_pkl_name, r_picked)
    
def _run_make_spatial_clusters_on_slides(ENV_task, clustering_pkl_name, keep_org_slide=True, cut_left=True):
    make_spatial_clusters_on_slides(ENV_task, clustering_pkl_name, keep_org_slide, cut_left=cut_left)
    
def _run_make_tiles_demo_clusters(ENV_task, clustering_pkl_name, nb_sample=50):
    make_tiles_demo_clusters(ENV_task, clustering_pkl_name, nb_sample)
    
def _run_make_tiles_ihcdab_demo_clusters(ENV_task, clustering_pkl_name, nb_sample=50):
    make_tiles_ihcdab_demo_clusters(ENV_task, clustering_pkl_name, nb_sample)

def _run_make_spatial_each_clusters_on_slides(ENV_task, clustering_pkl_name, sp_clst=None):
    make_spatial_each_clusters_on_slides(ENV_task, clustering_pkl_name, sp_clst)

def _run_make_spatial_iso_gath_on_slides(ENV_task, clustering_pkl_name, sp_clst=None, iso_thd=0.1, radius=5):
    make_spatial_iso_gath_on_slides(ENV_task, clustering_pkl_name, sp_clst, iso_thd, radius)
    
def _run_make_spatial_levels_on_slides(ENV_task, clustering_pkl_name, sp_clst, radius):
    make_spatial_levels_on_slides(ENV_task, clustering_pkl_name, sp_clst, radius)

def _run_cnt_tis_pct_abs_num_clsts_on_slides(ENV_task, clustering_pkl_name, slides_tiles_dict=None):
    cnt_tis_pct_abs_num_clsts_on_slides(ENV_task, clustering_pkl_name, slides_tiles_dict)
    
def _run_count_tis_pct_slides_ref_homo_sp_clst(ENV_task, clustering_pkl_name, sp_clst, iso_thd=0.1, radius=5):
    cnt_tis_pct_slides_ref_homo_sp_clst(ENV_task, clustering_pkl_name, sp_clst, iso_thd, radius)


if __name__ == '__main__':
    pass