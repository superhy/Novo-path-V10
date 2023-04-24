'''
Created on 17 Nov 2022

@author: Yang Hu
'''
import warnings

from PIL import Image
import PIL
import cmapy
import cv2
from scipy.stats._continuous_distns import dweibull

from interpre.prep_tools import safe_random_sample, tSNE_transform, \
    store_nd_dict_pkl, load_vis_pkg_from_pkl
from models import datasets
from models.functions_clustering import load_clustering_pkg_from_pkl
import numpy as np
from support.tools import Time
from wsi import image_tools, slide_tools


# fixed discrete color value mapping (with 20 colors) for cv2 color palette
def col_pal_cv2_20(i_nd):
    return 5.0 + (255.0 / 20) * i_nd
# same with above with 10 colors
def col_pal_cv2_10(i_nd):
    return 10.0 + (255.0 / 10) * i_nd


def load_clst_res_encode_label(model_store_dir, clustering_pkl_name, nb_points_clst=None):
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
    if nb_points_clst is not None:
        for label in clst_encode_dict.keys():
            label_encodes_list = clst_encode_dict[label]
            pick_clst_encode_dict[label] = safe_random_sample(label_encodes_list, nb_points_clst)
    else:
        pick_clst_encode_dict = clst_encode_dict
            
    for label, encode_list in pick_clst_encode_dict.items():
        for encode in encode_list:
            clst_encode_list.append((label, encode))
            
    return pick_clst_encode_dict, clst_encode_list
    
    
def load_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name):
    '''
    Return:
        {slide_id: [(tile, clst_label)]}
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

def make_clsuters_space_maps(ENV_task, clustering_pkl_name, nb_picked=None):
    '''
    reduce the clusters points to a feature space
    store the clst - dim_redu space in pkl
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    _, clst_encode_list = load_clst_res_encode_label(model_store_dir, clustering_pkl_name, nb_picked)
    clst_redu_en_dict = clst_encode_redu_tsne(clst_encode_tuples=clst_encode_list)
    clst_tsne_pkl_name = 'tsne_{}_{}'.format('all' if nb_picked is None else str(nb_picked), clustering_pkl_name)
    store_nd_dict_pkl(stat_store_dir, clst_redu_en_dict, clst_tsne_pkl_name)
    print('done the clusters dim-reduction and store as: ', clst_tsne_pkl_name)
    

def gen_single_slide_clst_spatial(ENV_task, slide_tile_clst_tuples, slide_id):
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
    return org_np_img, heat_clst_col

def gen_single_slide_clst_each_spatial(ENV_task, slide_tile_clst_tuples, slide_id, label_picked):
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
    return heat_s_clst_col

def make_spatial_clusters_on_slides(ENV_task, clustering_pkl_name, keep_org_slide=True):
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
        org_np_img, heat_clst_col = gen_single_slide_clst_spatial(ENV_task, tile_clst_tuples, slide_id)
        
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
        print('sampled %d tile demos for cluster-%d' % (nb_sample, label) )
            
    clst_tiledemo_pkl_name = clustering_pkl_name.replace('clst-res', 'clst-tiledemo')
    store_nd_dict_pkl(heat_store_dir, clst_tile_img_dict, clst_tiledemo_pkl_name)
    print('Store clusters tile demo image numpy package as: {}'.format(clst_tiledemo_pkl_name))
    
def make_spatial_each_clusters_on_slides(ENV_task, clustering_pkl_name, storage_batchsize=8):
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
            heat_s_clst_col = gen_single_slide_clst_each_spatial(ENV_task, tile_clst_tuples, slide_id, label_picked)
            label_spatmap_dict[label_picked] = heat_s_clst_col
        slide_clst_s_spatmap_dict[slide_id] = label_spatmap_dict
            
    clst_s_spatmap_pkl_name = clustering_pkl_name.replace('clst-res', 'clst-s-spat')
    store_nd_dict_pkl(heat_store_dir, slide_clst_s_spatmap_dict, clst_s_spatmap_pkl_name)
    print('Store slides clusters (for each) spatial maps numpy package as: {}'.format(clst_s_spatmap_pkl_name))
    
    
def count_tissue_pct_clsts_on_slides(ENV_task, clustering_pkl_name):
    '''
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    slide_tile_clst_dict = load_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name)
    
    slide_id_list = list(datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE).keys())
    slide_tis_pct_dict, tissue_pct_dict_list = {}, []
    for slide_id in slide_id_list:
        tile_clst_tuples = slide_tile_clst_dict[slide_id]
        # count tissue percentage
        tissue_pct_dict = tissue_pct_clst_single_slide(tile_clst_tuples, ENV_task.NUM_CLUSTERS)
        slide_tis_pct_dict[slide_id] = tissue_pct_dict
        tissue_pct_dict_list.append(tissue_pct_dict)
    
    slide_tis_pct_dict['avg'] = avg_tis_pct_clst_on_slides(tissue_pct_dict_list)
    tis_pct_pkl_name = clustering_pkl_name.replace('clst-res', 'clst-tis-pct')
    store_nd_dict_pkl(heat_store_dir, slide_tis_pct_dict, tis_pct_pkl_name)
    print('Store clusters tissue percentage record as: {}'.format(tis_pct_pkl_name))
    
def top_pct_slides_4_sp_clst(ENV_task, tis_pct_pkl_name, sp_clst, nb_top):
    '''
    PS: get both top-k and lowest-k
    '''
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    slide_id_list = list(datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE).keys())
    slide_sp_clst_tispct = [0.0] * len(slide_id_list)
    for i, slide_id in enumerate(slide_id_list):
        tissue_pct_dict = slide_tis_pct_dict[slide_id]
        tissue_pct_sp_clst = tissue_pct_dict[sp_clst]
        slide_sp_clst_tispct[i] = tissue_pct_sp_clst
    
    order = np.argsort(slide_sp_clst_tispct)
    lowest_ord = order[:nb_top]
    top_ord = order[nb_top:]
    
    top_slides_ids, lowest_slides_ids = [], []
    for ord in top_ord:
        top_slides_ids.append(slide_id_list[ord])
    for ord in lowest_ord:
        lowest_slides_ids.append(slide_id_list[ord])
        
    return top_slides_ids, lowest_slides_ids
        
    
''' --------- tissue percentage --------- '''
def tissue_pct_clst_single_slide(slide_tile_clst_tuples, nb_clst):
    '''
    '''
    # initial
    tissue_pct_dict, nb_tissue = {}, len(slide_tile_clst_tuples)
    for id in range(nb_clst):
        tissue_pct_dict[id] = .0
        
    for i, t_l_tuple in enumerate(slide_tile_clst_tuples):
        _, label = t_l_tuple
        tissue_pct_dict[label] += (1.0/nb_tissue)
        
    return tissue_pct_dict

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

def _run_make_clsuters_space_maps(ENV_task, clustering_pkl_name, nb_picked=1000):
    make_clsuters_space_maps(ENV_task, clustering_pkl_name, nb_picked)
    
def _run_make_spatial_clusters_on_slides(ENV_task, clustering_pkl_name, keep_org_slide=True):
    make_spatial_clusters_on_slides(ENV_task, clustering_pkl_name, keep_org_slide)
    
def _run_make_tiles_demo_clusters(ENV_task, clustering_pkl_name, nb_sample=50):
    make_tiles_demo_clusters(ENV_task, clustering_pkl_name, nb_sample)
    
def _run_make_spatial_each_clusters_on_slides(ENV_task, clustering_pkl_name):
    make_spatial_each_clusters_on_slides(ENV_task, clustering_pkl_name)

def _run_count_tis_pct_clsts_on_slides(ENV_task, clustering_pkl_name):
    count_tissue_pct_clsts_on_slides(ENV_task, clustering_pkl_name)


if __name__ == '__main__':
    pass