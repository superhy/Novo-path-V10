'''
Created on 17 Nov 2022

@author: Yang Hu
'''
from interpre.prep_tools import safe_random_sample, tSNE_transform, \
    store_nd_dict_pkl
from models.functions_clustering import load_clustering_pkg_from_pkl
import numpy as np
from support.tools import Time


# fixed discrete color value mapping (with 20 colors) for cv2 color palette
def col_pal_cv2_20(i_nd):
    return 5.0 + (255.0 / 20) * i_nd
# same with above with 10 colors
def col_pal_cv2_10(i_nd):
    return 10.0 + (255.0 / 10) * i_nd
# 5 colors
def col_pal_cv2_5(i_nd):
    return 20.0 + (255.0 / 5) * i_nd



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
    

def clst_encode_redu_tsne(clst_encode_tuples):
    '''
    using tSNE dim-reduction
    
    Return:
        {label: nd_array [embeds] dim-redu encodes}
    '''
    encodes, labels = [], []
    for labels, encode in clst_encode_tuples:
        encodes.append(encode)
        labels.append(labels)
        
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
    

def make_spatial_clusters_on_slides(ENV_task, clustering_pkl_name, slide_id):
    '''
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    
    slide_tile_clst_dict = load_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name)
    # TODO:
    
''' ---------------------------------------------------------------------------------- '''

def _run_make_clsuters_space_maps(ENV_task, clustering_pkl_name, nb_picked=1000):
    make_clsuters_space_maps(ENV_task, clustering_pkl_name, nb_picked)


if __name__ == '__main__':
    pass