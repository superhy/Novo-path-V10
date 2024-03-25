'''
Created on 24 Mar 2024

@author: super
'''
from interpre.prep_clst_vis import load_clst_res_slide_tile_label
from interpre.prep_dect_vis import load_1by1_assim_res_tiles
from interpre.prep_tools import store_nd_dict_pkl
from models import datasets


def localise_k_clsts_on_all_slides(ENV_task, clustering_pkl_name, c_assimilate_pkl_name, 
                                   sp_clsts):
    '''
    localize the picked clusters (for each tile) on the tissue, for all slides
    receive clustering results and clusters' assimilation results
    put them all on the tissue, assign the label for each tile, 
        assign 'none' for tiles not within sensitive clusters.
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    slides_tiles_dict = datasets.load_slides_tileslist(ENV_task)
    slide_tile_clst_dict = load_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name)
    slides_2_c_assim_tiles_dict = load_1by1_assim_res_tiles(model_store_dir, c_assimilate_pkl_name)
    
    slide_tile_label_dict = {}
    for slide_id in slides_tiles_dict.keys():
        slide_tiles_list = slides_tiles_dict[slide_id]
        
        slide_tile_label_dict[slide_id] = {}
        for tile in slide_tiles_list:
            # at begining, all tiles on all slides is belonging to 'none' (not belong to any sensi-clusters)
            tile_id = '{}-h{}-w{}'.format(slide_id, tile.h_id, tile.w_id)
            slide_tile_label_dict[slide_id][tile_id] = 'none'
        print(f'initialized empty localized map for slide: {slide_id}.')
    
    for slide_id in slide_tile_clst_dict.keys():
        slide_tile_clst_tuples = slide_tile_clst_dict[slide_id]
        for tile, tile_label in slide_tile_clst_tuples:
            if tile_label in sp_clsts:
                # find a sensitive cluster
                tile_id = '{}-h{}-w{}'.format(slide_id, tile.h_id, tile.w_id)
                slide_tile_label_dict[slide_id][tile_id] = tile_label
        print(f'localized the sensitive clusters for slide: {slide_id}.')
        
    for slide_id in slides_2_c_assim_tiles_dict.keys():
        slide_c_assim_dict = slides_2_c_assim_tiles_dict[slide_id]
        for label in sp_clsts:
            assim_tiles = slide_c_assim_dict.get(label, [])
            for tile in assim_tiles:
                # find a assimilated tile from sensitive cluster
                tile_id = '{}-h{}-w{}'.format(slide_id, tile.h_id, tile.w_id)
                slide_tile_label_dict[slide_id][tile_id] = label
        print(f'localized the assimilated tile for slide: {slide_id}.')
        
    new_name = f'{len(sp_clsts)}-1by1_c-a-local'
    local_filename = clustering_pkl_name.replace('clst-res', new_name)
    store_nd_dict_pkl(stat_store_dir, slide_tile_label_dict, local_filename)
    print('Store slides\' 1by1 sensi-clsts (and assims) localized maps numpy package as: {}'.format(local_filename))
    
    return slide_tile_label_dict
    

if __name__ == '__main__':
    pass