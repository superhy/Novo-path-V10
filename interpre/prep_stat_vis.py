'''
Created on 24 Mar 2024

@author: super
'''
import csv
import os

from _plotly_utils.png import group
import tqdm

from interpre.prep_clst_vis import load_clst_res_slide_tile_label
from interpre.prep_dect_vis import load_1by1_assim_res_tiles
from interpre.prep_tools import store_nd_dict_pkl, load_vis_pkg_from_pkl
from models import datasets
from support.tools import Time
from support.files import parse_caseid_from_slideid


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
        
    local_filename = f'c{len(sp_clsts)}-1by1_c-a-local_{Time().date}.pkl'
    store_nd_dict_pkl(stat_store_dir, slide_tile_label_dict, local_filename)
    print('Store slides\' 1by1 sensi-clsts (and assims) localized maps numpy package as: {}'.format(local_filename))
    
    return slide_tile_label_dict

def proportion_clst_gp_on_each_slides(slide_tile_label_dict, clst_gps):
    '''
    calculate the proportions for cluster groups at each slide
    '''
    # init the proportion dict 
    slide_group_props_dict = {}
    gp_names = {-1: 'N', 0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 
                6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K'}
    
    # run for all the slide
    for slide_id, tiles in slide_tile_label_dict.items():
        # init proportion dict for each slide
        slide_stats = {gp_names[group_idx]: 0 for group_idx in range(-1, len(clst_gps))}
        total_tiles = len(tiles)
        
        # go through every tile
        for tile_id, clst_label in tiles.items():
            # give the tile's belonging
            found_group = False
            for group_idx, clst_group in enumerate(clst_gps):
                if clst_label in clst_group:
                    slide_stats[gp_names[group_idx]] += 1
                    found_group = True
                    break
            
            # not belong to any cluster groups
            if not found_group:
                slide_stats[gp_names[-1]] += 1
        
        # calculate the proportion for each group
        slide_group_props_dict[slide_id] = {group_name: count / total_tiles for group_name, count in slide_stats.items()}
        print(f'Traversed and counted the slide: {slide_id}')
    
    return slide_group_props_dict

def save_slide_group_props_to_csv(ENV_task, slide_group_props_dict):
    '''
    store the slide_group_props_dict into CSV
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    csv_file_path = os.path.join(stat_store_dir, 'slide_clusters_props.csv')
    
    gp_names = set()
    for slide_props in slide_group_props_dict.values():
        gp_names.update({gp_name for gp_name in slide_props.keys() if gp_name != 'N'})
    
    # CSV column titles
    headers = ['slide_id'] + sorted(gp_names) + ['all']
    
    # open csv
    rows_data = []
    for slide_id, proportions in slide_group_props_dict.items():
        # initialize the rows
        row_data = {'slide_id': parse_caseid_from_slideid(slide_id)}
        
        # excluded
        for gp_name, prop_value in proportions.items():
            if gp_name != 'N':
                row_data[gp_name] = prop_value
        
        # sum
        total_prop = sum(value for key, value in proportions.items() if key != 'N')
        # for column 'all'
        row_data['all'] = total_prop
        
        rows_data.append(row_data)
    
    # sorted by 'case_id'
    sorted_rows_data = sorted(rows_data, key=lambda x: x['slide_id'])
    
    # write into csv
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted_rows_data)
    print(f'store the slide_group_props_dict in: {csv_file_path}.')


def aggregation_of_clst_gps_on_all_slides(ENV_task, slide_tile_label_dict, clst_gps, radius=5):
    '''
    calculate the aggregation score for each cluster group, search from neighbor_tile within a radius
    
    Return:
        slide_gp_agt_score_dict, name_gps: 1: the statistic results; 
                                      2: name dict to indicate the group name to the cluster group
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    # Helper function to parse tile position from tile_id
    def parse_tile_location(slide_id, tile_id):
        tile_loc = tile_id.replace(slide_id, 'slide')
        parts = tile_loc.split('-')
        return int(parts[1][1:]), int(parts[2][1:])  # h_id, w_id
    
    if len(clst_gps) > 1:
        gp_names = {-1: 'N', 0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 
                    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K'}
    else:
        gp_names = {0: 'All'}
    name_gps = {}
    for i, clst_gp in enumerate(clst_gps):
        name_gps[gp_names[i]] = clst_gp

    # Map each cluster label to its group index
    label_to_group = {}
    for group_idx, labels in enumerate(clst_gps):
        for label in labels:
            label_to_group[label] = group_idx

    # Initialise the result dictionary
    slide_gp_agt_score_dict = {}

    for slide_id, tiles in slide_tile_label_dict.items():
        group_counts = {g_idx: 0 for g_idx in range(len(clst_gps))}
        total_counts = {g_idx: 0 for g_idx in range(len(clst_gps))}

        for tile_id, clst_label in tiles.items(): # tqdm(, desc=f'load tiles in slide: {slide_id}'):
            h_id, w_id = parse_tile_location(slide_id, tile_id)

            # Check if this tile's cluster label belongs to any group
            if clst_label in label_to_group:
                group_idx = label_to_group[clst_label]

                # Count tiles belonging to the same group within the radius
                for h in range(h_id - radius, h_id + radius + 1):
                    for w in range(w_id - radius, w_id + radius + 1):
                        neighbor_tile_id = '{}-h{}-w{}'.format(slide_id, h, w)
                        # Skip if the neighbor tile is not in the dictionary
                        if neighbor_tile_id not in tiles:
                            continue

                        neighb_c_label = tiles[neighbor_tile_id]
                        if neighb_c_label in label_to_group and label_to_group[neighb_c_label] == group_idx:
                            group_counts[group_idx] += 1
                        total_counts[group_idx] += 1

        # Calculate aggregation scores for each group
        agt_scores = {}
        for g_idx in group_counts:
            if total_counts[g_idx] > 0:
                if total_counts[g_idx] > (radius * 2 + 1) ** 2:
                    agt_scores[gp_names[g_idx]] = group_counts[g_idx] / total_counts[g_idx]
                else:
                    agt_scores[gp_names[g_idx]] = 0
            else:
                agt_scores[gp_names[g_idx]] = 0

        slide_gp_agt_score_dict[slide_id] = agt_scores
        print(f'count aggregation score through all tiles in slide: {slide_id}...')
        
    aggregation_filename = f'agt_c-gps{len(clst_gps)}_rad{radius}_{Time().date}.pkl'
    store_nd_dict_pkl(stat_store_dir, (slide_gp_agt_score_dict, name_gps), aggregation_filename)
    print('Store slides\' 1by1 clst groups aggregation score dict package as: {}'.format(aggregation_filename))

    return slide_gp_agt_score_dict, name_gps

def aggregation_of_sp_clsts_on_all_slides(ENV_task, slide_tile_label_dict, sp_clsts, radius=5):
    '''
    calculate the aggregation score for selected sensitive clusters (one by one), 
        search from neighbor_tile within a radius
    
    Return:
        slide_spc_agt_score_dict
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    # Helper function to parse tile position from tile_id
    def parse_tile_location(slide_id, tile_id):
        tile_loc = tile_id.replace(slide_id, 'slide')
        parts = tile_loc.split('-')
        return int(parts[1][1:]), int(parts[2][1:])  # h_id, w_id

    # Initialise the result dictionary
    slide_spc_agt_score_dict = {}
    for slide_id, tiles in slide_tile_label_dict.items():
        group_counts = {sp_c: 0 for sp_c in sp_clsts}
        total_counts = {sp_c: 0 for sp_c in sp_clsts}

        for tile_id, clst_label in tiles.items(): # tqdm(, desc=f'load tiles in slide: {slide_id}'):
            h_id, w_id = parse_tile_location(slide_id, tile_id)

            # Check if this tile's cluster label belongs to any group
            if clst_label in sp_clsts:
                # Count tiles belonging to the same group within the radius
                for h in range(h_id - radius, h_id + radius + 1):
                    for w in range(w_id - radius, w_id + radius + 1):
                        neighbor_tile_id = '{}-h{}-w{}'.format(slide_id, h, w)
                        # Skip if the neighbor tile is not in the dictionary
                        if neighbor_tile_id not in tiles:
                            continue

                        neighb_c_label = tiles[neighbor_tile_id]
                        if neighb_c_label == clst_label:
                            group_counts[clst_label] += 1
                        total_counts[clst_label] += 1

        # Calculate aggregation scores for each group
        agt_scores = {}
        for sp_c in group_counts:
            if total_counts[sp_c] > 0:
                if total_counts[sp_c] > (radius * 2 + 1) ** 2:
                    agt_scores[sp_c] = group_counts[sp_c] / total_counts[sp_c]
                else:
                    agt_scores[sp_c] = 0
            else:
                agt_scores[sp_c] = 0

        slide_spc_agt_score_dict[slide_id] = agt_scores
        print(f'count aggregation score through all tiles in slide: {slide_id}...')
        
    aggregation_filename = f'agt_sp-c{len(sp_clsts)}_rad{radius}_{Time().date}.pkl'
    store_nd_dict_pkl(stat_store_dir, slide_spc_agt_score_dict, aggregation_filename)
    print('Store slides\' 1by1 sp clst aggregation score dict package as: {}'.format(aggregation_filename))

    return slide_spc_agt_score_dict


''' ----------- running functions ----------- '''
def _run_localise_k_clsts_on_all_slides(ENV_task, clustering_pkl_name, c_assimilate_pkl_name,
                                        sp_clsts):
    '''
    '''
    slide_tile_label_dict = localise_k_clsts_on_all_slides(ENV_task, clustering_pkl_name, c_assimilate_pkl_name,
                                                           sp_clsts)
    return slide_tile_label_dict
    
def _run_agt_of_clst_gps_on_all_slides(ENV_task, slide_t_label_dict_fname, clst_gps, radius=5):
    '''
    '''
    slide_tile_label_dict = load_vis_pkg_from_pkl(ENV_task.STATISTIC_STORE_DIR, slide_t_label_dict_fname)
    slide_gp_agt_score_dict, name_gps = aggregation_of_clst_gps_on_all_slides(ENV_task, slide_tile_label_dict, 
                                                                              clst_gps, radius=radius)
    return slide_gp_agt_score_dict, name_gps

def _run_agt_of_clst_gps_on_all_slides_continue(ENV_task, slide_tile_label_dict, clst_gps, radius=5):
    '''
    '''
    slide_gp_agt_score_dict, name_gps = aggregation_of_clst_gps_on_all_slides(ENV_task, slide_tile_label_dict, 
                                                                              clst_gps, radius=radius)
    return slide_gp_agt_score_dict, name_gps

def _run_agt_of_sp_clsts_on_all_slides(ENV_task, slide_t_label_dict_fname, sp_clsts, radius=5):
    '''
    '''
    slide_tile_label_dict = load_vis_pkg_from_pkl(ENV_task.STATISTIC_STORE_DIR, slide_t_label_dict_fname)
    slide_spc_agt_score_dict = aggregation_of_sp_clsts_on_all_slides(ENV_task, slide_tile_label_dict, 
                                                                     sp_clsts, radius=radius)
    return slide_spc_agt_score_dict

def _run_agt_of_sp_clsts_on_all_slides_continue(ENV_task, slide_tile_label_dict, sp_clsts, radius=5):
    '''
    '''
    slide_spc_agt_score_dict, = aggregation_of_sp_clsts_on_all_slides(ENV_task, slide_tile_label_dict, 
                                                                      sp_clsts, radius=radius)
    return slide_spc_agt_score_dict
    

if __name__ == '__main__':
    pass