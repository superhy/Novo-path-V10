'''
Created on 24 Mar 2024

@author: super
'''
from interpre.prep_clst_vis import load_clst_res_slide_tile_label
from interpre.prep_dect_vis import load_1by1_assim_res_tiles
from interpre.prep_tools import store_nd_dict_pkl
from models import datasets
from _plotly_utils.png import group


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
        slide_stats = {group_idx: 0 for group_idx in range(-1, len(clst_gps))}
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
    
    return slide_group_props_dict

def aggregation_of_clst_gps_on_all_slides(slide_tile_label_dict, clst_gps, radius=5):
    '''
    '''
    # Helper function to parse tile position from tile_id
    def parse_tile_location(tile_id):
        parts = tile_id.split('-')
        return int(parts[1][1:]), int(parts[2][1:])  # h_id, w_id

    # Map each cluster label to its group index
    label_to_group = {}
    for group_idx, labels in enumerate(clst_gps):
        for label in labels:
            label_to_group[label] = group_idx

    # Initialise the result dictionary
    slide_group_scores = {}

    for slide_id, tiles in slide_tile_label_dict.items():
        group_counts = {group_idx: 0 for group_idx in range(len(clst_gps))}
        total_counts = {group_idx: 0 for group_idx in range(len(clst_gps))}

        for tile_id, clst_label in tiles.items():
            h_id, w_id = parse_tile_location(tile_id)

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

                        neighbor_label = tiles[neighbor_tile_id]
                        if neighbor_label in label_to_group and label_to_group[neighbor_label] == group_idx:
                            group_counts[group_idx] += 1
                        total_counts[group_idx] += 1

        # Calculate aggregation scores for each group
        agt_scores = {}
        for group_idx in group_counts:
            if total_counts[group_idx] > 0:
                agt_scores[group_idx] = group_counts[group_idx] / total_counts[group_idx]
            else:
                agt_scores[group_idx] = 0

        slide_group_scores[slide_id] = agt_scores
        print('TODO:')

    return slide_group_scores
    

if __name__ == '__main__':
    pass