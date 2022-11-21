'''
@author: Yang Hu
'''
import os

from interpre.draw_maps import draw_original_image, draw_attention_heatmap
from interpre.prep_tools import load_clustering_pkg_from_pkl


def plot_vit_cls_map(ENV_task, clsmap_pkl_name):
    '''
    plot the cls attention map for the sampled tiles from test slides
    
    loaded slides_tiles_cls_map_dict format:
        {
            slide_id: [
                (att_map_cv, tile, org_img_nd)
                ...
            ]
            ...
        }
    '''
    slides_tiles_cls_map_dict = load_clustering_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, clsmap_pkl_name)
    map_dir = os.path.join(ENV_task.HEATMAP_STORE_DIR, clsmap_pkl_name.split('_')[0])
    alg_name = clsmap_pkl_name.split('_')[1]
    
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
        
    for slide_id in slides_tiles_cls_map_dict.keys():
        for tiles_cls_map_tuple in slides_tiles_cls_map_dict[slide_id]:
            tile_str = 'r{}c{}'.format(tiles_cls_map_tuple[1].h_id, tiles_cls_map_tuple[1].w_id)
            draw_original_image(map_dir, tiles_cls_map_tuple[2], (slide_id + '-org-' + tile_str, alg_name))
            print('* draw original tile image in: {} for slide: {}, tile: {}'.format(map_dir, slide_id, tile_str))
            draw_attention_heatmap(map_dir, tiles_cls_map_tuple[0], None, None,
                                   (slide_id + '-cls-' + tile_str, alg_name))
            print('* draw cls attention map in: {} for slide: {}, tile: {}'.format(map_dir, slide_id, tile_str))


def plot_vit_heads_map(ENV_task, headsmap_pkl_name):
    '''
    plot the cls attention map for the sampled tiles from test slides
    
    loaded slides_tiles_heads_map_dict format:
        {
            slide_id: [
                (heads_att_maps_cv: [
                    heads_att_maps[idx]
                    ...
                ], max_att_map_cv, tile, org_img_nd)
                ...
            ]
            ...
        }
    '''
    slides_tiles_heads_map_dict = load_clustering_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, headsmap_pkl_name)
    map_dir = os.path.join(ENV_task.HEATMAP_STORE_DIR, headsmap_pkl_name.split('_')[0])
    alg_name = headsmap_pkl_name.split('_')[1]
    
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
        
    for slide_id in slides_tiles_heads_map_dict.keys():
        for tiles_heads_map_tuple in slides_tiles_heads_map_dict[slide_id]:
            tile_str = 'r{}c{}'.format(tiles_heads_map_tuple[2].h_id, tiles_heads_map_tuple[2].w_id)
            draw_original_image(map_dir, tiles_heads_map_tuple[3], (slide_id + '-org-' + tile_str, alg_name))
            print('* draw original tile image in: {} for slide: {}, tile: {}'.format(map_dir, slide_id, tile_str))
            
            x_map_dir = os.path.join(map_dir, 'x')
            if not os.path.exists(x_map_dir): os.makedirs(x_map_dir)
            draw_attention_heatmap(x_map_dir, tiles_heads_map_tuple[1], None, None,
                                   (slide_id + '-x-' + tile_str, alg_name))
            print('* draw max-head attention map in: {} for slide: {}, tile: {}'.format(map_dir, slide_id, tile_str))
            
            for h, h_map_cv in enumerate(tiles_heads_map_tuple[0]):
                h_str = 'h%d' % h
                
                h_map_dir = os.path.join(map_dir, h_str)
                if not os.path.exists(h_map_dir): os.makedirs(h_map_dir)
                draw_attention_heatmap(h_map_dir, h_map_cv, None, None,
                                       (slide_id + '-{}-'.format(h_str) + tile_str, alg_name))
                print('** draw head: {} attention map in: {} for slide: {}, tile: {}'.format(h_str, map_dir, slide_id, tile_str))
    
    
def _run_plot_vit_cls_map(ENV_task, clsmap_pkl_name):
    plot_vit_cls_map(ENV_task, clsmap_pkl_name)
    
def _run_plot_vit_heads_map(ENV_task, headsmap_pkl_name):
    plot_vit_heads_map(ENV_task, headsmap_pkl_name)

if __name__ == '__main__':
    pass