'''
@author: Yang Hu
'''
import os

from interpre.draw_maps import draw_original_image, draw_attention_heatmap
from interpre.prep_tools import load_vis_pkg_from_pkl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from support.tools import normalization


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
    slides_tiles_cls_map_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, clsmap_pkl_name)
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
    slides_tiles_heads_map_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, headsmap_pkl_name)
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
    
def plot_reg_ass_homotiles_slides(ENV_task, sp_clst_reg_ass_pkl_name, edge_thd):
    '''
    plot regional association for iso/gath tiles in specific cluster
    
    slide_tile_reg_ass_dict format:
        {
            slide_id: [
                    (ass_mat, tile, sp_homo_lbl)...
                ]
        }
    '''
    slide_tile_reg_ass_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, sp_clst_reg_ass_pkl_name)
    
    reg_ass_dir = os.path.join(ENV_task.HEATMAP_STORE_DIR, sp_clst_reg_ass_pkl_name.split('-')[0])
    for slide_id in slide_tile_reg_ass_dict.keys():
        tile_reg_ass_tuples = slide_tile_reg_ass_dict[slide_id]
        for tile_reg_ass in tile_reg_ass_tuples:
            reg_ass_homo_dir = os.path.join(reg_ass_dir, tile_reg_ass[2])
            if not os.path.exists(reg_ass_homo_dir):
                os.makedirs(reg_ass_homo_dir)
                
            tile = tile_reg_ass[1]
            tile_id = '{}-h{}-w{}'.format(slide_id, tile.h_id, tile.w_id)
            
            reg_ass_mat = tile_reg_ass[0]
            G = nx.Graph() # create an empty graph
            center = np.array(reg_ass_mat.shape) // 2 # centre of the matrix
            reg_ass_mat[tuple(center)] = np.median(reg_ass_mat) # centre value is odd, avoid it affect Norm
            reg_ass_mat = normalization(reg_ass_mat)
            print('check tile: {}, with nb_linked/total: '.format(tile_id), np.sum(reg_ass_mat > edge_thd), reg_ass_mat.size)
            
            for i in range(reg_ass_mat.shape[0]):
                for j in range(reg_ass_mat.shape[1]):
                    # print(reg_ass_mat[i, j], edge_thd)
                    if i == j:
                        continue
                    if reg_ass_mat[i, j] > edge_thd:
                        G.add_edge(tuple(center), (i, j), weight=1)
                        # G.add_edge(tuple(center), (i, j), weight=reg_ass_mat[i, j])
            
            pos = {node: node for node in G.nodes()}
            weights = nx.get_edge_attributes(G, 'weight')
            nx.draw(G, pos, with_labels=False, node_color='blue', node_size=1500)
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
            
            # TDDO: have bug, need to clean the previous graph
            
            plt.savefig(os.path.join(reg_ass_homo_dir, '{}.png'.format(tile_id)), format='png')
            # clear the plt, otherwise, the figures will be repeated coverage
            plt.clf(), plt.cla(), plt.close()

    
    
def _run_plot_vit_cls_map(ENV_task, clsmap_pkl_name):
    plot_vit_cls_map(ENV_task, clsmap_pkl_name)
    
def _run_plot_vit_heads_map(ENV_task, headsmap_pkl_name):
    plot_vit_heads_map(ENV_task, headsmap_pkl_name)
    
def _run_plot_reg_ass_homotiles_slides(ENV_task, sp_clst_reg_ass_pkl_name, edge_thd):
    plot_reg_ass_homotiles_slides(ENV_task, sp_clst_reg_ass_pkl_name, edge_thd)

if __name__ == '__main__':
    pass