'''
@author: Yang Hu
'''

import os

from plotly.graph_objs.layout import xaxis, yaxis
from scipy.interpolate._bsplines import make_interp_spline

from interpre.draw_maps import draw_attention_heatmap, draw_original_image
from interpre.prep_tools import load_vis_pkg_from_pkl
from models import datasets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from support.files import parse_caseid_from_slideid
from support.metadata import query_task_label_dict_fromcsv
from wsi import slide_tools, image_tools


# the colors_10 is mapping with color_pannel cv2 in prep_clst_vis
colors_10 = ['blue', 'orange', 'green', 'red', 'purple', 
             'brown', 'pink', 'gray', 'olive', 'cyan']

def plot_clst_scatter(clst_redu_en_dict):
    '''
    using plotly tools for this chart
    
    k: top_k attentional embeddings
    d: last_d attentional embeddings
    
    Args:
        nb_demo: cut the first N embeds used as the demos
    '''
    
    colors = colors_10[:len(clst_redu_en_dict.keys())]
    
    labels, embed_nds, legend_names = [], [], []
    for i in range(len(clst_redu_en_dict.keys())):
        # keep the label from 0 to 1
        labels.append(i)
        embed_nds.append(clst_redu_en_dict[i])
        legend_names.append('cluster-%d' % (i + 1))
    
    fig = go.Figure()
    for embeds, color, node_name in zip(embed_nds, colors, legend_names):
        fig.add_trace(go.Scatter(
            x=embeds[:, 0], y=embeds[:, 1], mode='markers',
            name=node_name,
            marker=dict(
                color=color,
                size=8.0,
                )
            ))
        
    fig.update_layout(
        plot_bgcolor='white',
        margin=dict(l=10, b=10, r=10, t=10, pad=0),
        xaxis=dict(
            showticklabels=False
            ),
        yaxis=dict(
            showticklabels=False
            ),
        legend=dict(
            font=dict(color='black', size=24),
            bordercolor='black',
            borderwidth=1,
            yanchor="top",
            y=0.2,
            xanchor="right",
            x=0.2,
            ),
        width=1000, height=1000,
        )
        
    fig.write_html('fig5_demo_encodes.html', auto_open=True)
    

def plot_slides_clst_spatmap(ENV_task, clst_spatmaps_pkl_name):
    '''
    '''
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    slide_clst_spatmap_dict = load_vis_pkg_from_pkl(heat_store_dir, clst_spatmaps_pkl_name)
    
    clst_alg_name = clst_spatmaps_pkl_name[:clst_spatmaps_pkl_name.find('_encode')]
    clst_spatmap_dir = os.path.join(heat_store_dir, clst_alg_name)
    if not os.path.exists(clst_spatmap_dir):
        os.makedirs(clst_spatmap_dir)
        print('create file dir {}'.format(clst_spatmap_dir) )
        
    slide_tile_list_dict = datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE)
    for slide_id in slide_clst_spatmap_dict.keys():
        spatmap_into_dict = slide_clst_spatmap_dict[slide_id]
        if spatmap_into_dict['original'] is None:
            org_image, _ = slide_tools.original_slide_and_scaled_pil_image(slide_tile_list_dict[slide_id][0].original_slide_filepath,
                                                                           ENV_task.SCALE_FACTOR, print_opening=False)
            org_np_img = image_tools.pil_to_np_rgb(org_image)
        else:
            org_np_img = spatmap_into_dict['original']
        heat_clst_col = spatmap_into_dict['heat_clst']
        
        draw_original_image(clst_spatmap_dir, org_np_img, (slide_id, 'org') )
        print('draw original image in: {}'.format(os.path.join(clst_spatmap_dir, '{}-{}.png'.format(slide_id, 'org')) ))
        draw_attention_heatmap(clst_spatmap_dir, heat_clst_col, org_np_img, None, (slide_id, 'clst_spat'))
        print('draw cluster spatial map in: {}'.format(os.path.join(clst_spatmap_dir, '{}-{}.png'.format(slide_id, 'clst_spat')) ))
        
def plot_clst_tile_demo(ENV_task, clst_tiledemo_pkl_name):
    '''
    '''
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    clst_tile_slideid_dict = load_vis_pkg_from_pkl(heat_store_dir, clst_tiledemo_pkl_name)
    
    clst_alg_name = clst_tiledemo_pkl_name[:clst_tiledemo_pkl_name.find('_encode')]
    clst_tiledemo_dir = os.path.join(heat_store_dir, clst_alg_name)
    if not os.path.exists(clst_tiledemo_dir):
        os.makedirs(clst_tiledemo_dir)
        print('create file dir {}'.format(clst_tiledemo_dir) )
        
    for label in clst_tile_slideid_dict.keys():
        tiledemo_slideid_tuple = clst_tile_slideid_dict[label]
        clst_dir_name = f'cluster-{label}'
        clst_tiledeme_label_dir = os.path.join(clst_tiledemo_dir, clst_dir_name)
        if not os.path.exists(clst_tiledeme_label_dir):
            os.makedirs(clst_tiledeme_label_dir)
            print('create file dir {}'.format(clst_tiledeme_label_dir))
        
        for slide_id, tile, tile_img in tiledemo_slideid_tuple:
            tiledemo_str = '{}-tile_{}'.format(slide_id, 'h{}-w{}'.format(tile.h_id, tile.w_id) )
            draw_original_image(clst_tiledeme_label_dir, tile_img, (tiledemo_str, '') )
        print(f'draw {len(tiledemo_slideid_tuple)} tile demos for cluster {label}, at: {clst_tiledeme_label_dir}' )
        
def plot_clst_tile_ihcdab_demo(ENV_task, clst_t_dab_demo_pkl_name):
    '''
    '''
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    clst_t_dab_slideid_dict = load_vis_pkg_from_pkl(heat_store_dir, clst_t_dab_demo_pkl_name)
    
    clst_alg_name = clst_t_dab_demo_pkl_name[:clst_t_dab_demo_pkl_name.find('_encode')]
    clst_tiledemo_dir = os.path.join(heat_store_dir, clst_alg_name)
    if not os.path.exists(clst_tiledemo_dir):
        os.makedirs(clst_tiledemo_dir)
        print('create file dir {}'.format(clst_tiledemo_dir) )
        
    for label in clst_t_dab_slideid_dict.keys():
        tdabdemo_slideid_tuple = clst_t_dab_slideid_dict[label]
        clst_dir_name = f'cluster-{label}'
        clst_tiledeme_label_dir = os.path.join(clst_tiledemo_dir, clst_dir_name)
        if not os.path.exists(clst_tiledeme_label_dir):
            os.makedirs(clst_tiledeme_label_dir)
            print('create file dir {}'.format(clst_tiledeme_label_dir))
        
        for slide_id, tile, tile_img, tile_dab_img in tdabdemo_slideid_tuple:
            tiledemo_str = '{}-tile_{}'.format(slide_id, 'h{}-w{}'.format(tile.h_id, tile.w_id) )
            t_dab_demo_str = '{}-t_dab_{}'.format(slide_id, 'h{}-w{}'.format(tile.h_id, tile.w_id) )
            draw_original_image(clst_tiledeme_label_dir, tile_img, (tiledemo_str, '') )
            draw_original_image(clst_tiledeme_label_dir, tile_dab_img, (t_dab_demo_str, '') )
        print(f'draw {len(tdabdemo_slideid_tuple)} tiles(with ihc-dab image) for clst {label}, at: {clst_tiledeme_label_dir}' )
        
def plot_slides_clst_each_spatmap(ENV_task, clst_s_spatmap_pkl_name):
    '''
    '''
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    slide_clst_s_spatmap_dict = load_vis_pkg_from_pkl(heat_store_dir, clst_s_spatmap_pkl_name)
    
    clst_alg_name = clst_s_spatmap_pkl_name[:clst_s_spatmap_pkl_name.find('20')]
    clst_s_spatmap_dir = os.path.join(heat_store_dir, clst_alg_name)
    if not os.path.exists(clst_s_spatmap_dir):
        os.makedirs(clst_s_spatmap_dir)
        print('create file dir {}'.format(clst_s_spatmap_dir) )
        
    slide_tile_list_dict = datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE)
    for slide_id in slide_clst_s_spatmap_dict.keys():
        org_image, _ = slide_tools.original_slide_and_scaled_pil_image(slide_tile_list_dict[slide_id][0].original_slide_filepath,
                                                                           ENV_task.SCALE_FACTOR, print_opening=False)
        org_np_img = image_tools.pil_to_np_rgb(org_image)
        clst_s_spatmap_into_dict = slide_clst_s_spatmap_dict[slide_id]
        
        for label in clst_s_spatmap_into_dict.keys():
            heat_s_clst_col = clst_s_spatmap_into_dict[label]
            draw_attention_heatmap(clst_s_spatmap_dir, heat_s_clst_col, org_np_img, None, (slide_id, 'c%d_spat' % label) )
            print('draw cluster {}\'s spatial map in: {}'.format(str(label), 
                                                                 os.path.join(clst_s_spatmap_dir, '{}-{}.png'.format(slide_id, 'c%d_spat' % label) ) ))
            
def plot_slides_iso_gath_spatmap(ENV_task, clst_iso_spatmap_pkl_name, sp_clst):
    '''
    plot the spatial map for iso group in sp_clst
    '''
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    slide_iso_s_spatmap_dict = load_vis_pkg_from_pkl(heat_store_dir, clst_iso_spatmap_pkl_name)
    
    clst_alg_name = clst_iso_spatmap_pkl_name[:clst_iso_spatmap_pkl_name.find('20')]
    clst_iso_spatmap_dir = os.path.join(heat_store_dir, clst_alg_name)
    if not os.path.exists(clst_iso_spatmap_dir):
        os.makedirs(clst_iso_spatmap_dir)
        print('create file dir {}'.format(clst_iso_spatmap_dir) )
        
    slide_tile_list_dict = datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE)
    for slide_id in slide_iso_s_spatmap_dict.keys():
        org_image, _ = slide_tools.original_slide_and_scaled_pil_image(slide_tile_list_dict[slide_id][0].original_slide_filepath,
                                                                           ENV_task.SCALE_FACTOR, print_opening=False)
        org_np_img = image_tools.pil_to_np_rgb(org_image)
        heat_s_iso_col = slide_iso_s_spatmap_dict[slide_id]
        draw_attention_heatmap(clst_iso_spatmap_dir, heat_s_iso_col, org_np_img, None, (slide_id, 'c%d_spat' % sp_clst) )
        print('draw cluster {}\'s iso group tiles on spatial map in: {}'.format(str(sp_clst),
                                                                                os.path.join(clst_iso_spatmap_dir,
                                                                                             '{}-{}.png'.format(slide_id, 'c%d_spat' % sp_clst) ) ))
        
def plot_slides_levels_spatmap(ENV_task, clst_levels_spatmap_pkl_name, sp_clst):
    '''
    plot the spatial map for iso group in sp_clst
    '''
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    slide_levels_s_spatmap_dict = load_vis_pkg_from_pkl(heat_store_dir, clst_levels_spatmap_pkl_name)
    
    clst_alg_name = clst_levels_spatmap_pkl_name[:clst_levels_spatmap_pkl_name.find('20')]
    clst_levels_spatmap_dir = os.path.join(heat_store_dir, clst_alg_name)
    if not os.path.exists(clst_levels_spatmap_dir):
        os.makedirs(clst_levels_spatmap_dir)
        print('create file dir {}'.format(clst_levels_spatmap_dir) )
        
    slide_tile_list_dict = datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE)
    for slide_id in slide_levels_s_spatmap_dict.keys():
        org_image, _ = slide_tools.original_slide_and_scaled_pil_image(slide_tile_list_dict[slide_id][0].original_slide_filepath,
                                                                       ENV_task.SCALE_FACTOR, print_opening=False)
        org_np_img = image_tools.pil_to_np_rgb(org_image)
        heat_s_levels_col = slide_levels_s_spatmap_dict[slide_id]
        draw_attention_heatmap(clst_levels_spatmap_dir, heat_s_levels_col, org_np_img, None, (slide_id, 'c%d_spat' % sp_clst) )
        print('draw cluster {}\'s levels tiles on spatial map in: {}'.format(str(sp_clst),
                                                                             os.path.join(clst_levels_spatmap_dir,
                                                                                          '{}-{}.png'.format(slide_id, 'c%d_spat' % sp_clst) ) ))
    
            
def plot_demo_spatmap_4_sp_clst(ENV_task, clst_s_spatmap_pkl_name, sp_clst, lobular_label_fname,
                                  top_slides_ids, lowest_slides_ids):
    '''
    Args:
        sp_clst: the label of specific cluster
    '''
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    slide_clst_s_spatmap_dict = load_vis_pkg_from_pkl(heat_store_dir, clst_s_spatmap_pkl_name)
    
    clst_alg_name = clst_s_spatmap_pkl_name[:clst_s_spatmap_pkl_name.find('_unsupervised')] + '-c{}'.format(str(sp_clst))
    top_sp_spatmap_dir = os.path.join(heat_store_dir, clst_alg_name + '-top')
    lowest_sp_spatmap_dir = os.path.join(heat_store_dir, clst_alg_name + '-lowest')
    
    if not os.path.exists(top_sp_spatmap_dir):
        os.makedirs(top_sp_spatmap_dir)
        print('create file dir {}'.format(top_sp_spatmap_dir) )
    if not os.path.exists(lowest_sp_spatmap_dir):
        os.makedirs(lowest_sp_spatmap_dir)
        print('create file dir {}'.format(lowest_sp_spatmap_dir) )
        
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    slide_tile_list_dict = datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE)
    print('> draw spatial maps for slides with top tissue percentage for specific cluster...')
    for slide_id in top_slides_ids:
        org_image, _ = slide_tools.original_slide_and_scaled_pil_image(slide_tile_list_dict[slide_id][0].original_slide_filepath,
                                                                           ENV_task.SCALE_FACTOR, print_opening=False)
        org_np_img = image_tools.pil_to_np_rgb(org_image)
        clst_s_spatmap_into_dict = slide_clst_s_spatmap_dict[slide_id]
        heat_s_clst_col = clst_s_spatmap_into_dict[sp_clst]
        case_id = parse_caseid_from_slideid(slide_id)
        draw_attention_heatmap(top_sp_spatmap_dir, heat_s_clst_col, org_np_img, None, 
                               (slide_id, 'c%d_spat-lobular-%s' % (sp_clst, lobular_label_dict[case_id]) ) )
        print('draw cluster {}\'s spatial map in: {}'.format(str(sp_clst), 
                                                                 os.path.join(top_sp_spatmap_dir, '{}-{}.png'.format(slide_id, 'c%d_spat' % sp_clst) ) ))
    print('> draw spatial maps for slides with lowest tissue percentage for specific cluster...')
    for slide_id in lowest_slides_ids:
        org_image, _ = slide_tools.original_slide_and_scaled_pil_image(slide_tile_list_dict[slide_id][0].original_slide_filepath,
                                                                           ENV_task.SCALE_FACTOR, print_opening=False)
        org_np_img = image_tools.pil_to_np_rgb(org_image)
        clst_s_spatmap_into_dict = slide_clst_s_spatmap_dict[slide_id]
        heat_s_clst_col = clst_s_spatmap_into_dict[sp_clst]
        case_id = parse_caseid_from_slideid(slide_id)
        draw_attention_heatmap(lowest_sp_spatmap_dir, heat_s_clst_col, org_np_img, None, 
                               (slide_id, 'c%d_spat-lobular-%s' % (sp_clst, lobular_label_dict[case_id]) ) )
        print('draw cluster {}\'s spatial map in: {}'.format(str(sp_clst), 
                                                                 os.path.join(lowest_sp_spatmap_dir, '{}-{}.png'.format(slide_id, 'c%d_spat' % sp_clst) ) ))
        
def plot_demo_spatmap_4_iso_group(ENV_task, clst_iso_spatmap_pkl_name, sp_clst, lobular_label_fname,
                                    top_iso_slides_ids, lowest_iso_slides_ids):
    '''
    plot the spatial maps for slides with top/lowest number of iso group of sp_clst
    '''
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    slide_iso_s_spatmap_dict = load_vis_pkg_from_pkl(heat_store_dir, clst_iso_spatmap_pkl_name)
    
    clst_alg_name = clst_iso_spatmap_pkl_name[:clst_iso_spatmap_pkl_name.find('_unsupervised')] + '-c{}'.format(str(sp_clst))
    top_nb_iso_dir = os.path.join(heat_store_dir, clst_alg_name + '-top')
    lowest_nb_iso_dir = os.path.join(heat_store_dir, clst_alg_name + '-lowest')
    
    if not os.path.exists(top_nb_iso_dir):
        os.makedirs(top_nb_iso_dir)
        print('create file dir {}'.format(top_nb_iso_dir) )
    if not os.path.exists(lowest_nb_iso_dir):
        os.makedirs(lowest_nb_iso_dir)
        print('create file dir {}'.format(lowest_nb_iso_dir) )
        
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    slide_tile_list_dict = datasets.load_slides_tileslist(ENV_task, for_train=ENV_task.DEBUG_MODE)
    print('> draw spatial maps for slides with top number of iso tiles for specific cluster...')
    for slide_id in top_iso_slides_ids:
        org_image, _ = slide_tools.original_slide_and_scaled_pil_image(slide_tile_list_dict[slide_id][0].original_slide_filepath,
                                                                       ENV_task.SCALE_FACTOR, print_opening=False)
        org_np_img = image_tools.pil_to_np_rgb(org_image)
        heat_s_iso_col = slide_iso_s_spatmap_dict[slide_id]
        case_id = parse_caseid_from_slideid(slide_id)
        draw_attention_heatmap(top_nb_iso_dir, heat_s_iso_col, org_np_img, None, 
                               (slide_id, 'c%d_iso-lobular-%s' % (sp_clst, lobular_label_dict[case_id]) ) )
        print('draw cluster {}\'s spatial map in: {}'.format(str(sp_clst), 
                                                                 os.path.join(top_nb_iso_dir, '{}-{}.png'.format(slide_id, 'c%d_iso' % sp_clst) ) ))
    print('> draw spatial maps for slides with lowest number of iso tiles for specific cluster...')
    for slide_id in lowest_iso_slides_ids:
        org_image, _ = slide_tools.original_slide_and_scaled_pil_image(slide_tile_list_dict[slide_id][0].original_slide_filepath,
                                                                       ENV_task.SCALE_FACTOR, print_opening=False)
        org_np_img = image_tools.pil_to_np_rgb(org_image)
        heat_s_iso_col = slide_iso_s_spatmap_dict[slide_id]
        case_id = parse_caseid_from_slideid(slide_id)
        draw_attention_heatmap(lowest_nb_iso_dir, heat_s_iso_col, org_np_img, None, 
                               (slide_id, 'c%d_iso-lobular-%s' % (sp_clst, lobular_label_dict[case_id]) ) )
        print('draw cluster {}\'s iso spatial map in: {}'.format(str(sp_clst),
                                                                 os.path.join(lowest_nb_iso_dir, '{}-{}.png'.format(slide_id, 'c%d_iso' % sp_clst) ) ))

            
def print_slide_tis_pct(ENV_task, tis_pct_pkl_name, query_slide_id):
    '''
    '''
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    slide_tis_pct_dict = load_vis_pkg_from_pkl(heat_store_dir, tis_pct_pkl_name)
    print('--- for average ---')
    print(slide_tis_pct_dict['avg'])
    # for single queried slide
    if query_slide_id is not None and query_slide_id in slide_tis_pct_dict.keys():
        query_tis_pct_dict = slide_tis_pct_dict[query_slide_id]
        print('--- for slide: {} ---'.format(query_slide_id))
        print(query_tis_pct_dict)
            

''' ---------------------------------------------------------------------------------- '''
    
def _run_plot_clst_scatter(ENV_task, clst_space_pkl_name):
    clst_redu_en_dict = load_vis_pkg_from_pkl(ENV_task.STATISTIC_STORE_DIR, clst_space_pkl_name)
    plot_clst_scatter(clst_redu_en_dict)
    
def _run_plot_slides_clst_spatmap(ENV_task, clst_spatmaps_pkl_name):
    plot_slides_clst_spatmap(ENV_task, clst_spatmaps_pkl_name)
    
def _run_plot_clst_tile_demo(ENV_task, clst_tiledemo_pkl_name):
    plot_clst_tile_demo(ENV_task, clst_tiledemo_pkl_name)
    
def _run_plot_clst_tile_ihcdab_demo(ENV_task, clst_t_dab_demo_pkl_name):
    plot_clst_tile_ihcdab_demo(ENV_task, clst_t_dab_demo_pkl_name)
    
def _run_plot_slides_clst_each_spatmap(ENV_task, clst_s_spatmap_pkl_name):
    plot_slides_clst_each_spatmap(ENV_task, clst_s_spatmap_pkl_name)
    
def _run_plot_slides_iso_spatmap(ENV_task, clst_iso_spatmap_pkl_name, sp_clst):
    plot_slides_iso_gath_spatmap(ENV_task, clst_iso_spatmap_pkl_name, sp_clst)
    
def _run_plot_slides_levels_spatmap(ENV_task, clst_levels_spatmap_pkl_name, sp_clst):
    plot_slides_levels_spatmap(ENV_task, clst_levels_spatmap_pkl_name, sp_clst)

if __name__ == '__main__':
    pass