'''
@author: Yang Hu
'''

import os

from interpre.plot_clst_vis import _run_plot_clst_scatter, \
    _run_plot_slides_clst_spatmap, _run_plot_clst_tile_demo, \
    _run_plot_slides_clst_each_spatmap
from interpre.plot_graph import _run_plot_tiles_onehot_nx_graphs
from interpre.plot_slide_heat import _plot_draw_scaled_slide_imgs
from interpre.plot_vit_heat import _run_plot_vit_cls_map, \
    _run_plot_vit_heads_map
from support import env_flinc_cd45, env_flinc_he, env_flinc_psr


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



if __name__ == '__main__':
    
    ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
    # ENV_task = env_flinc_he.ENV_FLINC_HE_STEA_C2
#     ENV_task = env_flinc_psr.ENV_FLINC_PSR_FIB_C3

    # task_ids = [20, 21, 22]
    task_ids = [61]

    if 0 in task_ids:
        _plot_draw_scaled_slide_imgs(ENV_task)
    if 10 in task_ids:
        headsmap_pkl_name = 'headsmap_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pkl'
        _run_plot_vit_heads_map(ENV_task, headsmap_pkl_name)    
    if 11 in task_ids:
        clsmap_pkl_name = 'clsmap_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pkl'
        _run_plot_vit_cls_map(ENV_task, clsmap_pkl_name)
    if 20 in task_ids:
        # clst_space_pkl_name = 'tsne_all_clst-res_Kmeans-encode_unsupervised2022-11-24.pkl'
        clst_space_pkl_name = 'tsne_all_clst-res_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_plot_clst_scatter(ENV_task, clst_space_pkl_name)
    if 21 in task_ids:
        # clst_spatmaps_pkl_name = 'clst-spat_Kmeans-encode_unsupervised2022-11-24.pkl'
        clst_spatmaps_pkl_name = 'clst-spat_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_plot_slides_clst_spatmap(ENV_task, clst_spatmaps_pkl_name)
    if 22 in task_ids:
        # clst_tiledemo_pkl_name = 'clst-tiledemo_Kmeans-encode_unsupervised2022-11-24.pkl'
        clst_tiledemo_pkl_name = 'clst-tiledemo_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_plot_clst_tile_demo(ENV_task, clst_tiledemo_pkl_name)
    if 23 in task_ids:
        # clst_s_spatmap_pkl_name = 'clst-s-spat_Kmeans-encode_unsupervised2022-11-24.pkl'
        clst_s_spatmap_pkl_name = 'clst-s-spat_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_plot_slides_clst_each_spatmap(ENV_task, clst_s_spatmap_pkl_name)
        
    if 61 in task_ids:
        adjdict_pkl_name = 'c-2-adjmats_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_plot_tiles_onehot_nx_graphs(ENV_task, adjdict_pkl_name)