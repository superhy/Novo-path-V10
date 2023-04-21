'''
@author: Yang Hu
'''

import os

from interpre.plot_clst_stat import plot_lobular_clsts_avg_dist, \
    plot_clsts_avg_dist_in_HV
from interpre.plot_clst_vis import _run_plot_clst_scatter, \
    _run_plot_slides_clst_spatmap, _run_plot_clst_tile_demo, \
    _run_plot_slides_clst_each_spatmap, print_slide_tis_pct, \
    plot_slides_spatmap_4_sp_clst
from interpre.plot_graph import _run_plot_tiles_onehot_nx_graphs, \
    _run_plot_tiles_neb_nx_graphs
from interpre.plot_slide_heat import _plot_draw_scaled_slide_imgs
from interpre.plot_vit_heat import _run_plot_vit_cls_map, \
    _run_plot_vit_heads_map
from interpre.prep_clst_vis import top_pct_slides_4_sp_clst
from support import env_flinc_cd45, env_flinc_he, env_flinc_psr


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



if __name__ == '__main__':
    
    ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
    # ENV_task = env_flinc_he.ENV_FLINC_HE_STEA_C2
#     ENV_task = env_flinc_psr.ENV_FLINC_PSR_FIB_C3

    # task_ids = [21, 22]
    task_ids = [29.1]
    # task_ids = [61, 62]
    # task_ids = [29.1, 29.2]

    if 0 in task_ids:
        _plot_draw_scaled_slide_imgs(ENV_task)
    if 10 in task_ids:
        headsmap_pkl_name = 'headsmap_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pkl'
        _run_plot_vit_heads_map(ENV_task, headsmap_pkl_name)    
    if 11 in task_ids:
        clsmap_pkl_name = 'clsmap_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pkl'
        _run_plot_vit_cls_map(ENV_task, clsmap_pkl_name)
    if 20 in task_ids:
        # clst_space_pkl_name = 'tsne_5000_clst6-res_Kmeans-neb_encode_unsupervised2023-03-02.pkl'
        # clst_space_pkl_name = 'tsne_5000_clst10-res_Kmeans-neb_encode_unsupervised2023-03-03.pkl'
        clst_space_pkl_name = 'tsne_5000_clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl' # clst-6 reg
        _run_plot_clst_scatter(ENV_task, clst_space_pkl_name)
    if 21 in task_ids:
        # clst_spatmaps_pkl_name = 'clst-spat_Kmeans-neb_encode_unsupervised2023-03-02.pkl'
        # clst_spatmaps_pkl_name = 'clst-spat_Kmeans-neb_encode_unsupervised2023-03-03.pkl'
        clst_spatmaps_pkl_name = 'clst-spat_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        _run_plot_slides_clst_spatmap(ENV_task, clst_spatmaps_pkl_name)
    if 22 in task_ids:
        # clst_tiledemo_pkl_name = 'clst-tiledemo_Kmeans-neb_encode_unsupervised2023-03-02.pkl'
        # clst_tiledemo_pkl_name = 'clst-tiledemo_Kmeans-neb_encode_unsupervised2023-03-03.pkl'
        clst_tiledemo_pkl_name = 'clst-tiledemo_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        _run_plot_clst_tile_demo(ENV_task, clst_tiledemo_pkl_name)
    if 23 in task_ids:
        # clst_s_spatmap_pkl_name = 'clst-s-spat_Kmeans-encode_unsupervised2022-11-24.pkl'
        clst_s_spatmap_pkl_name = 'clst-s-spat_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_plot_slides_clst_each_spatmap(ENV_task, clst_s_spatmap_pkl_name)
    if 29.1 in task_ids:
        # tis_pct_pkl_name = 'clst-tis-pct_Kmeans-neb_encode_unsupervised2023-03-02.pkl' # nb_clst=6
        # tis_pct_pkl_name = 'clst-tis-pct_Kmeans-neb_encode_unsupervised2023-03-03.pkl' # nb_clst=10
        tis_pct_pkl_name = 'clst-tis-pct_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        lobular_label_fname = 'CD45_lobular_inflammation_score_bi.csv'
        # query_slide_id = '23910-158_Sl278-C18-CD45'
        plot_lobular_clsts_avg_dist(ENV_task, tis_pct_pkl_name, lobular_label_fname, nb_clst=6)
    if 29.2 in task_ids:
        # tis_pct_pkl_name = 'clst-tis-pct_Kmeans-neb_encode_unsupervised2023-03-02.pkl' # nb_clst=6
        tis_pct_pkl_name = 'clst-tis-pct_Kmeans-neb_encode_unsupervised2023-03-03.pkl' # nb_clst=10
        # query_slide_id = '23910-158_Sl278-C18-CD45'
        plot_clsts_avg_dist_in_HV(ENV_task, tis_pct_pkl_name)
    if 29.3 in task_ids:
        tis_pct_pkl_name = 'clst-tis-pct_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        clst_s_spatmap_pkl_name = ''
        lobular_label_fname = 'CD45_lobular_inflammation_score_bi.csv'
        sp_clst=5
        nb_top=10
        # load top and lowest tissue percentage slides
        top_slides_ids, lowest_slides_ids = top_pct_slides_4_sp_clst(ENV_task, tis_pct_pkl_name, sp_clst, nb_top)
        # plot top and lowest tissue percentage slides for specific cluster
        plot_slides_spatmap_4_sp_clst(ENV_task, clst_s_spatmap_pkl_name, sp_clst, lobular_label_fname,
                                      top_slides_ids, lowest_slides_ids)
        
    if 61 in task_ids:
        adjdict_pkl_name = 'c-2-adjs_o_0.5_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_plot_tiles_onehot_nx_graphs(ENV_task, adjdict_pkl_name)
    if 62 in task_ids:
        adjdict_pkl_name = 'c-2-adjs_x_0.0_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_plot_tiles_neb_nx_graphs(ENV_task, adjdict_pkl_name)
        
        
        
        