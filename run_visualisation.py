'''
@author: Yang Hu
'''

import os

from interpre.prep_clst_vis import _run_make_clsuters_space_maps, \
    _run_make_spatial_clusters_on_slides, _run_make_tiles_demo_clusters, \
    _run_make_spatial_each_clusters_on_slides, \
    _run_count_tis_pct_clsts_on_slides
from interpre.prep_vit_graph import _run_make_vit_graph_adj_clusters, \
    _run_make_vit_neb_graph_adj_clusters
from interpre.prep_vit_heat import _run_vit_d6_h8_cls_map_slides, \
    _run_vit_d6_h8_heads_map_slides, _run_vit_d6_h8_cls_heads_map_slides
from support import env_flinc_cd45, env_flinc_he, env_flinc_psr


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



if __name__ == '__main__':
    
    ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
#     ENV_task = env_flinc_he.ENV_FLINC_HE_STEA_C2
#     ENV_task = env_flinc_psr.ENV_FLINC_PSR_FIB_C3

    # task_ids = [20, 21, 22]
    task_ids = [20]
    # task_ids = [61, 62]
    # task_ids = [29]
    
    
    if 10 in task_ids:
        vit_model_filename = 'checkpoint_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pth'
        _run_vit_d6_h8_cls_heads_map_slides(ENV_task, vit_model_filename)
    if 11 in task_ids:
        vit_model_filename = 'checkpoint_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pth'
        _run_vit_d6_h8_cls_map_slides(ENV_task, vit_model_filename)
    if 12 in task_ids:
        vit_model_filename = 'checkpoint_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pth'
        _run_vit_d6_h8_heads_map_slides(ENV_task, vit_model_filename)
    if 20 in task_ids:
        # clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2023-03-02.pkl'
        clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2023-03-02.pkl'
        _run_make_clsuters_space_maps(ENV_task, clustering_pkl_name, nb_picked=5000)
    if 21 in task_ids:
        # clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2023-03-02.pkl'
        clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2023-03-02.pkl'
        _run_make_spatial_clusters_on_slides(ENV_task, clustering_pkl_name, keep_org_slide=False)
    if 22 in task_ids:
        # clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2023-03-02.pkl'
        clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2023-03-02.pkl'
        nb_sample=2000
        _run_make_tiles_demo_clusters(ENV_task, clustering_pkl_name, nb_sample)
    if 23 in task_ids:
        # clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2022-11-24.pkl'
        clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_make_spatial_each_clusters_on_slides(ENV_task, clustering_pkl_name)
    if 29 in task_ids:
        clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_count_tis_pct_clsts_on_slides(ENV_task, clustering_pkl_name)
    
    load_tile_slideids = None
    if 61 in task_ids:
        clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        # vit_model_filename = 'checkpoint_ViT-6-8-PT-Dino_unsupervised-16x16[50]2023-01-13.pth'
        vit_model_filename = 'checkpoint_ViT-4-6-PT-Dino_unsupervised-16x16[50]2023-01-17.pth'
        load_tile_slideids = _run_make_vit_graph_adj_clusters(ENV_task, clustering_pkl_name, 
                                                              vit_model_filename, load_tile_slideids=None,
                                                              clst_id=2, nb_sample=200, edge_th=0.5)
    if 62 in task_ids:
        clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        vit_model_filename = 'checkpoint_ViT-4-6-PT-Dino_unsupervised-16x16[50]2023-01-17.pth'
        load_tile_slideids = _run_make_vit_neb_graph_adj_clusters(ENV_task, clustering_pkl_name,
                                             vit_model_filename, load_tile_slideids=load_tile_slideids,
                                             clst_id=2, nb_sample=200, edge_th=0.0)
        
        