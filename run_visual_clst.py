'''
@author: Yang Hu
'''

import os

from interpre.prep_clst_vis import _run_make_clsuters_space_maps, \
    _run_make_spatial_clusters_on_slides, _run_make_tiles_demo_clusters, \
    _run_make_spatial_each_clusters_on_slides, \
    _run_count_tis_pct_clsts_on_slides, _run_make_spatial_iso_gath_on_slides, \
    _run_make_spatial_levels_on_slides, \
    _run_count_tis_pct_slides_ref_homo_sp_clst
from interpre.prep_vit_graph import _run_make_vit_graph_adj_clusters, \
    _run_make_vit_neb_graph_adj_clusters
from interpre.prep_vit_heat import _run_vit_d6_h8_cls_map_slides, \
    _run_vit_d6_h8_heads_map_slides, _run_vit_d6_h8_cls_heads_map_slides, \
    _run_reg_ass_sp_clst_homotiles_slides
from support import env_flinc_cd45, env_flinc_he, env_flinc_psr, env_flinc_p62


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



if __name__ == '__main__':
    
    # ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
#     ENV_task = env_flinc_he.ENV_FLINC_HE_STEA_C2
#     ENV_task = env_flinc_psr.ENV_FLINC_PSR_FIB_C3

    task_ids = [20, 21, 22, 29]
    # task_ids = [61, 62]
    # task_ids = [20, 21, 22, 29]
    # task_ids = [31]
    
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
        ''' cd45 '''
        # clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2023-03-02.pkl'
        # clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2023-03-02.pkl' # clst-6
        # clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2023-03-03.pkl' # clst-10
        # clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl' # clst-6 reg
        
        ''' p62 '''
        # clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-09-04.pkl' # clst-6 reg
        clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2023-09-18.pkl' # clst-6
        
        _run_make_clsuters_space_maps(ENV_task, clustering_pkl_name, nb_picked=5000)
    if 21 in task_ids:
        ''' cd45 '''
        # clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2023-03-02.pkl'
        # clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2023-03-02.pkl' # clst-6
        # clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2023-03-03.pkl' # clst-10
        # clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl' # clst-6 reg
        
        ''' p62 '''
        # clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-09-04.pkl' # clst-6 reg
        clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2023-09-18.pkl' # clst-6
        
        _run_make_spatial_clusters_on_slides(ENV_task, clustering_pkl_name, keep_org_slide=False)
    if 22 in task_ids:
        ''' cd45 '''
        # clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2023-03-02.pkl'
        # clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2023-03-02.pkl' # clst-6
        # clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2023-03-03.pkl' # clst-10
        # clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl' # clst-6 reg
        
        ''' p62 '''
        # clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-09-04.pkl' # clst-6 reg
        clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2023-09-18.pkl' # clst-6
        
        nb_sample=2000
        _run_make_tiles_demo_clusters(ENV_task, clustering_pkl_name, nb_sample)
    if 23 in task_ids:
        # clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2022-11-24.pkl'
        # clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl' # clst-6 reg
        sp_clst=5
        _run_make_spatial_each_clusters_on_slides(ENV_task, clustering_pkl_name, sp_clst=sp_clst)
    if 24 in task_ids:
        # clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2022-11-24.pkl'
        # clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl' # clst-6 reg
        sp_clst=5
        iso_thd=0.1
        radius=3
        _run_make_spatial_iso_gath_on_slides(ENV_task, clustering_pkl_name,
                                             sp_clst=sp_clst, iso_thd=iso_thd, radius=radius)
    if 25 in task_ids:
        clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl' # clst-6 reg
        sp_clst=5
        radius=3
        _run_make_spatial_levels_on_slides(ENV_task, clustering_pkl_name, sp_clst, radius)
    if 29 in task_ids:
        ''' cd45 '''
        # clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2023-03-02.pkl' # clst-6
        # clustering_pkl_name = 'clst-res_Kmeans-neb_encode_unsupervised2023-03-03.pkl' # clst-10
        # clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl' # clst-6 reg
        
        ''' p62 '''
        # clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-09-04.pkl' # clst-6 reg
        clustering_pkl_name = 'clst-res_Kmeans-encode_unsupervised2023-09-18.pkl' # clst-6
        
        _run_count_tis_pct_clsts_on_slides(ENV_task, clustering_pkl_name)
    if 30 in task_ids:
        clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl' # clst-6 reg
        sp_clst=5
        iso_thd=0.1
        radius=3
        _run_count_tis_pct_slides_ref_homo_sp_clst(ENV_task, clustering_pkl_name,
                                                   sp_clst=sp_clst, iso_thd=iso_thd, radius=radius)
    if 31 in task_ids:
        clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-05.pkl'
        # clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        vit_pt_name = 'checkpoint_ViT-6-8-PT-Dino_unsupervised-16x16[50]2023-01-13.pth'
        reg_vit_pt_name = 'checkpoint_ViT-Region-4-6-PT-Dino_unsupervised-9x9[50]2023-04-06.pth'
        sp_clst=5
        iso_thd=0.1
        centre_ass=False # default using the graph for all points
        # TODO: test it
        _run_reg_ass_sp_clst_homotiles_slides(ENV_task, clustering_pkl_name, sp_clst, iso_thd, 
                                              vit_pt_name, reg_vit_pt_name, centre_ass)
    
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
        
        