'''
Created on 17 Nov 2022

@author: Yang Hu
'''
import os
import sys

from models.functions_clustering import _run_keamns_region_ctx_encode_vit_6_8, \
    _run_kmeans_attKtiles_encode_resnet18, _run_tiles_assimilate_encode_resnet18, \
    load_clustering_pkg_from_pkl, _run_kmeans_act_K_tiles_encode_resnet18, \
    _run_hierarchical_kmeans_encode_same, \
    _run_kmeans_filter_act_K_tiles_encode_resnet18
from run_main import Logger
from support import env_flinc_cd45, tools, env_flinc_p62
from support.env_flinc_p62 import ENV_FLINC_P62_STEA_BI, ENV_FLINC_P62_LOB_BI


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


# 2023.11.06 on PC test of 58 slides
task_ids = [125]
task_str = '-' + '-'.join([str(id) for id in task_ids])

if __name__ == '__main__':
    
    # ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    ENV_annotation = env_flinc_p62.ENV_FLINC_P62_BALL_BI
    
    log_name = 'clustering_log{}-{}-{}.log'.format(ENV_task.FOLD_SUFFIX,
                                                ENV_task.TASK_NAME + task_str,
                                                str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    clustering_res_pkg = None
    
    if 120 in task_ids:
        # p62
        # agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-1_ballooning_score_bi_[114]2023-10-20.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-2_ballooning_score_bi_[125]2023-10-20.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-3_ballooning_score_bi_[153]2023-10-20.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-4_ballooning_score_bi_[99]2023-10-20.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-7_ballooning_score_bi_[149]2023-10-22.pth']
        
        agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_score_bi_[159]2023-10-02.pth']
        
        K_ratio = 0.25
        att_thd =  0.25
        fills = [3, 3, 3, 4]
        manu_n_clusters=4
        
        # tiles_r_tuples_pkl_name = 'ViT-6-8-encode_2022-11-23.pkl'
        # tiles_r_tuples_pkl_name = 'ViT-6-8-neb_encode_2022-11-27.pkl'
        tiles_r_tuples_pkl_name = None
        
        clustering_res_pkg = _run_kmeans_attKtiles_encode_resnet18(ENV_task, ENV_annotation, 
                                                                   agt_model_filenames, 
                                                                   K_ratio, att_thd, fills,
                                                                   manu_n_clusters=manu_n_clusters,
                                                                   tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name)
    if 121 in task_ids:
        tile_net_filenames = ['']
        
        K_ratio = 0.75
        act_thd =  0.36
        fills = [3, 3, 3, 4]
        manu_n_clusters=4
        
        # tiles_r_tuples_pkl_name = 'ViT-6-8-encode_2022-11-23.pkl'
        # tiles_r_tuples_pkl_name = 'ViT-6-8-neb_encode_2022-11-27.pkl'
        tiles_r_tuples_pkl_name = None
        
        clustering_res_pkg = _run_kmeans_act_K_tiles_encode_resnet18(ENV_task, ENV_annotation, 
                                                                     tile_net_filenames, 
                                                                     K_ratio, act_thd, fills,
                                                                     manu_n_clusters=manu_n_clusters,
                                                                     tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name)
    if 121.1 in task_ids:
        tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pth']
        K_ratio = 0.75
        act_thd = 0.36
        
        ENV_neg_annotations = [ENV_FLINC_P62_STEA_BI, ENV_FLINC_P62_LOB_BI]
        stea_model_filenames = ['']
        lob_model_filenames = ['']
        neg_t_filenames_list = [stea_model_filenames, lob_model_filenames]
        stea_K_ratio, stea_att_thd = 0.75, 0.4
        lob_K_ratio, lob_att_thd = 0.75, 0.4
        fills = [3, 3, 3, 4]
        neg_parames = [(stea_K_ratio, stea_att_thd), (lob_K_ratio, lob_att_thd)]
        
        manu_n_clusters=4
        tiles_r_tuples_pkl_name = None
        
        # TODO: clustering alg name with encoder information
        clustering_res_pkg = _run_kmeans_filter_act_K_tiles_encode_resnet18(ENV_task, ENV_annotation, ENV_neg_annotations,
                                                                            tile_net_filenames, neg_t_filenames_list,
                                                                            K_ratio, act_thd, fills, neg_parames,
                                                                            manu_n_clusters=manu_n_clusters, 
                                                                            tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name)
        
    if 123 in task_ids:
        init_clst_pkl_name = 'clst-res_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl' # 58 on PC n4
        silhouette_thd = 0.5
        max_rounds = 2
        
        hierarchical_res_pkg = _run_hierarchical_kmeans_encode_same(ENV_task, init_clst_pkl_name,
                                                                    silhouette_thd, max_rounds)
    
    if 129 in task_ids:
        # clustering_pkl_name = 'clst-res_Kmeans-ResNet18-encode_unsupervised2023-10-26.pkl' # after attention
        clustering_pkl_name = 'clst-res_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl' # 58 on PC n4
        print('need to re-load clustering results first!')
            
        sensitive_labels = [0]
        assim_ratio = 0.01
        fills=[5, 6]
        exc_clustered=False
        _run_tiles_assimilate_encode_resnet18(ENV_task, clustering_pkl_name, 
                                              sensitive_labels, exc_clustered, assim_ratio, fills)
        
        
        
        