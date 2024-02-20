'''
Created on 17 Nov 2022

@author: Yang Hu
'''
import os
import sys

from interpre.prep_clst_vis import pick_clusters_by_prefix
from models.functions_clustering import _run_keamns_region_ctx_encode_vit_6_8, \
    _run_kmeans_attKtiles_encode_resnet18, _run_tiles_assimilate_centre_encode_resnet18, \
    load_clustering_pkg_from_pkl, _run_kmeans_act_K_tiles_encode_resnet18, \
    _run_hierarchical_kmeans_encode_same, \
    _run_kmeans_filter_act_K_tiles_encode_resnet18, \
    _run_tiles_assimilate_each_encode_resnet18
from run_main import Logger
from support import env_flinc_cd45, tools, env_flinc_p62
from support.env_flinc_p62 import ENV_FLINC_P62_STEA_BI, ENV_FLINC_P62_LOB_BI
from support.metadata import get_slide_ids_with_b_cpt_lower_than_thd
from support.tools import Time


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


# 2023.11.06 on PC test of 58 slides
task_ids = [120]
# task_ids = [129]
# task_ids = [129.5]

task_str = '-' + '-'.join([str(lbl) for lbl in task_ids])

if __name__ == '__main__':
    
    # ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    # ENV_annotation = env_flinc_p62.ENV_FLINC_P62_BALL_BI
    ENV_annotation = env_flinc_p62.ENV_FLINC_P62_BALL_PCT_BI # for 120. Feb 19, 2024
    
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
        
        # agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_score_bi_[159]2023-10-02.pth']
        
        agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_pct_lbl_bi_[33]2024-02-19.pth']
        
        K_ratio = 0.25
        att_thd =  0.25
        fills = [3, 3, 4]
        manu_n_clusters=4
        
        # tiles_r_tuples_pkl_name = 'ViT-6-8-encode_2022-11-23.pkl'
        # tiles_r_tuples_pkl_name = 'ViT-6-8-neb_encode_2022-11-27.pkl'
        tiles_r_tuples_pkl_name = None
        
        ''' filtered out slides, actually here is case_id '''
        # filter_out_slide_keys = []
        Low_pct_slide_keys = get_slide_ids_with_b_cpt_lower_than_thd(os.path.join(ENV_task.META_FOLDER, 'P62_ballooning_pct.csv'),
                                                                     threshold=0.05)
        HV_slide_keys = ['Sl240', 'Sl241', 'Sl242', 'Sl243', 'Sl244',
                         'Sl245', 'Sl246', 'Sl247', 'Sl248', 'Sl249', 'Sl250'] # these are health volunteers
        filter_out_slide_keys = HV_slide_keys
        # filter_out_slide_keys = list(set(Low_pct_slide_keys + HV_slide_keys))
        
        
        clustering_res_pkg = _run_kmeans_attKtiles_encode_resnet18(ENV_task, ENV_annotation, 
                                                                   agt_model_filenames,
                                                                   K_ratio, att_thd, fills,
                                                                   filter_out_slide_keys=filter_out_slide_keys,
                                                                   manu_n_clusters=manu_n_clusters,
                                                                   tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name)
    if 121 in task_ids:
        ''' @temporarily deprecated '''
        tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pth']
        # tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_steatosis_score_bi_[3]2023-11-24.pth']
        
        K_ratio = 0.5
        act_thd =  0.36
        act_thd =  0.4
        fills = [3, 3, 3, 3, 3]
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
        ''' @temporarily deprecated '''
        tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pth']
        K_ratio = 0.75
        act_thd = 0.36
        
        stea_model_filenames = ['checkpoint_ResNet18-TK_MIL-0_steatosis_score_bi_[5]2023-11-20.pth']
        lob_model_filenames = ['checkpoint_ResNet18-TK_MIL-0_lobular_inflammation_score_bi_[5]2023-11-21.pth']
        neg_t_filenames_list = [stea_model_filenames, lob_model_filenames]
        stea_K_ratio, stea_att_thd = 0.75, 0.4
        lob_K_ratio, lob_att_thd = 0.75, 0.4
        fills = [3, 3, 3, 3, 3]
        neg_parames = [(stea_K_ratio, stea_att_thd), (lob_K_ratio, lob_att_thd)]
        
        manu_n_clusters=4
        tiles_r_tuples_pkl_name = None
        
        # TODO: clustering alg name with encoder information
        clustering_res_pkg = _run_kmeans_filter_act_K_tiles_encode_resnet18(ENV_task,
                                                                            tile_net_filenames, neg_t_filenames_list,
                                                                            K_ratio, act_thd, fills, neg_parames,
                                                                            manu_n_clusters=manu_n_clusters, 
                                                                            tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name)
        
    if 123 in task_ids:
        init_clst_pkl_name = 'clst-res_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl' # 251 on server n4; Nov 2023
        silhouette_thd = 0.02
        max_rounds = 4
        
        hierarchical_res_pkg = _run_hierarchical_kmeans_encode_same(ENV_task, init_clst_pkl_name,
                                                                    silhouette_thd, max_rounds)
    
    if 129 in task_ids:
        # clustering_pkl_name = 'clst-res_Kmeans-ResNet18-encode_unsupervised2023-10-26.pkl' # after attention
        # clustering_pkl_name = 'clst-res_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl' # 58 on PC n4
        clustering_pkl_name = 'hiera-res_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl'
        tile_net_filename = 'checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pth' # Dec 2023
            
        cluster_groups = ['0_1_0_0_0', '1_1_0_0_0', '1_1_0_0_1', '1_1_1_0_1',
                            '2_1_0_0_0', '2_1_0_0_1', '2_1_1_0_0', '2_1_1_0_1', 
                            '2_1_1_1_0']
        sp_clsts = pick_clusters_by_prefix(ENV_task, clustering_pkl_name, cluster_groups)
        
        str_time = Time().date
        record_t_tuples_name=f'remain_tiles_encode_{len(cluster_groups)}c-{str_time}.pkl'
        load_t_tuples_name=None
        if load_t_tuples_name is None:
            print('need to re-load clustering results first!')
        
        ''' rough '''
        # assim_ratio = 0.01
        # fills=[3, 3, 3, 4]
        ''' mid '''
        # assim_ratio = 0.002
        # fills=[3, 3, 3, 4]
        ''' closest '''
        assim_ratio = 0.001
        fills=[3, 3, 3, 4]
        
        exc_clustered=False
        _run_tiles_assimilate_centre_encode_resnet18(ENV_task, clustering_pkl_name, sp_clsts, 
                                              tile_net_filename=tile_net_filename,
                                              exc_clustered=exc_clustered, 
                                              assim_ratio=assim_ratio, fills=fills,
                                              record_t_tuples_name=record_t_tuples_name,
                                              load_t_tuples_name=load_t_tuples_name)
    if 129.5 in task_ids:
        clustering_pkl_name = 'hiera-res_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl'
        tile_net_filename = 'checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pth' # Dec 2023
        
        cluster_groups = ['0_1_0_0_0', '1_1_0_0_0', '1_1_0_0_1', '1_1_1_0_1',
                            '2_1_0_0_0', '2_1_0_0_1', '2_1_1_0_0', '2_1_1_0_1', 
                            '2_1_1_1_0']
        sp_clsts = pick_clusters_by_prefix(ENV_task, clustering_pkl_name, cluster_groups)
        
        str_time = Time().date
        record_t_tuples_name=f'remain_tiles_encode_{len(cluster_groups)}c-{str_time}.pkl'
        load_t_tuples_name=None
        if load_t_tuples_name is None:
            print('need to re-load clustering results first!')
        
        assim_ratio = 0.001
        fills=[3, 3, 3, 4]
        
        exc_clustered=False
        _run_tiles_assimilate_each_encode_resnet18(ENV_task, clustering_pkl_name, sp_clsts, 
                                              tile_net_filename=tile_net_filename,
                                              exc_clustered=exc_clustered, 
                                              assim_ratio=assim_ratio, fills=fills,
                                              record_t_tuples_name=record_t_tuples_name,
                                              load_t_tuples_name=load_t_tuples_name)
        
        
        
        