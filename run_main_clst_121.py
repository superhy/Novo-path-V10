'''
Created on 17 Nov 2022

@author: Yang Hu
'''
import os
import sys

from models.functions_clustering import _run_keamns_region_ctx_encode_vit_6_8, \
    _run_kmeans_attKtiles_encode_resnet18, _run_tiles_assimilate_encode_resnet18
from run_main import Logger
from support import env_flinc_cd45, tools, env_flinc_p62


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



task_ids = [121]
task_str = '-' + '-'.join([str(id) for id in task_ids])

if __name__ == '__main__':
    
    # ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    ENV_annotation = env_flinc_p62.ENV_FLINC_P62_BALL_BI
    
    log_name = 'running_log{}-{}-{}.log'.format(ENV_task.FOLD_SUFFIX,
                                                ENV_task.TASK_NAME + task_str,
                                                str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    clustering_res_pkg = None
    
    if 121 in task_ids:
        # p62
        agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool_-0_ballooning_score_bi_[126]2023-10-08.pth',
                               'checkpoint_GatedAttPool-g_Pool_-1_ballooning_score_bi_[13]2023-10-08.pth',
                               'checkpoint_GatedAttPool-g_Pool_-2_ballooning_score_bi_[41]2023-10-08.pth',
                               'checkpoint_GatedAttPool-g_Pool_-3_ballooning_score_bi_[187]2023-10-08.pth',
                               'checkpoint_GatedAttPool-g_Pool_-4_ballooning_score_bi_[20]2023-10-08.pth',
                               'checkpoint_GatedAttPool-g_Pool_-5_ballooning_score_bi_[10]2023-10-14.pth',
                               'checkpoint_GatedAttPool-g_Pool_-6_ballooning_score_bi_[10]2023-10-12.pth',
                               'checkpoint_GatedAttPool-g_Pool_-7_ballooning_score_bi_[360]2023-10-12.pth',
                               'checkpoint_GatedAttPool-g_Pool_-8_ballooning_score_bi_[335]2023-10-12.pth',
                               'checkpoint_GatedAttPool-g_Pool_-9_ballooning_score_bi_[21]2023-10-13.pth']
        
        '''
        agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool_-0_ballooning_score_bi_[126]2023-10-08.pth',
                               'checkpoint_GatedAttPool-g_Pool_-1_ballooning_score_bi_[13]2023-10-08.pth']
        '''
        
        K_ratio = 0.25
        att_thd =  0.3
        fill_void = True
        
        # tiles_r_tuples_pkl_name = 'ViT-6-8-encode_2022-11-23.pkl'
        # tiles_r_tuples_pkl_name = 'ViT-6-8-neb_encode_2022-11-27.pkl'
        tiles_r_tuples_pkl_name = None
        
        clustering_res_pkg = _run_kmeans_attKtiles_encode_resnet18(ENV_task, ENV_annotation, 
                                                                   agt_model_filenames, 
                                                                   K_ratio, att_thd, fill_void,
                                                                   tiles_r_tuples_pkl_name)
    if 121.1 in task_ids:
        if clustering_res_pkg is None:
            print('! need to load clustering results first')
        else:
            sensitive_labels = []
            assim_thd = 0.1
            fill_void = True
            _run_tiles_assimilate_encode_resnet18(ENV_task, clustering_res_pkg, 
                                                  sensitive_labels, assim_thd, fill_void)
        
        
        
        