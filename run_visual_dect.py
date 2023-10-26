'''
Created on 3 Oct 2023

@author: yang hu
'''

import os
import sys

from interpre.prep_dect_vis import _run_make_topK_attention_heatmap_resnet_P62, \
    _run_make_spatial_sensi_clusters_assims
from run_main import Logger
from support import env_flinc_p62, tools


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"




if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    
    # task_ids = [0]
    task_ids = [1.1]
    task_str = '-' + '-'.join([str(id) for id in task_ids])
    
    log_name = 'visualisation_log-{}-{}.log'.format(ENV_task.TASK_NAME + task_str,
                                                    str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    
    if 0 in task_ids:
        ''' For original attention-based heatmap '''
    if 1.1 in task_ids:
        agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-1_ballooning_score_bi_[114]2023-10-20.pth',
                               'checkpoint_GatedAttPool-g_Pool-2_ballooning_score_bi_[125]2023-10-20.pth',
                               'checkpoint_GatedAttPool-g_Pool-3_ballooning_score_bi_[153]2023-10-20.pth',
                               'checkpoint_GatedAttPool-g_Pool-4_ballooning_score_bi_[99]2023-10-20.pth',
                               'checkpoint_GatedAttPool-g_Pool-7_ballooning_score_bi_[149]2023-10-22.pth']
        
        # agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_score_bi_[157]2023-10-21.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-1_ballooning_score_bi_[200]2023-10-21.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-2_ballooning_score_bi_[200]2023-10-21.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-3_ballooning_score_bi_[200]2023-10-21.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-4_ballooning_score_bi_[200]2023-10-21.pth']
        
        K_ratio = 0.25
        att_thd = 0.25
        boost_rate = 2.0
        # pkg_range = [0, 50]
        pkg_range = None
        cut_left = True
        fills = [3, 4, 5]
        
        _run_make_topK_attention_heatmap_resnet_P62(ENV_task, agt_model_filenames, cut_left,
                                                    K_ratio, att_thd, boost_rate, fills, pkg_range)
    if 1.2 in task_ids:
        clustering_pkl_name = ''
        assimilate_pkl_name = ''
        sp_clsts = []
        cut_left = True
        
        _run_make_spatial_sensi_clusters_assims(ENV_task, clustering_pkl_name, assimilate_pkl_name, 
                                                sp_clsts, cut_left)
        
        
        
        
    