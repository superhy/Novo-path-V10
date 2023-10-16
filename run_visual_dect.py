'''
Created on 3 Oct 2023

@author: yang hu
'''

import os

from interpre.prep_dect_vis import _run_make_topK_attention_heatmap_resnet_P62
from support import env_flinc_p62

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"




if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    
    # task_ids = [0]
    task_ids = [1.1]
    
    if 0 in task_ids:
        pass
    if 1.1 in task_ids:
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
        K_ratio=0.25
        att_thd=0.3
        boost_rate=2.0
        
        _run_make_topK_attention_heatmap_resnet_P62(ENV_task, agt_model_filenames,
                                                    K_ratio, att_thd, boost_rate)
    