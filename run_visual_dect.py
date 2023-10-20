'''
Created on 3 Oct 2023

@author: yang hu
'''

import os
import sys

from interpre.prep_dect_vis import _run_make_topK_attention_heatmap_resnet_P62
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
        ''' For sensitive tiles demo visualisation '''
    if 1.1 in task_ids:
        agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_score_bi_[80]2023-10-20.pth',
                               'checkpoint_GatedAttPool-g_Pool-1_ballooning_score_bi_[114]2023-10-20.pth',
                               'checkpoint_GatedAttPool-g_Pool-2_ballooning_score_bi_[125]2023-10-20.pth']
                               # 'checkpoint_GatedAttPool-g_Pool_-3_ballooning_score_bi_[246]2023-10-17.pth',
                               # 'checkpoint_GatedAttPool-g_Pool_-4_ballooning_score_bi_[269]2023-10-17.pth']
        
        # agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_score_bi_[80]2023-10-20.pth']
        # agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-1_ballooning_score_bi_[114]2023-10-20.pth']
        # agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-2_ballooning_score_bi_[125]2023-10-20.pth']
        
        # agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_score_bi_[159]2023-10-02.pth']
        
        K_ratio=0.3
        att_thd=0.25
        boost_rate=2.0
        # pkg_range = [0, 50]
        pkg_range = None
        cut_left = True
        
        _run_make_topK_attention_heatmap_resnet_P62(ENV_task, agt_model_filenames, cut_left,
                                                    K_ratio, att_thd, boost_rate, pkg_range)
        
        
        
        
    