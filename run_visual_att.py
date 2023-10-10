'''
Created on 3 Oct 2023

@author: yang hu
'''

import os

from interpre.prep_att_vis import _run_make_topK_attention_heatmap_resnet_P62
from support import env_flinc_p62

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"




if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    
    # task_ids = [0]
    task_ids = [1.1]
    
    if 0 in task_ids:
        pass
    if 1.1 in task_ids:
        agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_score_bi_[159]2023-10-02.pth']
        K_ratio=0.25
        att_thd=0.3
        boost_rate=2.0
        
        _run_make_topK_attention_heatmap_resnet_P62(ENV_task, agt_model_filenames,
                                                    K_ratio, att_thd, boost_rate)
    