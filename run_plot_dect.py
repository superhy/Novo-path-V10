'''
@author: Yang Hu
'''

import os

from interpre.plot_dect_vis import _plot_topK_attention_heatmaps
from support import env_flinc_p62

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U

    # task_ids = [0]
    task_ids = [1.1]
    
    if 0 in task_ids:
        pass
    if 1.1 in task_ids:
        heatmap_pkl_name = 'topK_map_GatedAttPool-g_Pool-1_ballooning_score_bi_[114]2023-10-20.pkl'
        ENV_annotation = env_flinc_p62.ENV_FLINC_P62_BALL_BI
        
        _plot_topK_attention_heatmaps(ENV_task, ENV_annotation, heatmap_pkl_name)
    
    