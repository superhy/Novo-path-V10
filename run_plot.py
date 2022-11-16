'''
@author: Yang Hu
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

from interpre.plot_vit_heat import _run_plot_vit_cls_map, \
    _run_plot_vit_heads_map
from support import env_flinc_cd45, env_flinc_he, env_flinc_psr


if __name__ == '__main__':
    
    ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
#     ENV_task = env_flinc_he.ENV_FLINC_HE_STEA_C2
#     ENV_task = env_flinc_psr.ENV_FLINC_PSR_FIB_C3

    task_ids = [11]
    
    if 11 in task_ids:
        clsmap_pkl_name = 'clsmap_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pkl'
        _run_plot_vit_cls_map(ENV_task, clsmap_pkl_name)
    if 12 in task_ids:
        headsmap_pkl_name = 'headsmap_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pkl'
        _run_plot_vit_heads_map(ENV_task, headsmap_pkl_name)