'''
@author: Yang Hu
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


from visual.prep_vit_heat import _run_vit_d6_h8_cls_map_slides, \
    _run_vit_d6_h8_heads_map_slides
from support import env_flinc_cd45, env_flinc_he, env_flinc_psr


if __name__ == '__main__':
    
    ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
#     ENV_task = env_flinc_he.ENV_FLINC_HE_STEA_C2
#     ENV_task = env_flinc_psr.ENV_FLINC_PSR_FIB_C3

    task_ids = [11, 12]
    
    if 11 in task_ids:
        vit_model_filename = 'checkpoint_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pth'
        _run_vit_d6_h8_cls_map_slides(ENV_task, vit_model_filename)
    if 12 in task_ids:
        vit_model_filename = 'checkpoint_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pth'
        _run_vit_d6_h8_heads_map_slides(ENV_task, vit_model_filename)