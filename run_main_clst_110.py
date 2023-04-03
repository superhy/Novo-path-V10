'''
Created on 17 Nov 2022

@author: Yang Hu
'''
import os
import sys

from models.functions_clustering import _run_keamns_region_ctx_encode_vit_6_8
from run_main import Logger
from support import env_flinc_cd45, tools


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



task_ids = [110]
task_str = '-' + '-'.join([str(id) for id in task_ids])

if __name__ == '__main__':
    
    ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
    
    log_name = 'running_log{}-{}-{}.log'.format(ENV_task.FOLD_SUFFIX,
                                                ENV_task.TASK_NAME + task_str,
                                                str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    
    if 110 in task_ids:
        vit_pt_name = 'checkpoint_ViT-6-8-PT-Dino_unsupervised-16x16[50]2023-01-13.pth'
        reg_vit_pt_name = ''
        # tiles_r_tuples_pkl_name = 'ViT-6-8-encode_2022-11-23.pkl'
        # tiles_r_tuples_pkl_name = 'ViT-6-8-neb_encode_2022-11-27.pkl'
        tiles_r_tuples_pkl_name = None
        
        ctx_type='reg_ass'
        _run_keamns_region_ctx_encode_vit_6_8(ENV_task, vit_pt_name, reg_vit_pt_name,
                                              tiles_r_tuples_pkl_name, ctx_type)