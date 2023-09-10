'''
Created on 17 Nov 2022

@author: Yang Hu
'''
import os
import sys

from models.functions_clustering import _run_kmeans_encode_vit_6_8
from run_main import Logger
from support import env_flinc_cd45, tools, env_flinc_p62

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



task_ids = [100]
task_str = '-' + '-'.join([str(id) for id in task_ids])

if __name__ == '__main__':
    
    # ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    
    log_name = 'running_log{}-{}-{}.log'.format(ENV_task.FOLD_SUFFIX,
                                                ENV_task.TASK_NAME + task_str,
                                                str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    
    if 100 in task_ids:
        # cd45
        # vit_pt_name = 'checkpoint_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pth'
        # vit_pt_name = 'checkpoint_ViT-6-8-PT-Dino_unsupervised-16x16[50]2023-01-13.pth'
        
        # p62
        vit_pt_name = 'checkpoint_ViT-6-8-PT-Dino_unsupervised-16x16[50]2023-09-02.pth'
        
        # tiles_r_tuples_pkl_name = 'ViT-6-8-encode_2022-11-23.pkl'
        tiles_r_tuples_pkl_name = None
        _run_kmeans_encode_vit_6_8(ENV_task, vit_pt_name, tiles_r_tuples_pkl_name)