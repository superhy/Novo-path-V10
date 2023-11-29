'''
@author: Yang Hu
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import sys

from models.functions_enc_pt import _run_pretrain_3_4_t_dino
from run_main import Logger
from support import tools
from support.env_flinc_cd45 import ENV_FLINC_CD45_U





task_ids = [89]
task_str = '-' + '-'.join([str(lbl) for lbl in task_ids])

if __name__ == '__main__':
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    ENV_task = ENV_FLINC_CD45_U
    
    log_name = 'running_log{}-{}-{}.log'.format(ENV_task.FOLD_SUFFIX,
                                                ENV_task.TASK_NAME + task_str,
                                                str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    
    if 89 in task_ids:
        _run_pretrain_3_4_t_dino(ENV_task)