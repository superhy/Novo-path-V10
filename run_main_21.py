'''
@author: Yang Hu
'''

import os
import sys

from models.functions_attpool import _run_train_gated_attpool_resnet18,\
    _run_train_gated_attpool_ihcdab_resnet18
from run_main import Logger
from support import tools
from support.env_flinc_he import ENV_FLINC_HE_STEA, ENV_FLINC_HE_STEA_C2, \
    ENV_FLINC_HE_BALL_BI
from support.env_flinc_p62 import ENV_FLINC_P62_BALL_BI, ENV_FLINC_P62_STEA_BI, \
    ENV_FLINC_P62_LOB_BI, ENV_FLINC_P62_BALL_PCT, ENV_FLINC_P62_BALL_PCT_BI
from support.env_flinc_psr import ENV_FLINC_PSR_FIB, ENV_FLINC_PSR_FIB_C3


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



# task_ids = [21]
task_ids = [21.1]
fold_suffix = '-0'
# fold_suffix = '-[5-9]'
# fold_suffix = ENV_task.FOLD_SUFFIX
task_str = '-' + '-'.join([str(lbl) for lbl in task_ids])

if __name__ == '__main__':
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    # ENV_task = ENV_FLINC_HE_STEA_C2
    # ENV_task = ENV_FLINC_PSR_FIB_C3
    # ENV_task = ENV_FLINC_HE_BALL_BI
    # ENV_task = ENV_FLINC_P62_BALL_BI
    
    # ENV_task = ENV_FLINC_P62_STEA_BI
    # ENV_task = ENV_FLINC_P62_LOB_BI
    
    # ENV_task = ENV_FLINC_P62_BALL_PCT
    ENV_task = ENV_FLINC_P62_BALL_PCT_BI

    log_name = 'running_log{}-{}-{}.log'.format(fold_suffix,
                                                ENV_task.TASK_NAME + task_str,
                                                str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    
    if 21 in task_ids:
        folds = ['-0']
        # folds = ['-0', '-1', '-2', '-3', '-4']
        # folds = ['-5', '-6', '-7', '-8', '-9']
        for f in folds:
            ENV_task.refresh_fold_suffix(f)
            _run_train_gated_attpool_resnet18(ENV_task)
    if 21.1 in task_ids:
        folds = ['-0']
        # folds = ['-0', '-1', '-2', '-3', '-4']
        # folds = ['-5', '-6', '-7', '-8', '-9']
        for f in folds:
            ENV_task.refresh_fold_suffix(f)
            _run_train_gated_attpool_ihcdab_resnet18(ENV_task)
        
        
        
        