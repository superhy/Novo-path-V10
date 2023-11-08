'''
@author: Yang Hu
'''
'''
task_id:
    0: parse and produce the annotation csv
    1: move the download slides to running floder, remove the un-annotated slides
    2: prepare the training/test sets (pkl)
    3: prepare the training/test sets with tumor/background filter (pkl)
    6: download slides of liver from GTEx
        61: check damaged slides and forced re-download them
'''

from copy import copy
import os

from support import env_flinc_psr, env_flinc_he, env_flinc_cd45, env_flinc_p62
from support.env_flinc_he import ENV_FLINC_HE_BALL_BI
from support.env_flinc_p62 import ENV_FLINC_P62_BALL_BI, ENV_FLINC_P62_STEA_BI,\
    ENV_FLINC_P62_LOB_BI
from support.files import _move_slides_multi_stains
from wsi import process


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"





# task_ids = [1, 2]
# task_ids = [1]
task_ids = [2]

# ENV_task = env_flinc_he.ENV_FLINC_HE_STEA_C2
# ENV_task = env_flinc_psr_fib.ENV_FLINC_PSR_FIB
# ENV_task = env_flinc_psr_fib.ENV_FLINC_PSR_FIB_C3

# ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
# ENV_task = env_flinc_cd45.ENV_FLINC_CD45_REG_PT

ENV_task = env_flinc_p62.ENV_FLINC_P62_U
# ENV_task = env_flinc_p62.ENV_FLINC_P62_REG_PT
# ENV_task = ENV_FLINC_HE_BALL_BI
# ENV_task = ENV_FLINC_P62_BALL_BI

# ENV_task = ENV_FLINC_P62_STEA_BI
# ENV_task = ENV_FLINC_P62_LOB_BI


if __name__ == '__main__':
    if 1 in task_ids:
        _move_slides_multi_stains()
    if 2 in task_ids:
        if ENV_task.TASK_NAME == 'unsupervised':
            process.slide_tiles_split_keep_object_u(ENV_task)
        elif ENV_task.TASK_NAME == 'segmentation':
            pass
        else:
            fold_suffix_list = ['-0']
            # fold_suffix_list = ['-0', '-1', '-2', '-3', '-4',
            #                     '-5', '-6', '-7', '-8', '-9']
            # fold_suffix_list = ['-0', '-1', '-2', '-3', '-4']
            # fold_suffix_list = ['-5', '-6', '-7', '-8', '-9']
            for i, fold_suffix in enumerate(fold_suffix_list):
                ENV_fold = copy(ENV_task)
                ENV_fold.refresh_fold_suffix(fold_suffix)
                process.slide_tiles_split_keep_object_cls(ENV_task=ENV_fold)
