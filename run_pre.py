'''
@author: Yang Hu
'''
from support import env_flinc_psr_fib, env_flinc_he_stea
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
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

from wsi import process

from openslide import open_slide


task_ids = [2]

ENV_task = env_flinc_he_stea.ENV_FLINC_HE_STEA_C2
# ENV_task = env_flinc_psr_fib.ENV_FLINC_PSR_FIB
# ENV_task = env_flinc_psr_fib.ENV_FLINC_PSR_FIB_C3


if __name__ == '__main__':
    if 2 in task_ids:
        # fold_suffix_list = ['-0']
        fold_suffix_list = ['-0', '-1', '-2', '-3', '-4']
        # fold_suffix_list = ['-5', '-6', '-7', '-8', '-9']
        for i, fold_suffix in enumerate(fold_suffix_list):
            ENV_fold = copy(ENV_task)
            ENV_fold.refresh_fold_suffix(fold_suffix)
            process.slide_tiles_split_keep_object_cls(ENV_task=ENV_fold)
