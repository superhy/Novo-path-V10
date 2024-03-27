'''
Created on 24 Mar 2024

@author: super
'''
import os
import sys

from interpre.plot_stat_vis import _plot_c_group_props_by_henning_frac, \
    _plot_c_gp_agts_corr_henning_frac
from run_main import Logger
from support import env_flinc_p62, tools


if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    
    # task_ids = [1]
    task_ids = [2]
    
    task_str = '-' + '-'.join([str(lbl) for lbl in task_ids])
    
    log_name = 'statistic_log-{}-{}.log'.format(ENV_task.TASK_NAME + task_str,
                                                    str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    
    if 1 in task_ids:
        slide_tile_label_dict_filename = 'hiera-res-r5_Kmeans-ResNet18-encode-dab_unsupervised2024-03-01.pkl'
        clst_gps = [['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                     '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                     '0_1_0_1_0_1', '0_1_0_1_1_0'],
                    ['0_0_1_1_0_0', '0_0_1_1_1_1'],
                    ['1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0'],
                    ['3_0_0_0_0_1']] # Mar 2024, on ihc-dab, r5
        _plot_c_group_props_by_henning_frac(ENV_task, slide_tile_label_dict_filename, clst_gps)
        
    if 2 in task_ids:
        c_gp_aggregation_filename = 'agt_c-gps4_rad5.pkl'
        _plot_c_gp_agts_corr_henning_frac(ENV_task, c_gp_aggregation_filename)