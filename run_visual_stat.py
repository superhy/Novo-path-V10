'''
Created on 24 Mar 2024

@author: super
'''
import os
import sys

from interpre.prep_stat_vis import _run_localise_k_clsts_on_all_slides, \
    _run_agt_of_clst_gps_on_all_slides_continue, \
    _run_agt_of_sp_clsts_on_all_slides_continue, \
    _run_agt_of_clst_gps_on_all_slides, _run_agt_of_sp_clsts_on_all_slides
from run_main import Logger
from support import env_flinc_p62, tools


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    
    # task_ids = [1]
    task_ids = [2, 2.05, 2.1]
    # task_ids = [2.05]
    
    task_str = '-' + '-'.join([str(lbl) for lbl in task_ids])
    
    log_name = 'statistic_log-{}-{}.log'.format(ENV_task.TASK_NAME + task_str,
                                                    str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    
    if 1 in task_ids:
        ''' --- cluster 1by1, ihc-dab r5 assimilate --- '''
        clustering_pkl_name = 'hiera-res-r5_Kmeans-ResNet18-encode-dab_unsupervised2024-03-01.pkl' # Feb 28 2024, ihc-dab, r5
        c_assimilate_pkl_name = 'assimilate_1by1_ft_ass-encode-ResNet18-dab_unsupervised2024-03-15.pkl' # 15th Mar 2024, ihc-dab r5, 0.002[3,3]
        # c_assimilate_pkl_name = 'assimilate_1by1_ft_ass-encode-ResNet18-dab_unsupervised2024-03-27.pkl' # 27th Mar 2024, ihc-dab r5, 0.004[3,3] with HV
        
        sp_clsts = ['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                    '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                    '0_1_0_1_0_1', '0_1_0_1_1_0', # simi color - 1
                    '0_0_1_1_0_0', '0_0_1_1_1_1', # simi color - 2
                    '1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0', # simi color - 3
                    '3_0_0_0_0_1' # simi color - 4
                ] # Mar 2024, on ihc-dab, r5, not grouped, all use diff (but similar) colors
        
        _ = _run_localise_k_clsts_on_all_slides(ENV_task, clustering_pkl_name, c_assimilate_pkl_name, sp_clsts)
        
    if 2 in task_ids:
        '''
        only running _run_agt_of_clst_gps_on_all_slides     
        '''
        slide_t_label_dict_fname = 'c16-1by1_c-a-local_2024-03-26.pkl' # test on 15th Mar 2024, ihc-dab r5, 0.002[3,3]
        # slide_t_label_dict_fname = 'c16-1by1_c-a-local_2024-03-28.pkl' # 27th Mar 2024, ihc-dab r5, 0.004[3,3]
        clst_gps = [['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                     '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                     '0_1_0_1_0_1', '0_1_0_1_1_0'],
                    ['0_0_1_1_0_0', '0_0_1_1_1_1'],
                    ['1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0'],
                    ['3_0_0_0_0_1']] # Mar 2024, on ihc-dab, r5
        # radius = 3
        radius = 5
        slide_gp_agt_score_dict, name_gps = _run_agt_of_clst_gps_on_all_slides(ENV_task, 
                                                                               slide_t_label_dict_fname, 
                                                                               clst_gps, radius)
    if 2.05 in task_ids:
        '''
        only running _run_agt_of_clst_gps_on_all_slides, put all clsts in one group
        '''
        slide_t_label_dict_fname = 'c16-1by1_c-a-local_2024-03-26.pkl' # test on 15th Mar 2024, ihc-dab r5, 0.002[3,3]
        # slide_t_label_dict_fname = 'c16-1by1_c-a-local_2024-03-28.pkl' # 27th Mar 2024, ihc-dab r5, 0.004[3,3]
        clst_gps = [['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                     '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                     '0_1_0_1_0_1', '0_1_0_1_1_0', 
                     '0_0_1_1_0_0', '0_0_1_1_1_1',
                     '1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0',
                     '3_0_0_0_0_1']] # Mar 2024, on ihc-dab, r5, all in one group
        # radius = 3
        radius = 5
        slide_gp_agt_score_dict, name_gps = _run_agt_of_clst_gps_on_all_slides(ENV_task, 
                                                                               slide_t_label_dict_fname, 
                                                                               clst_gps, radius)
        
    if 2.1 in task_ids:
        '''
        only running _run_agt_of_sp_clsts_on_all_slides 
        '''
        slide_t_label_dict_fname = 'c16-1by1_c-a-local_2024-03-26.pkl' # test on 15th Mar 2024, ihc-dab r5, 0.002[3,3]
        # slide_t_label_dict_fname = 'c16-1by1_c-a-local_2024-03-28.pkl' # 27th Mar 2024, ihc-dab r5, 0.004[3,3]
        sp_clsts = ['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                    '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                    '0_1_0_1_0_1', '0_1_0_1_1_0', # simi color - 1
                    '0_0_1_1_0_0', '0_0_1_1_1_1', # simi color - 2
                    '1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0', # simi color - 3
                    '3_0_0_0_0_1' # simi color - 4
                ] # Mar 2024, on ihc-dab, r5, not grouped, all use diff (but similar) colors
        radius = 5
        slide_spc_agt_score_dict = _run_agt_of_sp_clsts_on_all_slides(ENV_task, 
                                                                      slide_t_label_dict_fname, 
                                                                      sp_clsts, radius)
        
    if 2.2 in task_ids:
        '''
        do 1, then 2
        '''
        
        ''' --- cluster 1by1, ihc-dab r5 assimilate --- '''
        clustering_pkl_name = 'hiera-res-r5_Kmeans-ResNet18-encode-dab_unsupervised2024-03-01.pkl' # Feb 28 2024, ihc-dab, r5
        c_assimilate_pkl_name = 'assimilate_1by1_ft_ass-encode-ResNet18-dab_unsupervised2024-03-15.pkl' # 15th Mar 2024, ihc-dab r5, 0.002[3,3]
        # c_assimilate_pkl_name = 'assimilate_1by1_ft_ass-encode-ResNet18-dab_unsupervised2024-03-27.pkl' # 27th Mar 2024, ihc-dab r5, 0.004[3,3] with HV
        
        sp_clsts = ['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                    '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                    '0_1_0_1_0_1', '0_1_0_1_1_0', # simi color - 1
                    '0_0_1_1_0_0', '0_0_1_1_1_1', # simi color - 2
                    '1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0', # simi color - 3
                    '3_0_0_0_0_1' # simi color - 4
                ] # Mar 2024, on ihc-dab, r5, not grouped, all use diff (but similar) colors
        
        slide_tile_label_dict = _run_localise_k_clsts_on_all_slides(ENV_task, clustering_pkl_name, c_assimilate_pkl_name, 
                                                                    sp_clsts)
        
        clst_gps = [['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                     '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                     '0_1_0_1_0_1', '0_1_0_1_1_0'],
                    ['0_0_1_1_0_0', '0_0_1_1_1_1'],
                    ['1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0'],
                    ['3_0_0_0_0_1']] # Mar 2024, on ihc-dab, r5
        radius = 5
        slide_gp_agt_score_dict, name_gps = _run_agt_of_clst_gps_on_all_slides_continue(ENV_task, 
                                                                                        slide_tile_label_dict, 
                                                                                        clst_gps, radius)
    if 2.3 in task_ids:
        '''
        do 1, then 2.1
        '''
        
        ''' --- cluster 1by1, ihc-dab r5 assimilate --- '''
        clustering_pkl_name = 'hiera-res-r5_Kmeans-ResNet18-encode-dab_unsupervised2024-03-01.pkl' # Feb 28 2024, ihc-dab, r5
        c_assimilate_pkl_name = 'assimilate_1by1_ft_ass-encode-ResNet18-dab_unsupervised2024-03-15.pkl' # 15th Mar 2024, ihc-dab r5, 0.002[3,3]
        # c_assimilate_pkl_name = 'assimilate_1by1_ft_ass-encode-ResNet18-dab_unsupervised2024-03-27.pkl' # 27th Mar 2024, ihc-dab r5, 0.004[3,3] with HV
        
        sp_clsts = ['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                    '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                    '0_1_0_1_0_1', '0_1_0_1_1_0', # simi color - 1
                    '0_0_1_1_0_0', '0_0_1_1_1_1', # simi color - 2
                    '1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0', # simi color - 3
                    '3_0_0_0_0_1' # simi color - 4
                ] # Mar 2024, on ihc-dab, r5, not grouped, all use diff (but similar) colors
        
        slide_tile_label_dict = _run_localise_k_clsts_on_all_slides(ENV_task, clustering_pkl_name, c_assimilate_pkl_name, 
                                                                    sp_clsts)
        
        radius = 5
        slide_spc_agt_score_dict = _run_agt_of_sp_clsts_on_all_slides_continue(ENV_task, 
                                                                               slide_tile_label_dict, 
                                                                               sp_clsts, radius)
        
