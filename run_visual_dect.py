'''
Created on 3 Oct 2023

@author: yang hu
'''

import os
import sys

from interpre.prep_dect_vis import _run_make_topK_attention_heatmap_resnet_P62, \
    _run_make_spatial_sensi_clusters_assims, \
    _run_cnt_tis_pct_sensi_clsts_assim_on_slides, \
    _run_make_filt_attention_heatmap_resnet_P62, \
    _run_make_topK_activation_heatmap_resnet_P62
from run_main import Logger
from support import env_flinc_p62, tools


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"




if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    
    # task_ids = [1.1]
    task_ids = [11]
    task_str = '-' + '-'.join([str(id) for id in task_ids])
    
    log_name = 'visualisation_log-{}-{}.log'.format(ENV_task.TASK_NAME + task_str,
                                                    str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    
    if 0 in task_ids:
        ''' For original attention-based heatmap '''
    if 1 in task_ids:
        # agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-1_ballooning_score_bi_[114]2023-10-20.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-2_ballooning_score_bi_[125]2023-10-20.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-3_ballooning_score_bi_[153]2023-10-20.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-4_ballooning_score_bi_[99]2023-10-20.pth',
        #                        'checkpoint_GatedAttPool-g_Pool-7_ballooning_score_bi_[149]2023-10-22.pth']
        
        agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_score_bi_[200]2023-11-09.pth']
        
        K_ratio = 0.2
        att_thd = 0.5
        boost_rate = 1.0
        # pkg_range = [0, 50]
        pkg_range = None
        cut_left = False
        fills = [4, 5]
        
        _run_make_topK_attention_heatmap_resnet_P62(ENV_task, agt_model_filenames, cut_left,
                                                    K_ratio, att_thd, boost_rate, fills, 'bwr', pkg_range)
    if 1.1 in task_ids:
        '''
        visualisation for negative attention map
        (for steatosis_score_bi and lobular_inflammation_score_bi)
        '''    
        stea_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_steatosis_score_bi_[43]2023-11-08.pth']
        lob_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_lobular_inflammation_score_bi_[190]2023-11-08.pth']
        stea_K_ratio, stea_att_thd = 0.3, 0.5
        lob_K_ratio, lob_att_thd = 0.3, 0.3
        stea_cmap = 'PiYG'
        lob_cmap = 'BrBG'

        boost_rate = 1.0
        pkg_range = None
        cut_left = True
        fills = [4, 5]
        
        _run_make_topK_attention_heatmap_resnet_P62(ENV_task, stea_model_filenames, cut_left,
                                                    stea_K_ratio, stea_att_thd, boost_rate, fills, 
                                                    stea_cmap, pkg_range)
        _run_make_topK_attention_heatmap_resnet_P62(ENV_task, lob_model_filenames, cut_left,
                                                    lob_K_ratio, lob_att_thd, boost_rate, fills, 
                                                    lob_cmap, pkg_range)
    if 1.2 in task_ids:
        '''
        visualisation of positive - negative attention map
        (keep the attention for ballooning and discard the attention for steatosis and lobular_inflammation)
        '''
        agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_score_bi_[159]2023-10-02.pth']
        K_ratio = 0.2
        att_thd = 0.5
        
        stea_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_steatosis_score_bi_[43]2023-11-08.pth']
        lob_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_lobular_inflammation_score_bi_[190]2023-11-08.pth']
        neg_model_filenames_list = [stea_model_filenames, lob_model_filenames]
        stea_K_ratio, stea_att_thd = 0.1, 0.6
        lob_K_ratio, lob_att_thd = 0.1, 0.6
        neg_parames = [(stea_K_ratio, stea_att_thd), (lob_K_ratio, lob_att_thd)]
        
        boost_rate = 1.0
        pkg_range = None
        cut_left = True
        fills = [4, 5]
        only_soft_map=True
        
        _run_make_filt_attention_heatmap_resnet_P62(ENV_task, agt_model_filenames, 
                                                    neg_model_filenames_list, 
                                                    cut_left, K_ratio, att_thd, boost_rate, fills, 
                                                    neg_parames, 'bwr', pkg_range, only_soft_map)
        
    
    if 2 in task_ids:
        # clustering_pkl_name = 'clst-res_Kmeans-ResNet18-encode_unsupervised2023-10-26.pkl'
        # assimilate_pkl_name = 'assimilate_ft_ass-encode-ResNet18_unsupervised2023-11-04.pkl'
        ''' on PC '''
        clustering_pkl_name = 'clst-res_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl'
        assimilate_pkl_name = 'assimilate_ft_ass-encode-ResNet18_unsupervised2023-11-06.pkl'
        sp_clsts = [0]
        cut_left = True
        # heat_style = 'clst'
        heat_style = 'both'
        
        if heat_style == 'both':
            _run_make_spatial_sensi_clusters_assims(ENV_task, clustering_pkl_name, assimilate_pkl_name, 
                                                    sp_clsts, cut_left)
        else:
            _run_make_spatial_sensi_clusters_assims(ENV_task, clustering_pkl_name, None, 
                                                    sp_clsts, cut_left)
    if 3 in task_ids:
        clustering_pkl_name = 'clst-res_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl'
        assimilate_pkl_name = 'assimilate_ft_ass-encode-ResNet18_unsupervised2023-11-06.pkl'
        sp_clsts = [0]
        
        _run_cnt_tis_pct_sensi_clsts_assim_on_slides(ENV_task, clustering_pkl_name, sp_clsts, assimilate_pkl_name)
        
    if 11 in task_ids:
        tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_hv_[5]2023-11-14.pth']
        
        K_ratio = 0.5
        act_thd = 0.4
        boost_rate = 2.0
        # pkg_range = [0, 50]
        color_map='bwr'
        pkg_range = None
        cut_left = False
        fills = [3, 4, 5]
        
        _run_make_topK_activation_heatmap_resnet_P62(ENV_task, tile_net_filenames, cut_left, 
                                                     K_ratio, act_thd, boost_rate, fills, color_map, pkg_range)
    if 11.1 in task_ids:
        '''
        visualisation for negative activation map
        (for steatosis_score_hv and lobular_inflammation_score_hv)
        '''    
        stea_t_net_filenames = ['']
        lob_t_net_filenames = ['']
        stea_K_ratio, stea_act_thd = 0.5, 0.75
        lob_K_ratio, lob_act_thd = 0.5, 0.75
        stea_cmap = 'PiYG'
        lob_cmap = 'BrBG'

        boost_rate = 1.0
        pkg_range = None
        cut_left = False
        fills = [4, 5]
        
        _run_make_topK_activation_heatmap_resnet_P62(ENV_task, stea_t_net_filenames, cut_left, 
                                                     stea_K_ratio, stea_act_thd, boost_rate, 
                                                     fills, stea_cmap, pkg_range)
        _run_make_topK_activation_heatmap_resnet_P62(ENV_task, lob_t_net_filenames, cut_left, 
                                                     lob_K_ratio, lob_act_thd, boost_rate, 
                                                     fills, lob_cmap, pkg_range)
        
        
        
    