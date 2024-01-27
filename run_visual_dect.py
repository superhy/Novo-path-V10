'''
Created on 3 Oct 2023

@author: yang hu
'''

import os
import sys

from interpre.prep_clst_vis import pick_clusters_by_prefix
from interpre.prep_dect_vis import _run_make_topK_attention_heatmap_resnet_P62, \
    _run_make_spatial_sensi_clusters_assims, \
    _run_cnt_tis_pct_sensi_c_assim_t_on_slides, \
    _run_make_filt_attention_heatmap_resnet_P62, \
    _run_make_topK_activation_heatmap_resnet_P62, \
    _load_activation_score_resnet_P62, _run_get_top_act_tiles_embeds_allslides, \
    _run_make_filt_activation_heatmap_resnet_P62, \
    _run_cnt_abs_nb_sensi_c_assim_t_on_slides
from run_main import Logger
from support import env_flinc_p62, tools


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"




if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    
    # task_ids = [1]
    task_ids = [2.1]
    # task_ids = [11.1]
    # task_ids = [10.5]
    # task_ids = [11.1]
    
    task_str = '-' + '-'.join([str(lbl) for lbl in task_ids])
    
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
        
        agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_score_bi_[159]2023-10-02.pth']
        
        K_ratio = 0.3
        att_thd = 0.3
        boost_rate = 2.0
        # pkg_range = [0, 50]
        pkg_range = None
        cut_left = False
        fills = [3, 3, 3, 3]
        
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
        pkg_range = [0, 100]
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
        sp_clsts = [0] # should be changed one-by-one
        cut_left = True
        # heat_style = 'clst'
        heat_style = 'both'
        if assimilate_pkl_name is None:
            heat_style = 'clst'
        
        if heat_style == 'both':
            _run_make_spatial_sensi_clusters_assims(ENV_task, clustering_pkl_name, assimilate_pkl_name, 
                                                    sp_clsts, cut_left)
        else:
            _run_make_spatial_sensi_clusters_assims(ENV_task, clustering_pkl_name, None, 
                                                    sp_clsts, cut_left)
    if 2.1 in task_ids:
        # clustering_pkl_name = 'clst-res_Kmeans-ResNet18-encode_unsupervised2023-10-26.pkl'
        # assimilate_pkl_name = 'assimilate_ft_ass-encode-ResNet18_unsupervised2023-11-04.pkl'
        ''' on NN-Cluster '''
        clustering_pkl_name = 'hiera-res_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl' 
        assimilate_pkl_name = 'assimilate_ft_ass-encode-ResNet18_unsupervised2023-12-15.pkl'
        cluster_groups = ['0_1_0_0_0', '1_1_0_0_0', '1_1_0_0_1', '1_1_1_0_1',
                          '2_1_0_0_0', '2_1_0_0_1', '2_1_1_0_0', '2_1_1_0_1', 
                          '2_1_1_1_0']
        
        sp_clsts = pick_clusters_by_prefix(ENV_task, clustering_pkl_name, cluster_groups)
        cut_left = True
        # heat_style = 'clst'
        part_vis = [0, 50 - 1]
        heat_style = 'both'
        if assimilate_pkl_name is None:
            heat_style = 'clst'
        
        if heat_style == 'both':
            _run_make_spatial_sensi_clusters_assims(ENV_task, clustering_pkl_name, assimilate_pkl_name, 
                                                    sp_clsts, cut_left, part_vis=part_vis)
        else:
            _run_make_spatial_sensi_clusters_assims(ENV_task, clustering_pkl_name, None, 
                                                    sp_clsts, cut_left, part_vis=part_vis)
    if 2.19 in task_ids:
        ''' on NN-Cluster '''
        clustering_pkl_name = 'hiera-res_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl' 
        assimilate_pkl_name = 'assimilate_ft_ass-encode-ResNet18_unsupervised2023-12-15.pkl'
        cluster_groups = ['0_1_0_0_0', '1_1_0_0_0', '1_1_0_0_1', '1_1_1_0_1',
                          '2_1_0_0_0', '2_1_0_0_1', '2_1_1_0_0', '2_1_1_0_1', 
                          '2_1_1_1_0']
        
        sp_clsts = pick_clusters_by_prefix(ENV_task, clustering_pkl_name, cluster_groups)
        cut_left = True
        aprx_list_len = 260 # approximate maximum list length
        step_len = 50
        heat_style = 'both'
        if assimilate_pkl_name is None:
            heat_style = 'clst'
            
        start = 0
        end = start + step_len - 1
        while end < aprx_list_len:
            part_vis = [start, end]
            if heat_style == 'both':
                _run_make_spatial_sensi_clusters_assims(ENV_task, clustering_pkl_name, assimilate_pkl_name, 
                                                        sp_clsts, cut_left, part_vis=part_vis)
            else:
                _run_make_spatial_sensi_clusters_assims(ENV_task, clustering_pkl_name, None, 
                                                        sp_clsts, cut_left, part_vis=part_vis)
            start += step_len
            end = start + step_len - 1
                
    
    if 3.1 in task_ids:
        ''' old '''
        clustering_pkl_name = 'clst-res_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl'
        assimilate_pkl_name = 'assimilate_ft_ass-encode-ResNet18_unsupervised2023-11-06.pkl'
        
        ''' on NN-Cluster '''
        clustering_pkl_name = 'hiera-res_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl' 
        assimilate_pkl_name = 'assimilate_ft_ass-encode-ResNet18_unsupervised2023-12-15.pkl'
        cluster_groups = ['0_1_0_0_0', '1_1_0_0_0', '1_1_0_0_1', '1_1_1_0_1',
                          '2_1_0_0_0', '2_1_0_0_1', '2_1_1_0_0', '2_1_1_0_1', 
                          '2_1_1_1_0']
        
        sp_clsts = pick_clusters_by_prefix(ENV_task, clustering_pkl_name, cluster_groups)
        
        _run_cnt_tis_pct_sensi_c_assim_t_on_slides(ENV_task, clustering_pkl_name, sp_clsts, assimilate_pkl_name)
    if 3.2 in task_ids:
        ''' on NN-Cluster '''
        clustering_pkl_name = 'hiera-res_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl' 
        assimilate_pkl_name = 'assimilate_ft_ass-encode-ResNet18_unsupervised2023-12-15.pkl'
        cluster_groups = ['0_1_0_0_0', '1_1_0_0_0', '1_1_0_0_1', '1_1_1_0_1',
                          '2_1_0_0_0', '2_1_0_0_1', '2_1_1_0_0', '2_1_1_0_1', 
                          '2_1_1_1_0']
        
        sp_clsts = pick_clusters_by_prefix(ENV_task, clustering_pkl_name, cluster_groups)
        
        _run_cnt_abs_nb_sensi_c_assim_t_on_slides(ENV_task, clustering_pkl_name, sp_clsts, assimilate_pkl_name)
    
    if 10 in task_ids:
        tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pth']
        # tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[4]2023-11-23.pth']
        # tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[7]2023-11-25.pth']
        for t_net_name in tile_net_filenames:
            _load_activation_score_resnet_P62(ENV_task, t_net_name)   
    if 10.5 in task_ids:
        tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pth']
        # tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[4]2023-11-23.pth']
        # tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[7]2023-11-25.pth']
        K=10
        for t_net_name in tile_net_filenames:
            _run_get_top_act_tiles_embeds_allslides(ENV_task, t_net_name, K, embed_type=0)
    if 11 in task_ids:
        tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pth']
        # tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[4]2023-11-23.pth']
        # tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-25.pth']
        
        K_ratio = 0.75
        act_thd = 0.36
        boost_rate = 2.0
        # pkg_range = [0, 50]
        color_map='bwr'
        pkg_range = None
        cut_left = False
        fills = [3, 3, 3, 3, 3]
        
        _run_make_topK_activation_heatmap_resnet_P62(ENV_task, tile_net_filenames, cut_left, 
                                                     K_ratio, act_thd, boost_rate, fills, color_map, pkg_range)
    if 11.1 in task_ids:
        '''
        visualisation for negative activation map
        (for steatosis_score_hv and lobular_inflammation_score_hv)
        '''    
        stea_t_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_steatosis_score_bi_[3]2023-11-24.pth']
        lob_t_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_lobular_inflammation_score_bi_[7]2023-11-26.pth']
        stea_K_ratio, stea_act_thd = 0.75, 0.4
        lob_K_ratio, lob_act_thd = 0.75, 0.4
        stea_cmap = 'PiYG'
        lob_cmap = 'BrBG'

        boost_rate = 2.0
        pkg_range = None
        cut_left = False
        fills = [3, 3, 3, 3, 3]
        
        # _run_make_topK_activation_heatmap_resnet_P62(ENV_task, stea_t_net_filenames, cut_left, 
        #                                              stea_K_ratio, stea_act_thd, boost_rate, 
        #                                              fills, stea_cmap, pkg_range)
        _run_make_topK_activation_heatmap_resnet_P62(ENV_task, lob_t_net_filenames, cut_left, 
                                                     lob_K_ratio, lob_act_thd, boost_rate, 
                                                     fills, lob_cmap, pkg_range)
    if 11.2 in task_ids:
        '''
        visualisation of positive - negative activation map
        (keep the activation for ballooning and discard the activation for steatosis and lobular_inflammation)
        '''
        tile_net_filenames = ['checkpoint_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pth']
        K_ratio = 0.75
        act_thd = 0.36
        
        stea_model_filenames = ['checkpoint_ResNet18-TK_MIL-0_steatosis_score_bi_[5]2023-11-20.pth']
        lob_model_filenames = ['checkpoint_ResNet18-TK_MIL-0_lobular_inflammation_score_bi_[5]2023-11-21.pth']
        neg_model_filenames_list = [stea_model_filenames, lob_model_filenames]
        stea_K_ratio, stea_att_thd = 0.75, 0.3
        lob_K_ratio, lob_att_thd = 0.75, 0.3
        neg_parames = [(stea_K_ratio, stea_att_thd), (lob_K_ratio, lob_att_thd)]
        
        boost_rate = 2.0
        pkg_range = [0, 100]
        cut_left = True
        fills = [3, 3, 3, 3, 3]
        only_soft_map=False
        
        _run_make_filt_activation_heatmap_resnet_P62(ENV_task, tile_net_filenames, neg_model_filenames_list,
                                                     cut_left, K_ratio, act_thd, 
                                                     boost_rate, fills, neg_parames, 'bwr', 
                                                     pkg_range, only_soft_map)
        
        
        
    