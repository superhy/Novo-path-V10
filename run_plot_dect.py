'''
@author: Yang Hu
'''

import os

from interpre.plot_dect_vis import _plot_topK_scores_heatmaps, \
    _plot_spatial_sensi_clusters_assims, df_plot_s_clst_assim_ball_dist_box, \
    df_plot_s_clst_assim_ball_corr_box, _plot_activation_kde_dist, \
    _plot_groups_K_embeds_scatter, plot_clsts_tis_pct_abs_nb_box, \
    plot_clst_gp_tis_pct_abs_nb_box, plot_cross_labels_parcats, \
    plot_cross_labels_parcats_lmh, plot_clst_gp_tis_pct_abs_nb_ball_df_stea, \
    plot_clst_gp_tis_pct_abs_nb_ball_df_lob, plot_henning_fraction_dist, \
    plot_he_rpt_henning_label_parcats
from support import env_flinc_p62


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    ENV_annotation = env_flinc_p62.ENV_FLINC_P62_BALL_BI
    ENV_annotation_stea = env_flinc_p62.ENV_FLINC_P62_STEA_BI
    ENV_annotation_lob = env_flinc_p62.ENV_FLINC_P62_LOB_BI
    
    ENV_annotation_hv = env_flinc_p62.ENV_FLINC_P62_BALL_HV
    ENV_annotation_hv_stea = env_flinc_p62.ENV_FLINC_P62_STEA_HV
    ENV_annotation_hv_lob = env_flinc_p62.ENV_FLINC_P62_LOB_HV

    # task_ids = [0]
    # task_ids = [2]
    # task_ids = [1]
    # task_ids = [10.5]
    # task_ids = [29.3, 29.4]
    
    task_ids = [30.2]
    
    # task_ids = [201, 201.1]
    
    if 0 in task_ids:
        pass
    if 1 in task_ids:
        heatmap_pkl_name = 'topK_map_GatedAttPool-g_Pool-0_ballooning_score_bi_[159]2023-10-02.pkl'
        # heatmap_pkl_name = 'actK_map_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pkl'
        # heatmap_pkl_name = 'actK_map_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-25.pkl'
        print(heatmap_pkl_name)
        _plot_topK_scores_heatmaps(ENV_task, ENV_annotation, heatmap_pkl_name, folder_sfx='ball')
    if 1.1 in task_ids:
        # stea_map_pkl_name = 'topK_map_GatedAttPool-g_Pool-0_steatosis_score_bi_[43]2023-11-08.pkl'
        stea_map_pkl_name = 'actK_map_ResNet18-TK_MIL-0_steatosis_score_bi_[3]2023-11-24.pkl'
        # stea_map_pkl_name = 'actK_map_ResNet18-TK_MIL-0_steatosis_score_bi_[3]2023-11-24.pkl'
        _plot_topK_scores_heatmaps(ENV_task, ENV_annotation_stea, stea_map_pkl_name, folder_sfx='stea')
        # lob_map_pkl_name = 'topK_map_GatedAttPool-g_Pool-0_lobular_inflammation_score_bi_[190]2023-11-08.pkl'
        lob_map_pkl_name = 'actK_map_ResNet18-TK_MIL-0_lobular_inflammation_score_bi_[7]2023-11-26.pkl'
        _plot_topK_scores_heatmaps(ENV_task, ENV_annotation_lob, lob_map_pkl_name, folder_sfx='lob')
    if 1.2 in task_ids:
        heatmap_pkl_name = 'filt_map_GatedAttPool-g_Pool-0_ballooning_score_bi_[159]2023-10-02.pkl'
        _plot_topK_scores_heatmaps(ENV_task, ENV_annotation, heatmap_pkl_name, folder_sfx='filt-ball')
    if 2 in task_ids:
        # spatmap_pkl_name = 'clst-[1, 3]-a-spat_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl'
        # spatmap_pkl_name = '17-clst-a-spat[0-10]_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl' # [0, 10] pick test at home 2024.1
        # spatmap_pkl_name = '17-clst-a-spat[0-50]-0.002_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl'
        spatmap_pkl_name = '17-clst-a-spat[0-50]-0.01_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl'
        
        draw_org = True
        _plot_spatial_sensi_clusters_assims(ENV_task, ENV_annotation, spatmap_pkl_name, draw_org=draw_org)
    if 3.1 in task_ids:
        s_clst_t_p_pkl_name = 'sensi_c-tis-pct_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl'
        assim_t_p_pkl_name = 'assim_t-tis-pct_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl'
        biom_label_fname = 'P62_ballooning_score_bi.csv'
        df_plot_s_clst_assim_ball_dist_box(ENV_task, s_clst_t_p_pkl_name, assim_t_p_pkl_name, 
                                           biom_label_fname)
    if 3.2 in task_ids:
        s_clst_t_p_pkl_name = 'sensi_c-tis-pct_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl'
        assim_t_p_pkl_name = 'assim_t-tis-pct_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl'
        df_plot_s_clst_assim_ball_corr_box(ENV_task, s_clst_t_p_pkl_name, assim_t_p_pkl_name)
    if 10 in task_ids:
        act_scores_pkl_name = 'act_score_ft-org_ResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pkl'
        ENV_label_hv = env_flinc_p62.ENV_FLINC_P62_BALL_HV
        cut_top=500
        # _plot_activation_kde_dist(ENV_task, ENV_label_hv, act_scores_pkl_name, act_type=0, cut_top=cut_top, conj_s_range=(0.2, 0.5)) # ft
        _plot_activation_kde_dist(ENV_task, ENV_label_hv, act_scores_pkl_name, 
                                  act_type=1, cut_top=cut_top, legend_loc='upper left') # org
    if 10.5 in task_ids:
        # K_t_embeds_pkl_name = 'K_t_org-embedsResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pkl'
        K_t_embeds_pkl_name = 'K_t_ft-embedsResNet18-TK_MIL-0_ballooning_score_bi_[5]2023-11-17.pkl'
        group_names = ['Healthy volunteers', 'Ballooning 0-1', 'Ballooning 2']
        legend_loc = 'best'
        
        _plot_groups_K_embeds_scatter(ENV_task, ENV_annotation_hv, K_t_embeds_pkl_name, group_names, legend_loc)
        
    if 29.1 in task_ids:
        '''
        plot the hierarchical clustering tissue percentage / absolute number  
        with specific prefix for a family branch
        on distribution of different label, like: Healthy volunteers, Ballooning 0-1, and Ballooning 2
        '''
        tis_pct_pkl_name = 'hiera-tis-pct_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl'
        
        branch_prefix_list = ['0', '1', '2', '3']
        avail_labels_list = [['Healthy volunteers', 'Ballooning 0-1', 'Ballooning 2'],
                             ['Healthy volunteers', 'Steatosis 0-2', 'Ballooning 3'],
                             ['Healthy volunteers', 'Lob-inflammation 0-2', 'Ballooning 3']]
        tis_pct = False # if False, use absolute number
        ENV_annotation_list = [ENV_annotation_hv, ENV_annotation_hv_stea, ENV_annotation_hv_lob]
        
        for branch_prefix in branch_prefix_list:
            for i, _env_annotation in enumerate(ENV_annotation_list):
                plot_clsts_tis_pct_abs_nb_box(ENV_task, _env_annotation, tis_pct_pkl_name, 
                                              branch_prefix, avail_labels_list[i], tis_pct)
    if 29.2 in task_ids:
        '''
        same with above, calculate the distribution 
        '''
        tis_pct_pkl_name = 'hiera-tis-pct_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl'
        
        # gp_prefixs_list = [['0', '1', '2', '3'],
        #                    ['0_0', '0_1', '1_0', '1_1', '2_0', '2_1', '3_1', '3_0']]
        
        gp_prefixs_list = [['0_0_0', '0_0_1', '0_1_0', '0_1_1'],
                           ['1_0', '1_1_0', '1_1_1'],
                           ['2_0_0', '2_0_1', '2_1_0', '2_1_1'],
                           ['3_0_0', '3_0_1', '3_1_0', '3_1_1']]
        # gp_prefixs_list = [['1_0', '1_1_0', '1_1_1']]
        # gp_prefixs_list = [['2_0_0', '2_0_1', '2_1_0', '2_1_1']]
        # gp_prefixs_list = [['3_0_0', '3_0_1', '3_1_0', '3_1_1']]
        avail_labels_list = [['Healthy volunteers', 'Ballooning 0-1', 'Ballooning 2']]
        # avail_labels_list = [['Healthy volunteers', 'Ballooning 0-1', 'Ballooning 2'],
        #                      ['Healthy volunteers', 'Steatosis 0-2', 'Ballooning 3'],
        #                      ['Healthy volunteers', 'Lob-inflammation 0-2', 'Ballooning 3']]
        tis_pct = True # if False, use absolute number
        ENV_annotation_list = [ENV_annotation_hv]
        # ENV_annotation_list = [ENV_annotation_hv, ENV_annotation_hv_stea, ENV_annotation_hv_lob]
        
        for gp_prefixs in gp_prefixs_list:
            for i, _env_annotation in enumerate(ENV_annotation_list):
                plot_clst_gp_tis_pct_abs_nb_box(ENV_task, _env_annotation, tis_pct_pkl_name, 
                                                gp_prefixs, avail_labels_list[i], tis_pct)
    if 29.3 in task_ids:
        tis_pct_pkl_name = 'hiera-tis-pct_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl'
        stea_csv_filename = 'P62_steatosis_score.csv'
        
        gp_prefixs_list = [['0', '1', '2', '3'],
                           ['0_0', '0_1', '1_0', '1_1', '2_0', '2_1', '3_1', '3_0']]
        hl_prefixs_list = [None] * 2

        # gp_prefixs_list = [['0_0_0', '0_0_1', '0_1_0', '0_1_1']]
        # gp_prefixs_list = [['1_0', '1_1_0', '1_1_1']]
        # gp_prefixs_list = [['2_0_0', '2_0_1', '2_1_0', '2_1_1']]
        # gp_prefixs_list = [['3_0_0', '3_0_1', '3_1_0', '3_1_1']]
        
        # gp_prefixs_list = [['0_1_0_0_0', '0_1_0_0_1', '0_1_0_1_0', '0_1_0_1_1']]
        # hl_prefixs_list = [['0_1_0_0_0']]
        # gp_prefixs_list = [['1_1_0_0_0', '1_1_0_0_1', '1_1_0_1']]
        # hl_prefixs_list = [['1_1_0_0_0', '1_1_0_0_1']]
        # gp_prefixs_list = [['1_1_1_0_0', '1_1_1_0_1', '1_1_1_1_0', '1_1_1_1_1']]
        # hl_prefixs_list = [['1_1_1_0_1']]
        # gp_prefixs_list = [['2_1_0_0_0', '2_1_0_0_1', '2_1_0_1_0', '2_1_0_1_1']]
        # hl_prefixs_list = [['2_1_0_0_0', '2_1_0_0_1']]
        # gp_prefixs_list = [['2_1_1_0_0', '2_1_1_0_1', '2_1_1_1_0', '2_1_1_1_1']]
        # hl_prefixs_list = [['2_1_1_0_0', '2_1_1_0_1', '2_1_1_1_0']]
        # gp_prefixs_list = [['3_1_0_0_0', '3_1_0_0_1', '3_1_0_1']]
        # hl_prefixs_list = [None]
        
        # gp_prefixs_list = [['0_1_0_0_0'],
        #                    ['1_1_0_0_0', '1_1_0_0_1'],
        #                    ['1_1_1_0_1'],
        #                    ['2_1_0_0_0', '2_1_0_0_1'],
        #                    ['2_1_1_0_0', '2_1_1_0_1', '2_1_1_1_0']]
        # hl_prefixs_list = [None] * 5
        set_color = 'skyblue'
        # set_color = 'lightsalmon'
        
        tis_pct = False # if False, use absolute number
        ENV_annotation_list = [ENV_annotation_hv, ENV_annotation_hv_stea, ENV_annotation_hv_lob]
        
        for i, gp_prefixs in enumerate(gp_prefixs_list):
            plot_clst_gp_tis_pct_abs_nb_ball_df_stea(ENV_task, ENV_annotation_hv, stea_csv_filename, 
                                                     tis_pct_pkl_name, gp_prefixs, hl_prefixs_list[i], 
                                                     tis_pct, set_color)
    if 29.4 in task_ids:
        tis_pct_pkl_name = 'hiera-tis-pct_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl'
        lob_csv_filename = 'P62_lobular_inflammation_score.csv'
        
        # gp_prefixs_list = [['0', '1', '2', '3'],
        #                    ['0_0', '0_1', '1_0', '1_1', '2_0', '2_1', '3_1', '3_0']]

        # gp_prefixs_list = [['0_0_0', '0_0_1', '0_1_0', '0_1_1']]
        # gp_prefixs_list = [['1_0', '1_1_0', '1_1_1']]
        # gp_prefixs_list = [['2_0_0', '2_0_1', '2_1_0', '2_1_1']]
        # gp_prefixs_list = [['3_0_0', '3_0_1', '3_1_0', '3_1_1']]
        
        # gp_prefixs_list = [['0_1_0_0_0', '0_1_0_0_1', '0_1_0_1_0', '0_1_0_1_1']]
        # hl_prefixs_list = [['0_1_0_0_0']]
        # gp_prefixs_list = [['1_1_0_0_0', '1_1_0_0_1', '1_1_0_1']]
        # hl_prefixs_list = [['1_1_0_0_0', '1_1_0_0_1']]
        # gp_prefixs_list = [['1_1_1_0_0', '1_1_1_0_1', '1_1_1_1_0', '1_1_1_1_1']]
        # hl_prefixs_list = [['1_1_1_0_1']]
        # gp_prefixs_list = [['2_1_0_0_0', '2_1_0_0_1', '2_1_0_1_0', '2_1_0_1_1']]
        # hl_prefixs_list = [['2_1_0_0_0', '2_1_0_0_1']]
        # gp_prefixs_list = [['2_1_1_0_0', '2_1_1_0_1', '2_1_1_1_0', '2_1_1_1_1']]
        # hl_prefixs_list = [['2_1_1_0_0', '2_1_1_0_1', '2_1_1_1_0']]
        # gp_prefixs_list = [['3_1_0_0_0', '3_1_0_0_1', '3_1_0_1']]
        # hl_prefixs_list = [None]
        
        gp_prefixs_list = [['0_1_0_0_0'],
                           ['1_1_0_0_0', '1_1_0_0_1'],
                           ['1_1_1_0_1'],
                           ['2_1_0_0_0', '2_1_0_0_1'],
                           ['2_1_1_0_0', '2_1_1_0_1', '2_1_1_1_0']]
        hl_prefixs_list = [None] * 5
        
        tis_pct = False # if False, use absolute number
        ENV_annotation_list = [ENV_annotation_hv, ENV_annotation_hv_stea, ENV_annotation_hv_lob]
        
        for i, gp_prefixs in enumerate(gp_prefixs_list):
            plot_clst_gp_tis_pct_abs_nb_ball_df_lob(ENV_task, ENV_annotation_hv, lob_csv_filename, 
                                                    tis_pct_pkl_name, gp_prefixs, hl_prefixs_list[i],
                                                    tis_pct, 'lightsalmon')
            
    if 30.1 in task_ids:
        percentage_csv_name = 'P62_ballooning_percentage.csv'
        plot_henning_fraction_dist(ENV_task, percentage_csv_name)  
    if 30.2 in task_ids:
        score_csv_name = 'P62_ballooning_score.csv'
        percentage_label_csv_name = 'P62_ballooning_percentage_label.csv'
        plot_he_rpt_henning_label_parcats(ENV_task, score_csv_name, percentage_label_csv_name)
    
    if 201 in task_ids:
        '''
        plot parcats to visualise the correlation across different labels
        '''
        plot_cross_labels_parcats(ENV_task, 
                                  ball_s_csv_filename='P62_ballooning_score.csv',
                                  stea_s_csv_filename='P62_steatosis_score.csv',
                                  lob_s_csv_filename='P62_lobular_inflammation_score.csv')
    if 201.1 in task_ids:
        '''
        plot parcats to visualise the correlation across different labels
        just with labels of: low, mid, high
        '''
        plot_cross_labels_parcats_lmh(ENV_task, 
                                      ball_s_csv_filename='P62_ballooning_score.csv',
                                      stea_s_csv_filename='P62_steatosis_score.csv',
                                      lob_s_csv_filename='P62_lobular_inflammation_score.csv')
        
        
        