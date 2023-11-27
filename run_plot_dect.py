'''
@author: Yang Hu
'''

import os

from interpre.plot_dect_vis import _plot_topK_scores_heatmaps, \
    _plot_spatial_sensi_clusters_assims, df_plot_s_clst_assim_ball_dist_box, \
    df_plot_s_clst_assim_ball_corr_box, _plot_activation_kde_dist
from support import env_flinc_p62


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    ENV_annotation = env_flinc_p62.ENV_FLINC_P62_BALL_BI
    ENV_annotation_stea = env_flinc_p62.ENV_FLINC_P62_STEA_BI
    ENV_annotation_lob = env_flinc_p62.ENV_FLINC_P62_LOB_BI

    # task_ids = [0]
    # task_ids = [2]
    task_ids = [1]
    # task_ids = [10]
    
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
        spatmap_pkl_name = 'clst-[1, 3]-a-spat_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl'
        _plot_spatial_sensi_clusters_assims(ENV_task, ENV_annotation, spatmap_pkl_name)
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
        