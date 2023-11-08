'''
@author: Yang Hu
'''

import os

from interpre.plot_dect_vis import _plot_topK_attention_heatmaps, \
    _plot_spatial_sensi_clusters_assims, df_plot_s_clst_assim_ball_dist_box, \
    df_plot_s_clst_assim_ball_corr_box
from support import env_flinc_p62


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    ENV_annotation = env_flinc_p62.ENV_FLINC_P62_BALL_BI
    ENV_annotation_stea = env_flinc_p62.ENV_FLINC_P62_STEA_BI
    ENV_annotation_lob = env_flinc_p62.ENV_FLINC_P62_LOB_BI

    # task_ids = [0]
    # task_ids = [2]
    # task_ids = [1, 1.1]
    task_ids = [1.2]
    
    if 0 in task_ids:
        pass
    if 1 in task_ids:
        heatmap_pkl_name = 'topK_map_GatedAttPool-g_Pool-0_ballooning_score_bi_[159]2023-10-02.pkl'
        _plot_topK_attention_heatmaps(ENV_task, ENV_annotation, heatmap_pkl_name, folder_sfx='ball')
    if 1.1 in task_ids:
        stea_map_pkl_name = 'topK_map_GatedAttPool-g_Pool-0_steatosis_score_bi_[67]2023-11-08.pkl'
        _plot_topK_attention_heatmaps(ENV_task, ENV_annotation_stea, stea_map_pkl_name, folder_sfx='stea')
        lob_map_pkl_name = 'topK_map_GatedAttPool-g_Pool-0_lobular_inflammation_score_bi_[159]2023-11-08.pkl'
        _plot_topK_attention_heatmaps(ENV_task, ENV_annotation_lob, lob_map_pkl_name, folder_sfx='lob')
    if 1.2 in task_ids:
        heatmap_pkl_name = 'filt_map_GatedAttPool-g_Pool-0_ballooning_score_bi_[159]2023-10-02.pkl'
        _plot_topK_attention_heatmaps(ENV_task, ENV_annotation, heatmap_pkl_name, folder_sfx='filt-ball')
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
        