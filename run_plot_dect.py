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

    # task_ids = [0]
    task_ids = [2]
    # task_ids = [3.1, 3.2]
    
    if 0 in task_ids:
        pass
    if 1 in task_ids:
        heatmap_pkl_name = 'topK_map_GatedAttPool-g_Pool-1_ballooning_score_bi_[114]2023-10-20.pkl'
        _plot_topK_attention_heatmaps(ENV_task, ENV_annotation, heatmap_pkl_name)
    if 2 in task_ids:
        spatmap_pkl_name = 'clst-[1, 2, 5, 6, 7, 8]-a-spat_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl'
        _plot_spatial_sensi_clusters_assims(ENV_task, ENV_annotation, spatmap_pkl_name)
    if 3.1 in task_ids:
        s_clst_t_p_pkl_name = ''
        assim_t_p_pkl_name = ''
        biom_label_fname = 'P62_ballooning_score_bi.csv'
        df_plot_s_clst_assim_ball_dist_box(ENV_task, s_clst_t_p_pkl_name, assim_t_p_pkl_name, 
                                           biom_label_fname)
    if 3.2 in task_ids:
        s_clst_t_p_pkl_name = ''
        assim_t_p_pkl_name = ''
        df_plot_s_clst_assim_ball_corr_box(ENV_task, s_clst_t_p_pkl_name, assim_t_p_pkl_name)
           
    