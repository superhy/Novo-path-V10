'''
@author: Yang Hu
'''

import os

from interpre.plot_clst_vis import _run_plot_clst_scatter, \
    _run_plot_slides_clst_spatmap
from interpre.plot_vit_heat import _run_plot_vit_cls_map, \
    _run_plot_vit_heads_map
from support import env_flinc_cd45, env_flinc_he, env_flinc_psr


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



if __name__ == '__main__':
    
    ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
#     ENV_task = env_flinc_he.ENV_FLINC_HE_STEA_C2
#     ENV_task = env_flinc_psr.ENV_FLINC_PSR_FIB_C3

    task_ids = [21]
    
    if 11 in task_ids:
        clsmap_pkl_name = 'clsmap_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pkl'
        _run_plot_vit_cls_map(ENV_task, clsmap_pkl_name)
    if 12 in task_ids:
        headsmap_pkl_name = 'headsmap_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pkl'
        _run_plot_vit_heads_map(ENV_task, headsmap_pkl_name)
    if 20 in task_ids:
        clst_space_pkl_name = 'tsne_all_clst-res_Kmeans-encode_unsupervised2022-11-23.pkl'
        _run_plot_clst_scatter(ENV_task, clst_space_pkl_name)
    if 21 in task_ids:
        clst_spatmaps_pkl_name = 'clst-spat_Kmeans-encode_unsupervised2022-11-23.pkl'
        _run_plot_slides_clst_spatmap(ENV_task, clst_spatmaps_pkl_name)