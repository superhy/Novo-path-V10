'''
Created on 3 Oct 2023

@author: yang hu
'''

import os

from interpre.prep_att_vis import _run_make_topK_attention_heatmap_resnet_P62
from interpre.prep_clst_vis import _run_make_clsuters_space_maps, \
    _run_make_spatial_clusters_on_slides, _run_make_tiles_demo_clusters, \
    _run_make_spatial_each_clusters_on_slides, \
    _run_count_tis_pct_clsts_on_slides, _run_make_spatial_iso_gath_on_slides, \
    _run_make_spatial_levels_on_slides, \
    _run_count_tis_pct_slides_ref_homo_sp_clst
from interpre.prep_vit_graph import _run_make_vit_graph_adj_clusters, \
    _run_make_vit_neb_graph_adj_clusters
from interpre.prep_vit_heat import _run_vit_d6_h8_cls_map_slides, \
    _run_vit_d6_h8_heads_map_slides, _run_vit_d6_h8_cls_heads_map_slides, \
    _run_reg_ass_sp_clst_homotiles_slides
from support import env_flinc_cd45, env_flinc_he, env_flinc_psr, env_flinc_p62

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"




if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    
    # task_ids = [0]
    task_ids = [1.1]
    
    if 0 in task_ids:
        pass
    if 1.1 in task_ids:
        agt_model_filenames = ['checkpoint_GatedAttPool-g_Pool-0_ballooning_score_bi_[159]2023-10-02.pth']
        K_ratio=0.3
        att_thd=0.3
        boost_rate=2.0
        
        _run_make_topK_attention_heatmap_resnet_P62(ENV_task, agt_model_filenames,
                                                    K_ratio, att_thd, boost_rate)
    