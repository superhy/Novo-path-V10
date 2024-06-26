'''
@author: Yang Hu
'''

import os

from interpre.plot_clst_stat import plot_biomarker_clsts_avg_dist, \
    plot_clsts_avg_dist_in_HV, plot_flex_clsts_avg_dist, \
    df_lobular_prop_group_dist, df_plot_lobular_prop_group_bar, \
    dfs_plot_lobular_prop_group_bar, df_lobular_prop_group_elements, \
    df_plot_lobular_prop_group_box, dfs_plot_lobular_prop_group_box, \
    df_lobular_prop_level_elements, df_plot_lobular_prop_level_box, \
    df_plot_lobular_gp_tis_pct_box, df_lobular_tis_pct_groups, \
    dfs_plot_lobular_gp_tis_pct_box
from interpre.plot_clst_vis import _run_plot_init_clst_scatter, \
    _run_plot_slides_clst_spatmap, _run_plot_clst_tile_demo, \
    _run_plot_slides_clst_each_spatmap, print_slide_tis_pct, \
    plot_demo_spatmap_4_sp_clst, plot_demo_spatmap_4_iso_group, \
    _run_plot_slides_iso_spatmap, _run_plot_slides_levels_spatmap, \
    _run_plot_clst_tile_ihcdab_demo, _run_plot_hiera_clst_scatter
from interpre.plot_dect_vis import _plot_draw_scaled_slide_imgs
from interpre.plot_graph import _run_plot_tiles_onehot_nx_graphs, \
    _run_plot_tiles_neb_nx_graphs
from interpre.plot_vit_heat import _run_plot_vit_cls_map, \
    _run_plot_vit_heads_map, _run_plot_reg_ass_homotiles_slides, \
    _run_plot_reg_ctx_g_homotiles_slides
from interpre.prep_clst_vis import top_pct_slides_4_sp_clst, \
    cnt_prop_slides_ref_homo_sp_clst, top_pop_slides_4_ref_group, \
    cnt_prop_slides_ref_levels_sp_clst
from interpre.prep_tools import load_vis_pkg_from_pkl
from interpre.statistics import df_cd45_cg_tis_pct_fib_score_corr, \
    df_plot_cd45_cg_tp_fib_box, df_cd45_cg_tis_pct_3a_score_corr, \
    df_plot_cd45_cg_tp_3a_scat, df_cd45_cg_tis_pct_lob_score_corr, \
    df_plot_cd45_cg_tp_lob_box
from support import env_flinc_cd45, env_flinc_he, env_flinc_psr, env_flinc_p62
from support.env_flinc_cd45 import ENV_FLINC_CD45_U
from support.env_flinc_he import ENV_FLINC_HE_STEA
from support.env_flinc_p62 import ENV_FLINC_P62_U
from support.env_flinc_psr import ENV_FLINC_PSR_FIB


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    
    # ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    # ENV_task = env_flinc_he.ENV_FLINC_HE_STEA_C2
#     ENV_task = env_flinc_psr.ENV_FLINC_PSR_FIB_C3

    # task_ids = [20.1]
    # task_ids = [21, 22, 29.1]
    # task_ids = [29.2]
    task_ids = [22.1]
    # task_ids = [29.1]
    # task_ids = [61, 62]
    # task_ids = [29.3]
    # task_ids = [101.1, 101.2]
    # task_ids = [101.2, 102]

    if 0 in task_ids:
        _plot_draw_scaled_slide_imgs(ENV_task)
    if 10 in task_ids:
        headsmap_pkl_name = 'headsmap_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pkl'
        _run_plot_vit_heads_map(ENV_task, headsmap_pkl_name)    
    if 11 in task_ids:
        clsmap_pkl_name = 'clsmap_ViT-6-8-PT-Dino_unsupervised[250]2022-11-02.pkl'
        _run_plot_vit_cls_map(ENV_task, clsmap_pkl_name)
    if 20 in task_ids:
        ''' CD45 '''
        # clst_scatter_pkl_name = 'tsne_5000_clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl'  # clst-6 reg
        
        ''' P62 '''
        # clst_scatter_pkl_name = 'tsne_5000_clst-res_Kmeans-ResNet18-encode_unsupervised2023-10-26.pkl' # after attention n=5
        clst_scatter_pkl_name = 'tsne_0.05_clst-res_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl' # n=4 PC
        
        _run_plot_init_clst_scatter(ENV_task, clst_scatter_pkl_name)
    if 20.1 in task_ids:
        ''' plot scatter for a part of clusters '''
        
        # color_pan = ['lime', 'aquamarine', 'lightseagreen', 'green',
        #              'slategrey', 'lightsteelblue', 'royalblue', 'blue', 'navy',
        #              'palegreen', 'yellowgreen',
        #              'khaki', 'gold', 'orange', 'olive',
        #              'salmon',
        #              'pink', 'orchid', 'purple', 'crimson']
        color_pan = ['greenyellow', 'mediumspringgreen', 'lime', 'aquamarine',
                    'mediumseagreen', 'limegreen', 'seagreen',
                    'green', 'forestgreen', # simi color - 1
                    'dodgerblue', 'blue', # simi color - 2
                    'yellow', 'gold', 'orange', 'darkorange', # simi color - 3
                    'red' # simi color - 4
                ]
        
        clst_scatter_pkl_name = 'umap_0.1_hiera-res-r5_Kmeans-ResNet18-encode-dab_unsupervised2024-03-01.pkl' # Mar 2024, ihc-dab, r5
        # label_order = ['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1', 
        #                '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', '0_0_1_1_0_0', '0_0_1_1_1_1', 
        #                '0_1_0_1_0_1', '0_1_0_1_1_0', 
        #                '1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0', 
        #                '3_0_0_0_0_1'] # Mar 2024, on ihc-dab, r5
        label_order = ['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                        '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                        '0_1_0_1_0_1', '0_1_0_1_1_0', # simi color - 1
                        '0_0_1_1_0_0', '0_0_1_1_1_1', # simi color - 2
                        '1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0', # simi color - 3
                        '3_0_0_0_0_1' # simi color - 4
                    ] # Mar 2024, on ihc-dab, r5, not grouped, all use diff (but similar) colors
        
        
        
        # TODO: shoud running on mac, test the part cluster scatter ploting
        
        _run_plot_hiera_clst_scatter(ENV_task, clst_scatter_pkl_name, color_pan, label_order)
        
    if 21 in task_ids:
        ''' CD45 '''
        # clst_spatmaps_pkl_name = 'clst-spat_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        
        ''' P62 '''
        # clst_spatmaps_pkl_name = 'clst-spat_Kmeans-ResNet18-encode_unsupervised2023-10-26.pkl' # after attention n=5
        clst_spatmaps_pkl_name = 'clst-spat_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl' # n=4 PC
        
        _run_plot_slides_clst_spatmap(ENV_task, clst_spatmaps_pkl_name)
    if 22 in task_ids:
        ''' CD45 '''
        # clst_tiledemo_pkl_name = 'clst-tiledemo_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        
        ''' P62 '''
        # clst_tiledemo_pkl_name = 'clst-tiledemo_Kmeans-ResNet18-encode_unsupervised2023-10-26.pkl' # after attention n=5
        # clst_tiledemo_pkl_name = 'clst-tiledemo_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl' # n=4 PC
        # clst_tiledemo_pkl_name = 'hiera-res_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl'
        # clst_tiledemo_pkl_name = 'hiera-tiledemo_Kmeans-ResNet18-encode_unsupervised2023-11-26.pkl' # before Dev 2023
        clst_tiledemo_pkl_name = 'hiera-tiledemo_Kmeans-ResNet18-encode_unsupervised2024-02-20.pkl' # Feb 21 2024
        _run_plot_clst_tile_demo(ENV_task, clst_tiledemo_pkl_name)
    if 22.1 in task_ids:
        # clst_t_dab_demo_pkl_name = 'hiera-t_dab-demo_Kmeans-ResNet18-encode-dab_unsupervised2024-02-28.pkl' # Feb 28 2024, ihc-dab
        clst_t_dab_demo_pkl_name = 'hiera-t_dab-demo-r5_Kmeans-ResNet18-encode-dab_unsupervised2024-03-01.pkl' # Mar 2024, ihc-dab, r5
        _run_plot_clst_tile_ihcdab_demo(ENV_task, clst_t_dab_demo_pkl_name)
        
    if 23 in task_ids:
        # clst_s_spatmap_pkl_name = 'clst-s-spat_Kmeans-encode_unsupervised2022-11-24.pkl'
        clst_s_spatmap_pkl_name = 'clst-s-spat_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_plot_slides_clst_each_spatmap(ENV_task, clst_s_spatmap_pkl_name)
    if 23.1 in task_ids:
        tis_pct_pkl_name = 'clst-tis-pct_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        clst_s_spatmap_pkl_name = 'clst-s-spat_Kmeans-region_ctx_unsupervised2023-04-05.pkl'
        lobular_label_fname = 'CD45_lobular_inflammation_score_bi.csv'
        sp_clst = 5
        nb_top = 10
        # load top and lowest tissue percentage slides
        top_slides_ids, lowest_slides_ids = top_pct_slides_4_sp_clst(ENV_task, tis_pct_pkl_name, lobular_label_fname,
                                                                     sp_clst, nb_top)
        # plot top and lowest tissue percentage slides for specific cluster
        plot_demo_spatmap_4_sp_clst(ENV_task, clst_s_spatmap_pkl_name, sp_clst, lobular_label_fname,
                                      top_slides_ids, lowest_slides_ids)
    if 24 in task_ids:
        clst_iso_spatmap_pkl_name = 'clst-s-iso_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        sp_clst = 5
        _run_plot_slides_iso_spatmap(ENV_task, clst_iso_spatmap_pkl_name, sp_clst)
    if 24.1 in task_ids:
        clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl'  # clst-6 reg
        clst_iso_spatmap_pkl_name = 'clst-s-iso_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        lobular_label_fname = 'CD45_lobular_inflammation_score_bi.csv'
        sp_clst = 5
        nb_top = 10
        slide_iso_gath_nb_dict = cnt_prop_slides_ref_homo_sp_clst(ENV_task, clustering_pkl_name, sp_clst=sp_clst)
        top_iso_slides_ids, lowest_iso_slides_ids, _, _ = top_pop_slides_4_ref_group(ENV_task, slide_iso_gath_nb_dict,
                                                                                    lobular_label_fname, nb_top)
        plot_demo_spatmap_4_iso_group(ENV_task, clst_iso_spatmap_pkl_name, sp_clst, lobular_label_fname,
                                        top_iso_slides_ids, lowest_iso_slides_ids)
    if 25 in task_ids:
        clst_levels_spatmap_pkl_name = 'clst-s-lv_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        sp_clst = 5
        _run_plot_slides_levels_spatmap(ENV_task, clst_levels_spatmap_pkl_name, sp_clst)
    if 29.1 in task_ids:
        if ENV_task.STAIN_TYPE == 'CD45':
            ''' CD45 '''
            tis_pct_pkl_name = 'clst-tis-pct_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
            lobular_label_fname = 'CD45_lobular_inflammation_score_bi.csv'
            # query_slide_id = '23910-158_Sl278-C18-CD45'
            plot_biomarker_clsts_avg_dist(ENV_task, tis_pct_pkl_name, lobular_label_fname, nb_clst=6)
        elif ENV_task.STAIN_TYPE == 'P62':
            ''' P62 '''
            # tis_pct_pkl_name = 'clst-tis-pct_Kmeans-ResNet18-encode_unsupervised2023-10-26.pkl' # after attention n=5
            tis_pct_pkl_name = 'clst-tis-pct_Kmeans-ResNet18-encode_unsupervised2023-11-06.pkl' # n=4 PC
            nb_clst=4
            ball_label_fname = 'P62_ballooning_score_bi.csv'
            # query_slide_id = '23910-158_Sl278-C18-CD45'
            plot_biomarker_clsts_avg_dist(ENV_task, tis_pct_pkl_name, ball_label_fname, nb_clst=nb_clst)
    if 29.2 in task_ids:
        ''' CD45 '''
        # tis_pct_pkl_name = 'clst-tis-pct_Kmeans-neb_encode_unsupervised2023-03-02.pkl' # nb_clst=6
        # tis_pct_pkl_name = 'clst-tis-pct_Kmeans-neb_encode_unsupervised2023-03-03.pkl'  # nb_clst=10
        
        ''' P62 '''
        # tis_pct_pkl_name = 'clst-tis-pct_Kmeans-region_ctx_unsupervised2023-09-04.pkl' # nb_clst=6 reg
        tis_pct_pkl_name = 'clst-tis-pct_Kmeans-encode_unsupervised2023-09-18.pkl' # nb_clst=6
        
        # query_slide_id = '23910-158_Sl278-C18-CD45'
        plot_clsts_avg_dist_in_HV(ENV_task, tis_pct_pkl_name, nb_clst=6)
    if 29.3 in task_ids:
        # cd45 lobular_inflammation
        # tis_pct_pkl_name = 'clst-tis-pct_Kmeans-region_ctx_unsupervised2023-04-10.pkl' # nb_clst=6 reg
        # n_clst=6
        
        # p62 ballooning
        tis_pct_pkl_name = 'clst-tis-pct_Kmeans-encode_unsupervised2023-09-18.pkl' # nb_clst=6
        n_clst=6
        # tis_pct_pkl_name = 'clst-tis-pct_Kmeans-region_ctx_unsupervised2023-09-04.pkl' # nb_clst=6 reg
        # # tis_pct_pkl_name = 'clst-tis-pct_Kmeans-encode_unsupervised2023-09-18.pkl' # nb_clst=6
        # n_clst=6
        
        flex_label_fname_1 = 'HE_steatosis_score_bi.csv'
        flex_label_fname_2 = 'PSR_fibrosis_score_bi.csv'
        felx_label_fname_3 = 'CD45_lobular_inflammation_score_bi.csv'
        felx_label_fname_4 = 'P62_ballooning_score_bi.csv'
        ENV_flex_list = [ENV_FLINC_HE_STEA, ENV_FLINC_PSR_FIB, ENV_FLINC_CD45_U, ENV_FLINC_P62_U]
        
        # TODO: make the box version
        for i, fname in enumerate([flex_label_fname_1, flex_label_fname_2, felx_label_fname_3, felx_label_fname_4]):
            plot_flex_clsts_avg_dist(ENV_task, ENV_flex_list[i], tis_pct_pkl_name,
                                     fname, nb_clst=n_clst, norm_t_pct=True)
    if 29.4 in task_ids:
        ''' plot lobular_pop_group_dist with one iso threshold, 
            show the proportion of each group
        '''
        
        clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl'  # clst-6 reg
        lobular_label_fname = 'CD45_lobular_inflammation_score_bi.csv'
        # plot_type = 'bar'
        plot_type = 'box'
        
        clst_lbl = 5
        radius = 3
        slide_iso_gath_nb_dict = cnt_prop_slides_ref_homo_sp_clst(ENV_task, clustering_pkl_name,
                                                                  sp_clst=clst_lbl, iso_thd=0.25, radius=radius)
        if plot_type == 'bar':
            df_alllob_prop_group = df_lobular_prop_group_dist(ENV_task, slide_iso_gath_nb_dict,
                                                              lobular_label_fname, clst_lbl)
            df_plot_lobular_prop_group_bar(ENV_task, df_alllob_prop_group, lobular_label_fname, clst_lbl)
        else:
            df_alllob_prop_elemts = df_lobular_prop_group_elements(ENV_task, slide_iso_gath_nb_dict,
                                                                   lobular_label_fname, clst_lbl)
            df_plot_lobular_prop_group_box(ENV_task, df_alllob_prop_elemts, lobular_label_fname, clst_lbl)
    if 29.5 in task_ids:
        ''' plot lobular_pop_group_dist with multiple iso thresholds,
            show the proportion of each group
        '''
        
        clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl'  # clst-6 reg
        lobular_label_fname = 'CD45_lobular_inflammation_score_bi.csv'
        # plot_type = 'bar'
        plot_type = 'box'
        
        iso_th_list = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        clst_lbl = 5
        radius = 3
        nb_or_prop = 0
        if plot_type == 'bar':
            df_alllob_tis_pct_list = []
            for i, th in enumerate(iso_th_list):
                slide_iso_gath_nb_dict = cnt_prop_slides_ref_homo_sp_clst(ENV_task, clustering_pkl_name,
                                                                          sp_clst=clst_lbl, iso_thd=th, radius=radius)
                df_alllob_prop_group = df_lobular_prop_group_dist(ENV_task, slide_iso_gath_nb_dict,
                                                                lobular_label_fname, clst_lbl, nb_or_prop)
                df_alllob_tis_pct_list.append(df_alllob_prop_group)
            dfs_plot_lobular_prop_group_bar(ENV_task, df_alllob_tis_pct_list,
                                            lobular_label_fname, clst_lbl, iso_th_list)
        else:
            df_alllob_prop_elemts_list = []
            for i, th in enumerate(iso_th_list):
                slide_iso_gath_nb_dict = cnt_prop_slides_ref_homo_sp_clst(ENV_task, clustering_pkl_name,
                                                                          sp_clst=5, iso_thd=th, radius=radius)
                df_alllob_prop_elemts = df_lobular_prop_group_elements(ENV_task, slide_iso_gath_nb_dict,
                                                                       lobular_label_fname, clst_lbl, nb_or_prop)
                df_alllob_prop_elemts_list.append(df_alllob_prop_elemts)
            dfs_plot_lobular_prop_group_box(ENV_task, df_alllob_prop_elemts_list,
                                            lobular_label_fname, clst_lbl, iso_th_list)
    if 29.6 in task_ids:
        # not suggest to use
        clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl'  # clst-6 reg
        lobular_label_fname = 'CD45_lobular_inflammation_score_bi.csv'
        
        clst_lbl = 5
        radius = 3
        nb_or_prop = 0
        slide_levels_nb_dict, bounds = cnt_prop_slides_ref_levels_sp_clst(ENV_task, clustering_pkl_name,
                                                                          sp_clst=clst_lbl, radius=radius)
        df_alllob_prop_elemts = df_lobular_prop_level_elements(ENV_task, slide_levels_nb_dict, bounds,
                                                               lobular_label_fname, clst_lbl, nb_or_prop)
        df_plot_lobular_prop_level_box(ENV_task, df_alllob_prop_elemts, lobular_label_fname, clst_lbl)
    if 30 in task_ids:
        ''' show the tissue percentage of each group, with one iso threshold.
            the task_id continue with 29.x
        '''
        
        clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl'  # clst-6 reg
        lobular_label_fname = 'CD45_lobular_inflammation_score_bi.csv'
        # tis_pct_pkl_name = 'clst-gp-tis-pct_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        clst_lbl = 5
        radius = 3
        
        # slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
        slide_iso_gath_nb_dict = cnt_prop_slides_ref_homo_sp_clst(ENV_task, clustering_pkl_name,
                                                                  sp_clst=clst_lbl, iso_thd=0.2, radius=radius)
        df_alllob_tis_pct_elemts = df_lobular_tis_pct_groups(ENV_task, slide_iso_gath_nb_dict,
                                                             lobular_label_fname)
        
        df_plot_lobular_gp_tis_pct_box(ENV_task, df_alllob_tis_pct_elemts, lobular_label_fname, clst_lbl)
    
    if 30.1 in task_ids:
        ''' show the tissue percentage of each group, with multiple iso thresholds '''
        
        clustering_pkl_name = 'clst-res_Kmeans-region_ctx_unsupervised2023-04-10.pkl'  # clst-6 reg
        lobular_label_fname = 'CD45_lobular_inflammation_score_bi.csv'
        # tis_pct_pkl_name = 'clst-gp-tis-pct_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        iso_th_list = [0.02, 0.04, 0.06, 0.08, 0.10,
                       0.12, 0.14, 0.16, 0.18, 0.20,
                       0.22, 0.24, 0.26, 0.28, 0.30]
        clst_lbl = 5
        radius = 3
        
        df_alllob_tis_pct_list = []
        for i, th in enumerate(iso_th_list):
            # slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
            slide_iso_gath_nb_dict = cnt_prop_slides_ref_homo_sp_clst(ENV_task, clustering_pkl_name,
                                                                      sp_clst=clst_lbl, iso_thd=th, radius=radius)
            df_alllob_tis_pct_elemts = df_lobular_tis_pct_groups(ENV_task, slide_iso_gath_nb_dict,
                                                                 lobular_label_fname)
            df_alllob_tis_pct_list.append(df_alllob_tis_pct_elemts)
        dfs_plot_lobular_gp_tis_pct_box(ENV_task, df_alllob_tis_pct_list, lobular_label_fname, clst_lbl, iso_th_list)
    if 31 in task_ids:
        sp_clst_reg_ass_pkl_name = 'sp_clst_homotiles_reg_ass-2023-08-09.pkl'
        edge_thd = 0.75
        _run_plot_reg_ass_homotiles_slides(ENV_task, sp_clst_reg_ass_pkl_name, edge_thd)
    if 31.1 in task_ids:
        sp_clst_reg_mat_pkl_name = 'sp_clst_homotiles_reg_ctx-2023-08-20.pkl'
        # plot tile context graph with centre_ass=False
        # TODO: test it
        _run_plot_reg_ctx_g_homotiles_slides(ENV_task, sp_clst_reg_mat_pkl_name)
    
    if 61 in task_ids:
        adjdict_pkl_name = 'c-2-adjs_o_0.5_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_plot_tiles_onehot_nx_graphs(ENV_task, adjdict_pkl_name)
    if 62 in task_ids:
        adjdict_pkl_name = 'c-2-adjs_x_0.0_Kmeans-neb_encode_unsupervised2022-11-28.pkl'
        _run_plot_tiles_neb_nx_graphs(ENV_task, adjdict_pkl_name)
        
    if 101.1 in task_ids:
        tis_pct_pkl_name = 'clst-gp-tis-pct_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        
        slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
        df_tis_pct_lob_elemts = df_cd45_cg_tis_pct_lob_score_corr(ENV_task, slide_tis_pct_dict)
        df_plot_cd45_cg_tp_lob_box(ENV_task, df_tis_pct_lob_elemts)
    if 101.2 in task_ids:
        tis_pct_pkl_name = 'clst-gp-tis-pct_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        
        slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
        df_tis_pct_fib_elemts = df_cd45_cg_tis_pct_fib_score_corr(ENV_task, slide_tis_pct_dict)
        df_plot_cd45_cg_tp_fib_box(ENV_task, df_tis_pct_fib_elemts)
    if 102 in task_ids:
        tis_pct_pkl_name = 'clst-gp-tis-pct_Kmeans-region_ctx_unsupervised2023-04-10.pkl'
        
        slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
        df_tis_pct_alt_elemts, df_tis_pct_ast_elemts, df_tis_pct_alp_elemts = \
            df_cd45_cg_tis_pct_3a_score_corr(ENV_task, slide_tis_pct_dict)
        df_plot_cd45_cg_tp_3a_scat(ENV_task, df_tis_pct_alt_elemts, df_tis_pct_ast_elemts, df_tis_pct_alp_elemts)
        
        
