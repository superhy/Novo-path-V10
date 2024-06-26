'''
Created on 24 Mar 2024

@author: super
'''
import os
import sys

from interpre.plot_stat_vis import _plot_c_group_props_by_henning_frac, \
    _plot_c_gp_agts_corr_henning_frac, _plot_sp_c_agts_corr_henning_frac, \
    _plot_c_gp_agts_dist_h_l_frac2, _plot_spc_agt_heatmap_by_henning_frac, \
    _plot_spc_gps_agt_heatmap_by_henning_frac, _plot_c_gps_props_dist_in_slides, \
    _plot_c_gp_props_box_in_diff_slides_025, _plot_c_gp_agts_dist_h_l_frac3, \
    load_slide_ids_from_vis_pkg, plot_corr_between_items_from_2csv, \
    _rewrite_csv_with_slide_id, _plot_pearson_corr_heatmap_p62_fibrosis, \
    _plot_pearson_corr_heatmap_p62thd_fibrosis, _combine_org_heatmap_proptext, \
    _plot_slides_umap_sp_features_idx
from run_main import Logger
from support import env_flinc_p62, tools


if __name__ == '__main__':
    
    ENV_task = env_flinc_p62.ENV_FLINC_P62_U
    
    # task_ids = [1]
    # task_ids = [1.1, 1.2]
    # task_ids = [2]
    # task_ids = [2.21]
    # task_ids = [2.6]
    # task_ids = [3.1]
    # task_ids = [3.2, 3.21]
    # task_ids = [4]
    task_ids = [3.3]
    
    task_str = '-' + '-'.join([str(lbl) for lbl in task_ids])
    
    log_name = 'statistic_log-{}-{}.log'.format(ENV_task.TASK_NAME + task_str,
                                                    str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    
    if 1 in task_ids:
        slide_tile_label_dict_filename = 'c16-1by1_c-a-local_2024-03-26.pkl' # test on 15th Mar 2024, ihc-dab r5, 0.002[3,3]
        # slide_tile_label_dict_filename = 'c16-1by1_c-a-local_2024-03-28.pkl' # 27th Mar 2024, ihc-dab r5, 0.004[3,3]
        clst_gps = [['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                     '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                     '0_1_0_1_0_1', '0_1_0_1_1_0'],
                    ['0_0_1_1_0_0', '0_0_1_1_1_1'],
                    ['1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0'],
                    ['3_0_0_0_0_1']] # Mar 2024, on ihc-dab, r5
        _plot_c_group_props_by_henning_frac(ENV_task, slide_tile_label_dict_filename, clst_gps)
    if 1.1 in task_ids:
        slide_tile_label_dict_filename = 'c16-1by1_c-a-local_2024-03-26.pkl' # test on 15th Mar 2024, ihc-dab r5, 0.002[3,3]
        # slide_tile_label_dict_filename = 'c16-1by1_c-a-local_2024-03-28.pkl' # 27th Mar 2024, ihc-dab r5, 0.004[3,3]
        clst_gps = [['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                     '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                     '0_1_0_1_0_1', '0_1_0_1_1_0'],
                    ['0_0_1_1_0_0', '0_0_1_1_1_1'],
                    ['1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0'],
                    ['3_0_0_0_0_1']] # Mar 2024, on ihc-dab, r5
        gp_names = ['A', 'B', 'C', 'D']
        _plot_c_gps_props_dist_in_slides(ENV_task, slide_tile_label_dict_filename, clst_gps, gp_names)
    if 1.2 in task_ids:
        slide_tile_label_dict_filename = 'c16-1by1_c-a-local_2024-03-26.pkl' # test on 15th Mar 2024, ihc-dab r5, 0.002[3,3]
        # slide_tile_label_dict_filename = 'c16-1by1_c-a-local_2024-03-28.pkl' # 27th Mar 2024, ihc-dab r5, 0.004[3,3]
        clst_gps = [['0_0_0_0_0_0', '0_0_0_0_0_1', '0_0_0_1_1_0', '0_0_0_1_1_1',
                     '0_0_1_0_0_0', '0_0_1_0_0_1', '0_0_1_0_1_1', 
                     '0_1_0_1_0_1', '0_1_0_1_1_0'],
                    ['0_0_1_1_0_0', '0_0_1_1_1_1'],
                    ['1_0_0_0_1_0', '1_0_0_1_1_1', '1_0_1_0_0_0', '1_1_0_0_1_0'],
                    ['3_0_0_0_0_1']] # Mar 2024, on ihc-dab, r5
        gp_names = ['A', 'B', 'C', 'D']
        _plot_c_gp_props_box_in_diff_slides_025(ENV_task, slide_tile_label_dict_filename, 
                                                clst_gps, gp_names)
        
    if 2 in task_ids:
        # c_gp_aggregation_filename = 'agt_c-gps4_rad5_2024-03-26.pkl'
        # c_gp_aggregation_filename = 'agt_c-gps4_rad3_2024-03-29.pkl'
        # c_gp_aggregation_filename = 'agt_c-gps4_rad1_2024-03-29.pkl'
        # ylims = [(-0.001, 0.1), (-0.001, 0.2), (-0.001, 0.3)] # for 4 gps
        # ylims = [(-0.001, 0.4), (-0.001, 0.6), (-0.001, 0.8)] # for 4 gps
        # colors = ['lightseagreen', 'royalblue', 'blueviolet']
        
        c_gp_aggregation_filename = 'agt_c-gps1_rad5_2024-03-26.pkl'
        # c_gp_aggregation_filename = 'agt_c-gps1_rad3_2024-03-29.pkl'
        # c_gp_aggregation_filename = 'agt_c-gps1_rad1_2024-03-29.pkl'
        colors = ['cyan', 'dodgerblue', 'navy'] 
        ylims = [(-0.001, 0.15), (-0.001, 0.3), (-0.001, 0.5)] # for 1 gps
        # ylims = [(-0.001, 0.4), (-0.001, 0.6), (-0.001, 0.8)] # for 1 gps
        
        xlims = [(0, 0.2), (0.2, 1.0), (1.0, 5.0)]
        
        _plot_c_gp_agts_corr_henning_frac(ENV_task, c_gp_aggregation_filename,
                                          colors=colors, 
                                          xlims=xlims, ylims=ylims)
    if 2.1 in task_ids:
        sp_c_aggregation_filename = 'agt_sp-c16_rad5_2024-03-26.pkl'
        colors = ['salmon', 'orangered', 'red']
        xlims = [(0, 0.2), (0.2, 1.0), (1.0, 5.0)]
        ylims = [(-0.001, 0.05), (-0.001, 0.07), (-0.001, 0.14)]
        _plot_sp_c_agts_corr_henning_frac(ENV_task, sp_c_aggregation_filename,
                                          colors=colors, 
                                          xlims=xlims, ylims=ylims)
    if 2.2 in task_ids:
        # c_gp_aggregation_filename = 'agt_c-gps4_rad5_2024-03-26.pkl'
        # c_gp_aggregation_filename = 'agt_c-gps4_rad3_2024-03-29.pkl'
        # c_gp_aggregation_filename = 'agt_c-gps4_rad1_2024-03-29.pkl'
        # colors = ['salmon', 'lightseagreen'] # for 4 gps
        
        # c_gp_aggregation_filename = 'agt_c-gps1_rad5_2024-03-26.pkl'
        # c_gp_aggregation_filename = 'agt_c-gps1_rad3_2024-03-29.pkl'
        c_gp_aggregation_filename = 'agt_c-gps1_rad1_2024-03-29.pkl'
        colors = ['red', 'blue'] # for 1 gps
        
        frac_thd = 0.2
        _plot_c_gp_agts_dist_h_l_frac2(ENV_task, c_gp_aggregation_filename, 
                                      frac_thd, colors)
    if 2.21 in task_ids:
        # c_gp_aggregation_filename = 'agt_c-gps4_rad5_2024-03-26.pkl'
        c_gp_aggregation_filename = 'agt_c-gps4_rad3_2024-03-29.pkl'
        # c_gp_aggregation_filename = 'agt_c-gps4_rad1_2024-03-29.pkl'
        colors = ['magenta', 'salmon', 'lightseagreen'] # for 4 gps
        
        # c_gp_aggregation_filename = 'agt_c-gps1_rad5_2024-03-26.pkl'
        # c_gp_aggregation_filename = 'agt_c-gps1_rad3_2024-03-29.pkl'
        # c_gp_aggregation_filename = 'agt_c-gps1_rad1_2024-03-29.pkl'
        # colors = ['red', 'orange', 'blue'] # for 1 gps
        
        # frac_thd_tuple = (0, 0.2, 0.5)
        frac_thd_tuple = (0.2, 0.3, 0.5)
        _plot_c_gp_agts_dist_h_l_frac3(ENV_task, c_gp_aggregation_filename, 
                                       frac_thd_tuple, colors)
    
    if 2.5 in task_ids:
        sp_c_aggregation_filename = 'agt_sp-c16_rad5_2024-03-26.pkl'
        _plot_spc_agt_heatmap_by_henning_frac(ENV_task, sp_c_aggregation_filename)
    if 2.6 in task_ids:
        spc_gps_agt_filename_list = ['agt_sp-c16_rad5_2024-03-26.pkl',
                                     # 'agt_c-gps4_rad1_2024-03-29.pkl',
                                     # 'agt_c-gps4_rad3_2024-03-29.pkl',
                                     # 'agt_c-gps4_rad5_2024-03-26.pkl',
                                     # 'agt_c-gps1_rad1_2024-03-29.pkl',
                                     # 'agt_c-gps1_rad3_2024-03-29.pkl',
                                     # 'agt_c-gps1_rad5_2024-03-26.pkl'
                                     ]
        # spc_gps_agt_filename_list = ['agt_sp-c16_rad5_2024-03-26.pkl',
        #                              # 'agt_c-gps4_rad1_2024-03-29.pkl',
        #                              'agt_c-gps4_rad3_2024-03-29.pkl',
        #                              'agt_c-gps4_rad5_2024-03-26.pkl',
        #                              # 'agt_c-gps1_rad1_2024-03-29.pkl',
        #                              'agt_c-gps1_rad3_2024-03-29.pkl',
        #                              'agt_c-gps1_rad5_2024-03-26.pkl']
        
        given_y_markers = [0, 16]
        # given_y_markers = [0, 16, 20, 21]
        # given_y_markers = [0, 16, 24, 26]
        # given_y_markers = [0, 16, 28, 31]
        _plot_spc_gps_agt_heatmap_by_henning_frac(ENV_task, spc_gps_agt_filename_list,
                                                  given_y_markers)
        
    if 3.1 in task_ids:
        any_vis_pkg_name = 'c16-1by1_c-a-local_2024-03-26.pkl'
        cohort_s_marta_p_dict, _ = load_slide_ids_from_vis_pkg(ENV_task, any_vis_pkg_name)
        print(cohort_s_marta_p_dict)
        
        x_csv_filename = 'P62_ballooning_pct.csv'
        x_col_n, y_col_n = 'ballooning_percentage', '6'
        
        # x_csv_filename = 'slide_clusters_props-yang.csv'
        y_csv_filename = 'fibrosis_all_metrics_HVs-1-marta.csv'
        # x_col_n, y_col_n = 'all', 'CPA_fat_free'
        # x_col_n, y_col_n = 'all', '5'
        # x_col_n, y_col_n = 'all', '6'
        # x_col_n, y_col_n = 'all', '4'
        # x_col_n, y_col_n = 'A', 'CPA_fat_free'
        
        # y_csv_filename = 'fat-marta.csv'
        # x_col_n, y_col_n = 'C', 'number_holes'
        
        plot_corr_between_items_from_2csv(ENV_task, cohort_s_marta_p_dict, 
                                          x_csv_filename, y_csv_filename, 
                                          x_col_n, y_col_n)
    if 3.19 in task_ids:
        '''
        rewrite the csv file for aligning the column id of Yang and Marta
        '''
        any_vis_pkg_name = 'c16-1by1_c-a-local_2024-03-26.pkl'
        _, marta_p_cohort_s_dict = load_slide_ids_from_vis_pkg(ENV_task, any_vis_pkg_name)
        
        for old_csv_name in ['fat-marta.csv', 'fibrosis_all_metrics_HVs-1-marta.csv']:
            _rewrite_csv_with_slide_id(ENV_task, old_csv_name, marta_p_cohort_s_dict)  
    if 3.2 in task_ids:
        '''
        plot the pearson corr of all measurements from csv files of Yang and Marta
        '''
        csv_file_names = ['P62_ballooning_pct.csv',
                          'slide_clusters_props-yang.csv',
                          'fibrosis_all_metrics_HVs-1-marta_slideid.csv',
                          'fat-marta_slideid.csv']
        do_p_val = False
        _plot_pearson_corr_heatmap_p62_fibrosis(ENV_task, csv_file_names, 
                                                do_p_val=do_p_val)
    if 3.21 in task_ids:
        '''
        plot the pearson corr of all measurements from csv files of Yang and Marta
        '''
        csv_file_names = ['P62_ballooning_pct.csv',
                          'slide_clusters_props-yang.csv',
                          'fibrosis_all_metrics_HVs-1-marta_slideid.csv',
                          'fat-marta_slideid.csv']
        l_thds = [0.3, 0.5]
        do_p_val = False
        for l_thd in l_thds:
            _plot_pearson_corr_heatmap_p62thd_fibrosis(ENV_task, csv_file_names, 
                                                       l_thd=l_thd, do_p_val=do_p_val)
            
    if 3.3 in task_ids:
        csv_file_names = ['P62_ballooning_pct.csv',
                          'slide_clusters_props-yang.csv',
                          'fibrosis_all_metrics_HVs-1-marta_slideid.csv',
                          'fat-marta_slideid.csv']
        norm = True
        split_value=2
        keyword_feature_idx_dict = {'P62-clusters_Fibrosis': (['A', 'B', 'C', 'D', 'all'], 
                                                              'fibrosis_score'),
                                    'Fib-clusters_Fibrosis': (['0', '1', '2', '3', '4', '5', '6', 'all'], 
                                                              'fibrosis_score'),
                                    'Fat-metrics_Fibrosis': (['0 - 200 um', '200 - 400 um', '400 - 600 um', '600 - 800 um', 
                                                               '800 - 1000 um', '1000 - 1200 um', '1200 - 1400 um', '1400 - 1600 um',
                                                               '1600 - 1800 um', '1800 - 2000 um', '2000+ um',
                                                               'mean_hole_size', 'median_hole_size', 'number_holes'], 
                                                               'fibrosis_score'),
                                    'P62-Fib-Fat_features_Fibrosis': (['A', 'B', 'C', 'D', 'all',
                                                                       '0', '1', '2', '3', '4', '5', '6', 'all',
                                                                       '0 - 200 um', '200 - 400 um', '400 - 600 um', '600 - 800 um', 
                                                                       '800 - 1000 um', '1000 - 1200 um', '1200 - 1400 um', '1400 - 1600 um',
                                                                       '1600 - 1800 um', '1800 - 2000 um', '2000+ um',
                                                                       'mean_hole_size', 'median_hole_size', 'number_holes'], 
                                                                      'fibrosis_score')
        }
        
        for keyword in keyword_feature_idx_dict.keys():
            feature_columns = keyword_feature_idx_dict[keyword][0]
            index_column = keyword_feature_idx_dict[keyword][1]
            _plot_slides_umap_sp_features_idx(ENV_task, csv_file_names, 
                                              feature_columns, index_column, 
                                              fig_keyword=keyword, norm=norm, split_value=split_value)
        
            
    if 4 in task_ids:
        ''' combine the org-heatmap images '''
        _combine_org_heatmap_proptext(ENV_task)
        
        
        
        
        