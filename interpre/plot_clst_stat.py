'''
Created on 15 Feb 2023

@author: laengs2304
'''

import math
import os

from interpre.prep_clst_vis import norm_t_pct_clst_single_slide
from interpre.prep_tools import load_vis_pkg_from_pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from support.files import parse_caseid_from_slideid
from support.metadata import query_task_label_dict_fromcsv, \
    extract_slideid_subid_for_stain


def plot_biomarker_clsts_avg_dist(ENV_task, tis_pct_pkl_name, biom_label_fname,
                                  nb_clst, norm_t_pct=False):
    '''
    counting all clusters' average distribution on all slides, for biomarker/non-biomarker,
        like lobular-inf / non-lobular-inf, ballooning / ballooning
    PS: lobular label is only available for cd45 staining
        ballooning label is only available for P62 staining
    '''
    
    ''' {STAIN_TYPE: (biom_keys, palette_dict, label_name)} '''
    biomarker_info_dict = {
            'CD45': (['lobular-inf cases', 'non-lobular-inf cases'],
                     {'lobular-inf cases': 'blue', 
                      'non-lobular-inf cases': 'springgreen'},
                      'lobular-inf'),
            'P62': (['ballooning cases', 'non-ballooning cases'],
                    {'ballooning cases': 'blue', 
                     'non-ballooning cases': 'springgreen'},
                     'ballooning')
        }
    
    ''' loading/processing data '''
    biom_label_dict = query_task_label_dict_fromcsv(ENV_task, biom_label_fname)
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    if norm_t_pct is True:
        slide_tis_pct_dict = norm_t_pct_clst_single_slide(slide_tis_pct_dict, nb_clst)
    
    biom_tis_pcts, nb_biom_cases = [0.0] * nb_clst, 0
    non_biom_tis_pcts, nb_nbiom_cases = [0.0] * nb_clst, 0
    for slide_id in slide_tis_pct_dict.keys():
        tissue_pct_dict = slide_tis_pct_dict[slide_id]
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id not in biom_label_dict.keys() or slide_id == 'avg':
            continue
        if biom_label_dict[case_id] == 0:
            # non-biomarker cases
            nb_nbiom_cases += 1
            for c in range(nb_clst):
                non_biom_tis_pcts[c] += tissue_pct_dict[c]
        else:
            nb_biom_cases += 1
            for c in range(nb_clst):
                biom_tis_pcts[c] += tissue_pct_dict[c]
                
    print(biom_tis_pcts)
    print(non_biom_tis_pcts)
    print(nb_biom_cases)
    print(nb_nbiom_cases)
                
    nd_biom_tis_pct = np.array(biom_tis_pcts)
    nd_nonbiom_tis_pct = np.array(non_biom_tis_pcts)
    nd_biom_tis_pct = nd_biom_tis_pct / nb_biom_cases
    nd_nonbiom_tis_pct = nd_nonbiom_tis_pct / nb_nbiom_cases
    
    biom_keys = biomarker_info_dict[ENV_task.STAIN_TYPE][0]
    palette_dict = biomarker_info_dict[ENV_task.STAIN_TYPE][1]
    biom_lbl_name = biomarker_info_dict[ENV_task.STAIN_TYPE][2]
    
    ''' plot '''
    biom_t_pct_tuples = [['c-{}'.format(c + 1), nd_biom_tis_pct[c], biom_keys[0]] for c in range(nb_clst) ]
    nonbiom_t_pct_tuples = [['c-{}'.format(c + 1), nd_nonbiom_tis_pct[c], biom_keys[1]] for c in range(nb_clst) ]
    df_allbiom_t_pct = pd.DataFrame(biom_t_pct_tuples + nonbiom_t_pct_tuples, columns=['clusters', 'tissue_percentage', 
                                                                                    '{}_label'.format(biom_lbl_name)])
    
    fig = plt.figure(figsize=(5, 5))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1.set_ylim(0, 0.5)
    ax_1 = sns.barplot(x='clusters', y='tissue_percentage', palette=palette_dict, data=df_allbiom_t_pct, 
                       hue='{}_label'.format(biom_lbl_name))
    ax_1.set_title('tissue percentage of each clusters')
    
    # ax_2 = fig.add_subplot(1, 2, 2)
    # ax_2.set_ylim(0, 0.5)
    # ax_2 = sns.barplot(x='cluster', y='tissue_percentage', color='green', data=df_nonlob_t_pct)
    # ax_2.set_title('tissue percentage of clusters, for non-lobular cases')
    
    plt.tight_layout()
    lbl_suffix = biom_label_fname[:biom_label_fname.find('.csv')].split('_')[-1]
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR, 
                             tis_pct_pkl_name.replace('.pkl', '-{}_{}.png'.format(biom_lbl_name, lbl_suffix))))
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))
    plt.close(fig)
    
    
def plot_clsts_avg_dist_in_HV(ENV_task, tis_pct_pkl_name, nb_clst, norm_t_pct=False):
    '''
    counting all clusters' average distribution on the slides of health volunteers
    for lobular/non-lobular
    PS: lobular label is only available for cd45 staining
    '''
    xmeta_name = 'FLINC_23910-158_withSubjectID.xlsx'
    xlsx_path_158 = '{}/{}'.format(ENV_task.META_FOLDER, xmeta_name)
    slideid_subid_dict = extract_slideid_subid_for_stain(xlsx_path_158, ENV_task.STAIN_TYPE)
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    if norm_t_pct is True:
        slide_tis_pct_dict = norm_t_pct_clst_single_slide(slide_tis_pct_dict, nb_clst)
    
    nb_clst = ENV_task.NUM_CLUSTERS
    
    tis_pcts, nb_hv_cases = [0.0] * nb_clst, 0
    for slide_id in slide_tis_pct_dict.keys():
        if slide_id == 'avg':
            continue
        slide_org_id = slide_id.split('_')[1].split('-')[0]
        tissue_pct_dict = slide_tis_pct_dict[slide_id]
        subject_id = slideid_subid_dict[slide_org_id]
        if type(subject_id) != int and subject_id.startswith('HV'):
            nb_hv_cases += 1
            for c in range(nb_clst):
                tis_pcts[c] += tissue_pct_dict[c]
                
    print(tis_pcts)
    print(nb_hv_cases)
    nd_tis_pcts = np.array(tis_pcts)
    nd_tis_pcts = nd_tis_pcts / nb_hv_cases
    
    t_pct_tuples = [['c-{}'.format(c + 1), nd_tis_pcts[c]] for c in range(nb_clst) ]
    df_hvlob_t_pct = pd.DataFrame(t_pct_tuples, columns=['clusters', 'tissue_percentage'])
    
    fig = plt.figure(figsize=(5, 5))
    ax_1 = fig.add_subplot(1, 1, 1)
    # ax_1.set_ylim(0, 0.5)
    ax_1 = sns.barplot(x='clusters', y='tissue_percentage', palette=['gray'], data=df_hvlob_t_pct)
    ax_1.set_title('health volunteers\' cluster tissue percentage')
    plt.tight_layout()
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name.replace('.pkl', '-hv_lobular.png')))
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))
    plt.close(fig)


def plot_flex_clsts_avg_dist(ENV_task, ENV_flex_lbl, tis_pct_pkl_name,
                             flex_label_fname, nb_clst, norm_t_pct=False):
    '''
    Args:
        ENV_task: 
        ENV_flex_lbl:
        tis_pct_pkl_name:
        flex_label_fname: PS, cannot be None please
        nb_clst:
    '''
    
    ''' loading/processing data '''
    flex_label_dict = query_task_label_dict_fromcsv(ENV_flex_lbl, flex_label_fname)
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    if norm_t_pct is True:
        slide_tis_pct_dict = norm_t_pct_clst_single_slide(slide_tis_pct_dict, nb_clst)
    ''' for HV cases '''
    if ENV_flex_lbl.STAIN_TYPE in ['HE', 'PSR']:
        xmeta_name = 'FLINC_23910-157_withSubjectID.xlsx'
    else:
        xmeta_name = 'FLINC_23910-158_withSubjectID.xlsx'
    xlsx_path_15X = '{}/{}'.format(ENV_task.META_FOLDER, xmeta_name)
    slideid_subid_dict = extract_slideid_subid_for_stain(xlsx_path_15X, ENV_flex_lbl.STAIN_TYPE)
    print(slideid_subid_dict)
    
    label_tis_pcts, nb_lob_cases = [0.0] * nb_clst, 0
    non_label_tis_pcts, nb_nlob_cases = [0.0] * nb_clst, 0
    hv_tis_pcts, nb_hv_cases = [0.0] * nb_clst, 0
    for slide_id in slide_tis_pct_dict.keys():
        tissue_pct_dict = slide_tis_pct_dict[slide_id]
        case_id = parse_caseid_from_slideid(slide_id)
        
        if case_id not in flex_label_dict.keys() or slide_id == 'avg':
            continue
        if flex_label_dict[case_id] == 0:
            # non-lobular cases
            nb_nlob_cases += 1
            for c in range(nb_clst):
                non_label_tis_pcts[c] += tissue_pct_dict[c]
        else:
            nb_lob_cases += 1
            for c in range(nb_clst):
                label_tis_pcts[c] += tissue_pct_dict[c]
        
        slide_org_id = slide_id.split('_')[1].split('-')[0]
        subject_id = slideid_subid_dict[slide_org_id]
        if type(subject_id) != int and subject_id.startswith('HV'):
            nb_hv_cases += 1
            for c in range(nb_clst):
                hv_tis_pcts[c] += tissue_pct_dict[c]
    
    print(label_tis_pcts)
    print(non_label_tis_pcts)
    print(nb_lob_cases)
    print(nb_nlob_cases)
    nd_lbl_tis_pct = np.array(label_tis_pcts)
    nd_nonlbl_tis_pct = np.array(non_label_tis_pcts)
    nd_lbl_tis_pct = nd_lbl_tis_pct / nb_lob_cases
    nd_nonlbl_tis_pct = nd_nonlbl_tis_pct / nb_nlob_cases
    
    print(hv_tis_pcts)
    print(nb_hv_cases)
    nd_hv_tis_pcts = np.array(hv_tis_pcts)
    nd_hv_tis_pcts = nd_hv_tis_pcts / nb_hv_cases
    
    ''' plot '''
    label_name = flex_label_fname.split('_')[1]
    palette_dict = {'{} cases'.format(label_name): 'blue', 
                    'non-{} cases'.format(label_name): 'springgreen',
                    'hv cases': 'gray'}
    
    lbl_t_pct_tuples = [['c-{}'.format(c + 1), nd_lbl_tis_pct[c], '{} cases'.format(label_name)] for c in range(nb_clst) ]
    nonlbl_t_pct_tuples = [['c-{}'.format(c + 1), nd_nonlbl_tis_pct[c], 'non-{} cases'.format(label_name)] for c in range(nb_clst) ]
    hv_t_pct_tuples = [['c-{}'.format(c + 1), nd_hv_tis_pcts[c], 'hv cases'] for c in range(nb_clst) ]
    df_alllbl_t_pct = pd.DataFrame(lbl_t_pct_tuples + nonlbl_t_pct_tuples + hv_t_pct_tuples,
                                   columns=['clusters', 'tissue_percentage', 'case_label'])
    
    fig = plt.figure(figsize=(7, 5))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1.set_ylim(0, 1.0)
    ax_1 = sns.barplot(x='clusters', y='tissue_percentage', palette=palette_dict,
                       data=df_alllbl_t_pct, hue='case_label')
    ax_1.set_title('tissue percentage of each clusters')
    
    plt.tight_layout()
    lbl_suffix = flex_label_fname[:flex_label_fname.find('.csv')].split('_')[-1]
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR,
                             tis_pct_pkl_name.replace('.pkl', '-{}_{}.png'.format(label_name, lbl_suffix))))
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))
    plt.close(fig)

    
def df_lobular_prop_group_dist(ENV_task, slide_iso_gath_nb_dict, lobular_label_fname, clst_lbl):
    '''
    generate the dataFrame, calculate the average amount distribution
    '''
    
    ''' loading/processing data '''
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    lobular_prop_iso, nb_lob_cases = 0, 0
    non_lobular_prop_iso, nb_nlob_cases = 0, 0
    lobular_prop_gath = 0
    non_lobular_prop_gath = 0
    for slide_id in slide_iso_gath_nb_dict.keys():
        nb_iso, nb_gath, pop_iso, pop_gath = slide_iso_gath_nb_dict[slide_id]
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id not in lobular_label_dict.keys() or slide_id == 'avg':
            continue
        if lobular_label_dict[case_id] == 0:
            # non-lobular cases
            nb_nlob_cases += 1
            non_lobular_prop_iso += pop_iso
            non_lobular_prop_gath += pop_gath
        else:
            nb_lob_cases += 1
            lobular_prop_iso += pop_iso
            lobular_prop_gath += pop_gath
                
    print(lobular_prop_iso, lobular_prop_gath)
    print(non_lobular_prop_iso, non_lobular_prop_gath)
    print(nb_lob_cases)
    print(nb_nlob_cases)
                
    avg_lob_prop_iso = lobular_prop_iso * 1.0 / nb_lob_cases
    avg_nonlob_prop_iso = non_lobular_prop_iso * 1.0 / nb_nlob_cases
    avg_lob_prop_gath = lobular_prop_gath * 1.0 / nb_lob_cases
    avg_nonlob_prop_gath = non_lobular_prop_gath * 1.0 / nb_nlob_cases
    
    ''' plot '''
    both_nb_tuples = [['iso tiles in c-%d' % clst_lbl, avg_lob_prop_iso, 'lobular-inf cases'],
                      ['iso tiles in c-%d' % clst_lbl, avg_nonlob_prop_iso, 'non-lobular-inf cases'],
                      ['gath tiles in c-%d' % clst_lbl, avg_lob_prop_gath, 'lobular-inf cases'],
                      ['gath tiles in c-%d' % clst_lbl, avg_nonlob_prop_gath, 'non-lobular-inf cases']]
    df_alllob_prop_group = pd.DataFrame(both_nb_tuples,
                                        columns=['groups', 'number_in_group', 'lobular_label'])
    
    return df_alllob_prop_group

def df_lobular_prop_group_elements(ENV_task, slide_iso_gath_nb_dict, lobular_label_fname, clst_lbl, nb_or_prop=0):
    '''
    generate the dataFrame, list the values with different labels
    '''
    ''' loading/processing data '''
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    
    all_prop_tuples = []
    for slide_id in slide_iso_gath_nb_dict.keys():
        nb_iso, nb_gath, pop_iso, pop_gath = slide_iso_gath_nb_dict[slide_id]
        
        if nb_iso > 500 or nb_gath > 500:
            continue
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id not in lobular_label_dict.keys() or slide_id == 'avg':
            continue
        
        if lobular_label_dict[case_id] == 0:
            lob_label_str = 'non-lobular-inf cases'
        else:
            lob_label_str = 'lobular-inf cases'
        # add iso tile element and gath tile element
        all_prop_tuples.append(['iso tiles in c-%d' % clst_lbl, pop_iso if nb_or_prop == 1 else nb_iso, lob_label_str])
        all_prop_tuples.append(['gath tiles in c-%d' % clst_lbl, pop_gath if nb_or_prop == 1 else nb_gath, lob_label_str])
        
    df_alllob_prop_elemts = pd.DataFrame(all_prop_tuples,
                                        columns=['groups', 'nb/prop_group', 'lobular_label'])
    
    return df_alllob_prop_elemts

def df_lobular_prop_level_elements(ENV_task, slide_levels_nb_dict, bounds, lobular_label_fname, clst_lbl, nb_or_prop=0):
    '''
    generate the dataFrame, list the proportion of values with different levels
    '''
    ''' loading/processing data '''
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    
    all_prop_tuples = []
    for slide_id in slide_levels_nb_dict.keys():
        levels_nb_dict, levels_pop_dict = slide_levels_nb_dict[slide_id]
        
        flag_out_dist = False
        for level in range(len(bounds)):
            level_nb = levels_nb_dict[level]
            if level_nb > 500:
                flag_out_dist = True
        if flag_out_dist == True:
            continue
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id not in lobular_label_dict.keys() or slide_id == 'avg':
            continue
        
        if lobular_label_dict[case_id] == 0:
            lob_label_str = 'non-lobular-inf cases'
        else:
            lob_label_str = 'lobular-inf cases'
        # add different levels of iso tile element
        for level in range(len(bounds)):
            (b_s, b_e) = (bounds[level - 1], bounds[level]) if level >= 1 else (0.0, bounds[level])
            levels_value = levels_pop_dict[level] if nb_or_prop == 1 else levels_nb_dict[level]
            all_prop_tuples.append(['iso-(%.2f - %.2f) in c-%d' % (b_s, b_e, clst_lbl), levels_value, lob_label_str])
        
    df_alllob_prop_elemts = pd.DataFrame(all_prop_tuples,
                                         columns=['levels', 'nb/prop_level', 'lobular_label'])
    
    return df_alllob_prop_elemts

def df_lobular_tis_pct_groups(ENV_task, slide_iso_gath_nb_dict, lobular_label_fname):
    '''
    generate the dataFrame, for tissue percentage, list the values with different labels
    '''
    ''' loading/processing data '''
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    xmeta_name = 'FLINC_23910-158_withSubjectID.xlsx'
    xlsx_path_15X = '{}/{}'.format(ENV_task.META_FOLDER, xmeta_name)
    slideid_subid_dict = extract_slideid_subid_for_stain(xlsx_path_15X, ENV_task.STAIN_TYPE)
    
    all_tis_pct_tuples = []
    for slide_id in slide_iso_gath_nb_dict.keys():
        nb_iso, nb_gath, pop_iso, pop_gath = slide_iso_gath_nb_dict[slide_id]
        
        if nb_iso > 500 or nb_gath > 500:
            continue
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id not in lobular_label_dict.keys() or slide_id == 'avg':
            continue
        
        if lobular_label_dict[case_id] == 0:
            lob_label_str = 'non-lobular-inf cases'
        else:
            lob_label_str = 'lobular-inf cases'
        slide_org_id = slide_id.split('_')[1].split('-')[0]
        subject_id = slideid_subid_dict[slide_org_id]
        if type(subject_id) != int and subject_id.startswith('HV'):
            lob_label_str = 'healthy volunteers'
            
        tissue_pct_dict = slide_iso_gath_nb_dict[slide_id]
            
        # add iso tile element and gath tile element
        all_tis_pct_tuples.append(['iso tiles', tissue_pct_dict[0], lob_label_str])
        all_tis_pct_tuples.append(['gath tiles', tissue_pct_dict[1], lob_label_str])
    
    df_alllob_tis_pct_elemts = pd.DataFrame(all_tis_pct_tuples,
                                            columns=['groups', 'tissue_percentage', 'lobular_label'])
    
    return df_alllob_tis_pct_elemts
        
    
def df_plot_lobular_prop_group_bar(ENV_task, df_alllob_prop_group, lobular_label_fname, clst_lbl):
    '''
    plot with dateFrame, bar with average proportion 
    '''
    
    palette_dict = {'lobular-inf cases': 'blue', 
                    'non-lobular-inf cases': 'springgreen'}
    
    fig = plt.figure(figsize=(3.5, 5))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1 = sns.barplot(x='groups', y='number_in_group', palette=palette_dict,
                       data=df_alllob_prop_group, hue='lobular_label')
    ax_1.set_title('iso tiles in lob/non-lob slides ((c-%d))' % clst_lbl)
    
    plt.tight_layout()
    lbl_suffix = lobular_label_fname[:lobular_label_fname.find('.csv')].split('_')[-1]
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR,
                             'ref-group(c-{})_dist-lobular_{}-bar.png'.format(str(clst_lbl), lbl_suffix)))
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))
    plt.close(fig)

def dfs_plot_lobular_prop_group_bar(ENV_task, df_alllob_prop_group_list,
                                    lobular_label_fname, clst_lbl, iso_th_list):
    '''
    plot with a list of dataFrames, bar with average proportion
    '''
    palette_dict = {'lobular-inf cases': 'blue', 
                    'non-lobular-inf cases': 'springgreen'}
    
    fig = plt.figure(figsize=(3.5 * len(df_alllob_prop_group_list) / 2, 5 * 2))
    
    for i, df_alllob_prop_group in enumerate(df_alllob_prop_group_list):
        ax_th = fig.add_subplot(2, int((len(df_alllob_prop_group_list) + 1) / 2), i + 1)
        ax_th = sns.barplot(x='groups', y='number_in_group', palette=palette_dict,
                            data=df_alllob_prop_group, hue='lobular_label')
        ax_th.set_title('iso-th: %.2f nb_tiles ((c-%d))' % (iso_th_list[i], clst_lbl))
        
    plt.tight_layout()
    lbl_suffix = lobular_label_fname[:lobular_label_fname.find('.csv')].split('_')[-1]
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR,
                             'ref-group(c-{})_dist-lobular_{}(multi-thd)-bar.png'.format(str(clst_lbl), lbl_suffix)))
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))
    plt.close(fig)
    
def df_plot_lobular_prop_group_box(ENV_task, df_alllob_prop_elemts, lobular_label_fname, clst_lbl):
    '''
    plot with dateFrame, box with average proportion 
    '''
    palette_dict = {'lobular-inf cases': 'blue', 
                    'non-lobular-inf cases': 'springgreen'}
    
    fig = plt.figure(figsize=(3.5, 5))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1 = sns.boxplot(x='groups', y='nb/prop_group', palette=palette_dict,
                       data=df_alllob_prop_elemts, hue='lobular_label')
    ax_1.set_title('iso tiles in lob/non-lob slides ((c-%d))' % clst_lbl)
    
    plt.tight_layout()
    lbl_suffix = lobular_label_fname[:lobular_label_fname.find('.csv')].split('_')[-1]
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR,
                             'ref-group(c-{})_dist-lobular_{}-box.png'.format(str(clst_lbl), lbl_suffix)))
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))
    
def dfs_plot_lobular_prop_group_box(ENV_task, df_alllob_prop_elemts_list,
                                    lobular_label_fname, clst_lbl, iso_th_list):
    '''
    plot with a list of dataFrames, bar with average proportion
    '''
    palette_dict = {'lobular-inf cases': 'blue', 
                    'non-lobular-inf cases': 'springgreen'}
    
    fig = plt.figure(figsize=(3.5 * len(df_alllob_prop_elemts_list) / 2, 5 * 2))
    
    for i, df_alllob_prop_elemts in enumerate(df_alllob_prop_elemts_list):
        ax_th = fig.add_subplot(2, int((len(df_alllob_prop_elemts_list) + 1) / 2), i + 1)
        ax_th = sns.boxplot(x='groups', y='nb/prop_group', palette=palette_dict,
                            data=df_alllob_prop_elemts, hue='lobular_label')
        ax_th.set_title('iso-th: %.2f nb_tiles ((c-%d))' % (iso_th_list[i], clst_lbl))
        
    plt.tight_layout()
    lbl_suffix = lobular_label_fname[:lobular_label_fname.find('.csv')].split('_')[-1]
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR,
                             'ref-group(c-{})_dist-lobular_{}(multi-thd)-box.png'.format(str(clst_lbl), lbl_suffix)))
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))
    plt.close(fig)

def df_plot_lobular_prop_level_box(ENV_task, df_alllob_prop_elemts, lobular_label_fname, clst_lbl):
    '''
    plot with dateFrame, box with average proportion for levels
    '''
    palette_dict = {'lobular-inf cases': 'blue', 
                    'non-lobular-inf cases': 'springgreen'}
    
    fig = plt.figure(figsize=(3.5 * 3, 5))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1 = sns.boxplot(x='levels', y='nb/prop_level', palette=palette_dict,
                       data=df_alllob_prop_elemts, hue='lobular_label')
    ax_1.set_title('proportion of diff-level tiles in lob/non-lob slides ((c-%d))' % clst_lbl)
    
    plt.tight_layout()
    lbl_suffix = lobular_label_fname[:lobular_label_fname.find('.csv')].split('_')[-1]
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR,
                             'ref-level(c-{})_dist-lobular_{}-box.png'.format(str(clst_lbl), lbl_suffix)))
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))
    
def df_plot_lobular_tis_pct_box(ENV_task, df_alllob_tis_pct_elemts, lobular_label_fname, clst_lbl):
    '''
    plot with dateFrame, for tissue percentage in each slide
    '''
    palette_dict = {'lobular-inf cases': 'red', 
                    'non-lobular-inf cases': 'blue',
                    'healthy volunteers': 'gray'}
    
    fig = plt.figure(figsize=(3.5, 5))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1 = sns.boxplot(x='groups', y='tissue_percentage', palette=palette_dict,
                       data=df_alllob_tis_pct_elemts, hue='lobular_label')
    ax_1.set_title('tis-pct in lob/non-lob slides ((c-%d))' % clst_lbl)
    
    plt.tight_layout()
    lbl_suffix = lobular_label_fname[:lobular_label_fname.find('.csv')].split('_')[-1]
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR,
                             'ref-group(c-{})_tp-lobular_{}-box.png'.format(str(clst_lbl), lbl_suffix)))
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))
    plt.close(fig)
    
def dfs_plot_lobular_tis_pct_box(ENV_task, df_alllob_tis_pct_elemts_list,
                                 lobular_label_fname, clst_lbl, iso_th_list):
    '''
    plot with a list of dataFrames, bar with average proportion
    '''
    palette_dict = {'lobular-inf cases': 'red', 
                    'non-lobular-inf cases': 'blue',
                    'healthy volunteers': 'gray'}
    
    nb_row = int((len(df_alllob_tis_pct_elemts_list) + 1) / 5)
    fig = plt.figure(figsize=(3.5 * 5, 5 * nb_row))
    
    for i, df_alllob_prop_elemts in enumerate(df_alllob_tis_pct_elemts_list):
        ax_th = fig.add_subplot(nb_row, 5, i + 1)
        ax_th = sns.boxplot(x='groups', y='tissue_percentage', palette=palette_dict,
                            data=df_alllob_prop_elemts, hue='lobular_label')
        ax_th.set_title('iso-th: %.2f nb_tiles ((c-%d))' % (iso_th_list[i], clst_lbl))
        
    plt.tight_layout()
    lbl_suffix = lobular_label_fname[:lobular_label_fname.find('.csv')].split('_')[-1]
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR,
                             'ref-group(c-{})_tp-lobular_{}(multi-thd)-box.png'.format(str(clst_lbl), lbl_suffix)))
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))
    plt.close(fig)
    
def plot_lobular_sp_clst_pct_dist(ENV_task, tis_pct_pkl_name, lobular_label_fname):
    '''
    counting specific cluster's percentage distribution on all slides (numbers), for lobular/non-lobular
    '''
    
    ''' loading/processing data '''
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    
    nb_clst = ENV_task.NUM_CLUSTERS
    
    clst_t_pct_dist_dict = {}
    for c in range(nb_clst):
        clst_t_pct_dist_dict['c-{}'.format(c + 1)] = ([0] * 10, [0] * 10)
    
    tis_pct_range_labels = ['< 10', '10 ~ 20', '20 ~ 30', '30 ~ 40', '40 ~ 50',
                            '50 ~ 60', '60 ~ 70', '70 ~ 80', '80 ~ 90', '>= 90']
    
    for c in range(nb_clst):
        lob_tis_pct_dist = [0] * 10
        nonlob_tis_pct_dist = [0] * 10
        for slide_id in slide_tis_pct_dict.keys():
            if slide_id == 'avg':
                continue
            tissue_pct_dict = slide_tis_pct_dict[slide_id]
            case_id = parse_caseid_from_slideid(slide_id)
            if lobular_label_dict[case_id] == 0:
                lob_tis_pct_dist[math.floor(tissue_pct_dict[c] / 0.1)] += 1
            else:
                nonlob_tis_pct_dist[math.floor(tissue_pct_dict[c] / 0.1)] += 1
        clst_t_pct_dist_dict['c-{}'.format(c + 1)] = (lob_tis_pct_dist, nonlob_tis_pct_dist)
        
    ''' plot '''
    fig = plt.figure(figsize=(10, 15))
    for c in range(nb_clst):
        lob_t_pct_dist_tuples = [[r, clst_t_pct_dist_dict['c-{}'.format(c + 1)][0][i], 'lobular cases'] for i, r in enumerate(tis_pct_range_labels) ]
        nonlob_t_pct_dist_tuples = [[r, clst_t_pct_dist_dict['c-{}'.format(c + 1)][1][i], 'non-lobular cases'] for i, r in enumerate(tis_pct_range_labels) ]
        df_alllob_t_pct_dist = pd.DataFrame(lob_t_pct_dist_tuples + nonlob_t_pct_dist_tuples,
                                            columns=['tissue_percentage', 'number_cases', 'lobular_label'])
        
        ax_c = fig.add_subplot(2, int((nb_clst + 1) / 2), c + 1)
        ax_c = sns.displot(data=df_alllob_t_pct_dist, x='tissue_percentage', y='number_cases',
                           palette=['blue', 'springgreen'], hue='lobular_label')
        ax_c.set_title('cluster-%d' % (c + 1))
        
    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':
    pass

