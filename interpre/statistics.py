'''
Created on 25 Jun 2023

@author: Yang
'''
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from support.files import parse_23910_clinicalid_from_slideid, \
    parse_caseid_from_slideid
from support.metadata import parse_flinc_clinical_elsx, \
    query_task_label_dict_fromcsv


def df_cd45_cg_tis_pct_lob_score_corr(ENV_task, slide_tis_pct_dict):
    '''
    count the correlation between tissue percentage and lobular_inflammation_score
    tissue percentage was checked for iso/gath tiles in specific cluster(c) and group(g)
    
    return a dataFrame
    '''
    
    # clinical metadata file name
    cmeta_name = 'FLINC_clinical_data_DBI_2022-0715_EDG.xlsx'
    xlsx_path_clinical = '{}/{}'.format(ENV_task.META_FOLDER, cmeta_name)
    ''' clinical_label_dicts['col_label'][clinical_id] = score value / stage rank, etc. '''
    clinical_label_dicts = parse_flinc_clinical_elsx(xlsx_path_clinical)
    clinical_lob_dict = clinical_label_dicts['lobular_inflammation_score']
    
    tis_pct_lob_tuples = []
    for slide_id in slide_tis_pct_dict.keys():
        clinical_id = parse_23910_clinicalid_from_slideid(slide_id)
        tis_pct_dict = slide_tis_pct_dict[slide_id]
        if clinical_id.find('HV') == -1 and int(clinical_id) not in clinical_lob_dict.keys():
            warnings.warn('there is no such slide in the record!')
            continue
        
        lob_label = ''
        if clinical_id.startswith('HV'):
            lob_label = 'HV'
        else:
            lob_label = 'Lob-Inf-' + str(clinical_lob_dict[int(clinical_id)])
            
        tis_pct_lob_tuples.append([lob_label, tis_pct_dict[0], 'indicative Lobular-inf']) 
        tis_pct_lob_tuples.append([lob_label, tis_pct_dict[1], 'indicative (peri-)portal-inf'])
        
    df_tis_pct_lob_elemts = pd.DataFrame(tis_pct_lob_tuples,
                                         columns=['lobular_inflammation_score', 'tissue_percentage', 'groups'])
    print(df_tis_pct_lob_elemts)
    return df_tis_pct_lob_elemts

def df_plot_cd45_cg_tp_lob_box(ENV_task, df_tis_pct_lob_elemts):
    '''
    plot the cluster special group tissue percentage to fibrosis score in box chart
    '''
    palette_dict = {'indicative Lobular-inf': 'hotpink',
                    'indicative (peri-)portal-inf': 'dodgerblue'}
    order = ['HV', 'Lob-Inf-0', 'Lob-Inf-1', 'Lob-Inf-2', 'Lob-Inf-3']
    df_tis_pct_lob_elemts['lobular_inflammation_score'] = pd.Categorical(df_tis_pct_lob_elemts['lobular_inflammation_score'], 
                                                                         categories=order, ordered=True)

    
    fig = plt.figure(figsize=(6.2, 3.5))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1 = sns.boxplot(x='lobular_inflammation_score', y='tissue_percentage', palette=palette_dict,
                       data=df_tis_pct_lob_elemts, hue='groups')
    ax_1.set_title('cd45 groups (specific cluster) tissue percentage to lobular-inflammation score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ENV_task.STATISTIC_STORE_DIR, 'ref-group_tp-lob_score-box.png'))
    print('store the picture in {}'.format(ENV_task.STATISTIC_STORE_DIR))
    plt.close(fig)

def df_cd45_cg_tis_pct_fib_score_corr(ENV_task, slide_tis_pct_dict):
    '''
    count the correlation between tissue percentage and fibrosis score
    tissue percentage was checked for iso/gath tiles in specific cluster(c) and group(g)
    
    return a dataFrame
    '''
    
    # clinical metadata file name
    cmeta_name = 'FLINC_clinical_data_DBI_2022-0715_EDG.xlsx'
    xlsx_path_clinical = '{}/{}'.format(ENV_task.META_FOLDER, cmeta_name)
    ''' clinical_label_dicts['col_label'][clinical_id] = score value / stage rank, etc. '''
    clinical_label_dicts = parse_flinc_clinical_elsx(xlsx_path_clinical)
    clinical_fib_dict = clinical_label_dicts['fibrosis_score']
    
    tis_pct_fib_tuples = []
    for slide_id in slide_tis_pct_dict.keys():
        clinical_id = parse_23910_clinicalid_from_slideid(slide_id)
        tis_pct_dict = slide_tis_pct_dict[slide_id]
        if clinical_id.find('HV') == -1 and int(clinical_id) not in clinical_fib_dict.keys():
            warnings.warn('there is no such slide in the record!')
            continue
        
        fib_label = ''
        if clinical_id.startswith('HV'):
            fib_label = 'HV'
        else:
            fib_label = 'Fib-' + str(clinical_fib_dict[int(clinical_id)])
            
        tis_pct_fib_tuples.append([fib_label, tis_pct_dict[0], 'indicative Lobular-inf']) 
        tis_pct_fib_tuples.append([fib_label, tis_pct_dict[1], 'indicative (peri-)portal-inf'])
        
    df_tis_pct_fib_elemts = pd.DataFrame(tis_pct_fib_tuples,
                                         columns=['fibrosis_score', 'tissue_percentage', 'groups'])
    print(df_tis_pct_fib_elemts)
    return df_tis_pct_fib_elemts

def df_plot_cd45_cg_tp_fib_box(ENV_task, df_tis_pct_fib_elemts):
    '''
    plot the cluster special group tissue percentage to fibrosis score in box chart
    '''
    palette_dict = {'indicative Lobular-inf': 'lightcoral',
                    'indicative (peri-)portal-inf': 'turquoise'}
    order = ['HV', 'Fib-0', 'Fib-1', 'Fib-2', 'Fib-3', 'Fib-4']
    df_tis_pct_fib_elemts['fibrosis_score'] = pd.Categorical(df_tis_pct_fib_elemts['fibrosis_score'], 
                                                             categories=order, ordered=True)

    
    fig = plt.figure(figsize=(6.5, 3.5))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1 = sns.boxplot(x='fibrosis_score', y='tissue_percentage', palette=palette_dict,
                       data=df_tis_pct_fib_elemts, hue='groups')
    ax_1.set_title('cd45 groups (specific cluster) tissue percentage to fibrosis score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ENV_task.STATISTIC_STORE_DIR, 'ref-group_tp-fib_score-box.png'))
    print('store the picture in {}'.format(ENV_task.STATISTIC_STORE_DIR))
    plt.close(fig)

def df_cd45_cg_tis_pct_3a_score_corr(ENV_task, slide_tis_pct_dict):
    '''
    count the correlation between tissue percentage and alt, ast, alp
    tissue percentage was checked for iso/gath tiles in specific cluster and group
    
    return a dataFrame
    '''
    
    # clinical metadata file name
    cmeta_name = 'FLINC_clinical_data_DBI_2022-0715_EDG.xlsx'
    xlsx_path_clinical = '{}/{}'.format(ENV_task.META_FOLDER, cmeta_name)
    ''' clinical_label_dicts['col_label'][clinical_id] = score value / stage rank, etc. '''
    clinical_label_dicts = parse_flinc_clinical_elsx(xlsx_path_clinical, aim_columns=['alt', 'ast', 'alp'])
    clinical_alt_dict = clinical_label_dicts['alt']
    clinical_ast_dict = clinical_label_dicts['ast']
    clinical_alp_dict = clinical_label_dicts['alp']
    
    tis_pct_alt_tuples = []
    tis_pct_ast_tuples = []
    tis_pct_alp_tuples = []
    for slide_id in slide_tis_pct_dict.keys():
        clinical_id = parse_23910_clinicalid_from_slideid(slide_id)
        tis_pct_dict = slide_tis_pct_dict[slide_id]
        if clinical_id.startswith('HV'):
            continue
        
        clinical_id = int(clinical_id)
        alt_score = int(clinical_alt_dict[clinical_id]) if clinical_id in clinical_alt_dict.keys() else None
        ast_score = int(clinical_ast_dict[clinical_id]) if clinical_id in clinical_ast_dict.keys() else None
        alp_score = int(clinical_alp_dict[clinical_id]) if clinical_id in clinical_alp_dict.keys() else None
            
        if alt_score is not None:       
            tis_pct_alt_tuples.append([alt_score, tis_pct_dict[0], 'iso tiles']) 
            tis_pct_alt_tuples.append([alt_score, tis_pct_dict[1], 'gath tiles'])
        if ast_score is not None:
            tis_pct_ast_tuples.append([ast_score, tis_pct_dict[0], 'iso tiles'])
            tis_pct_ast_tuples.append([ast_score, tis_pct_dict[1], 'gath tiles'])
        if alp_score is not None:       
            tis_pct_alp_tuples.append([alp_score, tis_pct_dict[0], 'iso tiles']) 
            tis_pct_alp_tuples.append([alp_score, tis_pct_dict[1], 'gath tiles'])
        
    df_tis_pct_alt_elemts = pd.DataFrame(tis_pct_alt_tuples,
                                         columns=['alt_score', 'tissue_percentage', 'groups'])
    df_tis_pct_ast_elemts = pd.DataFrame(tis_pct_ast_tuples,
                                         columns=['ast_score', 'tissue_percentage', 'groups'])
    df_tis_pct_alp_elemts = pd.DataFrame(tis_pct_alp_tuples,
                                         columns=['alp_score', 'tissue_percentage', 'groups'])
    return df_tis_pct_alt_elemts, df_tis_pct_ast_elemts, df_tis_pct_alp_elemts

def df_plot_cd45_cg_tp_3a_scat(ENV_task, df_tis_pct_alt_elemts, df_tis_pct_ast_elemts, df_tis_pct_alp_elemts):
    '''
    plot the cluster special group tissue percentage to alt, ast, alp score in step chart
    '''
    group_to_color = {'iso tiles': 'red', 'gath tiles': 'blue'}
    group_to_label = {'iso tiles': 'potential lobular-inflammation',
                      'gath tiles': 'potential portal-inflammation'}
    
    df_tp_alt = df_tis_pct_alt_elemts.sort_values(by='alt_score')
    df_tp_ast = df_tis_pct_ast_elemts.sort_values(by='ast_score')
    df_tp_alp = df_tis_pct_alp_elemts.sort_values(by='alp_score')
    
    fig = plt.figure(figsize=(12, 3.5))
    
    ax_1 = fig.add_subplot(1, 3, 1)
    ax_1 = sns.scatterplot(data=df_tp_alt, x='alt_score', y='tissue_percentage',
                           hue='groups', palette=group_to_color, s=8)
    # ax_1.legend(title='groups', labels=list(group_to_label.values()))
    ax_1.set_title('correlation tis-pct to alt results')
    
    ax_2 = fig.add_subplot(1, 3, 2)
    ax_2 = sns.scatterplot(data=df_tp_ast, x='ast_score', y='tissue_percentage',
                           hue='groups', palette=group_to_color, s=8)
    # ax_2.legend(title='groups', labels=list(group_to_label.values()))
    ax_2.set_title('correlation tis-pct to ast results')
    
    ax_3 = fig.add_subplot(1, 3, 3)
    ax_3 = sns.scatterplot(data=df_tp_alp, x='alp_score', y='tissue_percentage',
                           hue='groups', palette=group_to_color, s=8)
    # ax_3.legend(title='groups', labels=list(group_to_label.values()))
    ax_3.set_title('correlation tis-pct to alp results')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ENV_task.STATISTIC_STORE_DIR, 'ref-group_tp-3a-scatter.png'))
    print('store the picture in {}'.format(ENV_task.STATISTIC_STORE_DIR))
    plt.close(fig)
    
''' ----------- data frame for p62 ballooning statistics ----------- '''

def df_p62_s_clst_assim_tis_pct_ball_dist(ENV_task, s_clst_slide_t_p_dict, assim_slide_t_p_dict,
                                          ball_label_fname):
    '''
    '''
    ball_label_dict = query_task_label_dict_fromcsv(ENV_task, ball_label_fname)
    
    tis_pct_ball_tuples = []
    for slide_id in s_clst_slide_t_p_dict.keys():
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id not in ball_label_dict.keys() or slide_id == 'avg':
            continue
        
        ball_label = 'ballooning-0' if ball_label_dict[case_id] == 0 else 'ballooning-2'
        s_clst_t_p_dict = s_clst_slide_t_p_dict[slide_id]
        s_clst_avg_t_p = sum(s_clst_t_p_dict.values()) * 1.0 / len(s_clst_slide_t_p_dict)
        tis_pct_ball_tuples.append(['sensitive clusters', ball_label, s_clst_avg_t_p])
        
        if slide_id in assim_slide_t_p_dict.keys():
            assim_t_p = assim_slide_t_p_dict[slide_id]
            both_avg_t_p = (sum(s_clst_t_p_dict.values()) + assim_t_p) * 1.0 / (len(s_clst_slide_t_p_dict) + 1)
            tis_pct_ball_tuples.append(['assimilated patches', ball_label, assim_t_p])
            tis_pct_ball_tuples.append(['both', ball_label, both_avg_t_p])
            
    df_s_clst_assim_t_p_ball_elemts = pd.DataFrame(tis_pct_ball_tuples,
                                                   columns=['patch_type', 'ballooning_label', 'tissue_percentage'])
    print(df_s_clst_assim_t_p_ball_elemts)
    return df_s_clst_assim_t_p_ball_elemts

def df_p62_s_clst_and_assim_t_p_ball_corr(ENV_task, s_clst_slide_t_p_dict, assim_slide_t_p_dict):
    '''
    '''
    cmeta_name = 'FLINC_clinical_data_DBI_2022-0715_EDG.xlsx'
    xlsx_path_clinical = '{}/{}'.format(ENV_task.META_FOLDER, cmeta_name)
    ''' clinical_label_dicts['col_label'][clinical_id] = score value / stage rank, etc. '''
    clinical_label_dicts = parse_flinc_clinical_elsx(xlsx_path_clinical)
    clinical_ball_dict = clinical_label_dicts['ballooning_score']
    
    tis_pct_ball_tuples = []
    for slide_id in s_clst_slide_t_p_dict.keys():
        clinical_id = parse_23910_clinicalid_from_slideid(slide_id)
        if clinical_id.find('HV') == -1 and int(clinical_id) not in clinical_ball_dict.keys():
            warnings.warn('there is no such slide in the record!')
            continue
        ball_label = ''
        if clinical_id.startswith('HV'):
            ball_label = 'Health volunteers'
        else:
            ball_label = 'Ballooning-' + str(clinical_ball_dict[int(clinical_id)])
        
        s_clst_t_p_dict = s_clst_slide_t_p_dict[slide_id]
        s_clst_avg_t_p = sum(s_clst_t_p_dict.values()) * 1.0 / len(s_clst_slide_t_p_dict)
        tis_pct_ball_tuples.append(['sensitive clusters', ball_label, s_clst_avg_t_p])
        
        if slide_id in assim_slide_t_p_dict.keys():
            assim_t_p = assim_slide_t_p_dict[slide_id]
            both_avg_t_p = (sum(s_clst_t_p_dict.values()) + assim_t_p) * 1.0 / (len(s_clst_slide_t_p_dict) + 1)
            tis_pct_ball_tuples.append(['assimilated patches', ball_label, assim_t_p])
            tis_pct_ball_tuples.append(['both', ball_label, both_avg_t_p])
            
    df_s_clst_assim_t_p_ball_elemts = pd.DataFrame(tis_pct_ball_tuples,
                                                   columns=['patch_type', 'ballooning_label', 'tissue_percentage'])
    print(df_s_clst_assim_t_p_ball_elemts)
    return df_s_clst_assim_t_p_ball_elemts

if __name__ == '__main__':
    pass