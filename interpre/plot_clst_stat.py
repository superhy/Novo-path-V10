'''
Created on 15 Feb 2023

@author: laengs2304
'''

import math
import os

from interpre.prep_tools import load_vis_pkg_from_pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from support.files import parse_caseid_from_slideid
from support.metadata import query_task_label_dict_fromcsv, \
    extract_slideid_subid_for_stain


def plot_lobular_clsts_avg_dist(ENV_task, tis_pct_pkl_name, lobular_label_fname, nb_clst):
    '''
    counting all clusters' average distribution on all slides, for lobular/non-lobular
    PS: lobular label is only available for cd45 staining
    '''
    
    ''' loading/processing data '''
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    lobular_tis_pcts, nb_lob_cases = [0.0] * nb_clst, 0
    non_lobular_tis_pcts, nb_nlob_cases = [0.0] * nb_clst, 0
    for slide_id in slide_tis_pct_dict.keys():
        tissue_pct_dict = slide_tis_pct_dict[slide_id]
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id not in lobular_label_dict.keys() or slide_id == 'avg':
            continue
        if lobular_label_dict[case_id] == 0:
            # non-lobular cases
            nb_nlob_cases += 1
            for c in range(nb_clst):
                non_lobular_tis_pcts[c] += tissue_pct_dict[c]
        else:
            nb_lob_cases += 1
            for c in range(nb_clst):
                lobular_tis_pcts[c] += tissue_pct_dict[c]
                
    print(lobular_tis_pcts)
    print(non_lobular_tis_pcts)
    print(nb_lob_cases)
    print(nb_nlob_cases)
                
    nd_lob_tis_pct = np.array(lobular_tis_pcts)
    nd_nonlob_tis_pct = np.array(non_lobular_tis_pcts)
    nd_lob_tis_pct = nd_lob_tis_pct / nb_lob_cases
    nd_nonlob_tis_pct = nd_nonlob_tis_pct / nb_nlob_cases
    
    ''' plot '''
    lob_t_pct_tuples = [['c-{}'.format(c+1), nd_lob_tis_pct[c], 'lobular cases'] for c in range(nb_clst) ]
    nonlob_t_pct_tuples = [['c-{}'.format(c+1), nd_nonlob_tis_pct[c], 'non-lobular cases'] for c in range(nb_clst) ]
    df_alllob_t_pct = pd.DataFrame(lob_t_pct_tuples + nonlob_t_pct_tuples, columns=['clusters', 'tissue_percentage', 'lobular_label'])
    
    fig = plt.figure(figsize=(5, 5))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1.set_ylim(0, 0.5)
    ax_1 = sns.barplot(x='clusters', y='tissue_percentage', palette=['blue', 'springgreen'], data=df_alllob_t_pct, hue='lobular_label')
    ax_1.set_title('tissue percentage of each clusters')
    
    # ax_2 = fig.add_subplot(1, 2, 2)
    # ax_2.set_ylim(0, 0.5)
    # ax_2 = sns.barplot(x='cluster', y='tissue_percentage', color='green', data=df_nonlob_t_pct)
    # ax_2.set_title('tissue percentage of clusters, for non-lobular cases')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name.replace('.pkl', '-lobular.png')) )
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))
    
def plot_clsts_avg_dist_in_HV(ENV_task, tis_pct_pkl_name, nb_clst):
    '''
    counting all clusters' average distribution on the slides of health volunteers
    for lobular/non-lobular
    PS: lobular label is only available for cd45 staining
    '''
    xmeta_name = 'FLINC_23910-158_withSubjectID.xlsx'
    xlsx_path_158 = '{}/{}'.format(ENV_task.META_FOLDER, xmeta_name)
    slideid_subid_dict = extract_slideid_subid_for_stain(xlsx_path_158, ENV_task.STAIN_TYPE)
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    
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
    
    t_pct_tuples = [['c-{}'.format(c+1), nd_tis_pcts[c]] for c in range(nb_clst) ]
    df_hvlob_t_pct = pd.DataFrame(t_pct_tuples, columns=['clusters', 'tissue_percentage'])
    
    fig = plt.figure(figsize=(5, 5))
    ax_1 = fig.add_subplot(1, 1, 1)
    # ax_1.set_ylim(0, 0.5)
    ax_1 = sns.barplot(x='clusters', y='tissue_percentage', palette=['gray'], data=df_hvlob_t_pct)
    ax_1.set_title('health volunteers\' cluster tissue percentage')
    plt.tight_layout()
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name.replace('.pkl', '-hv_lobular.png')) )
    print('store the picture in {}'.format(ENV_task.HEATMAP_STORE_DIR))

    
def plot_lobular_sp_clst_pct_dist(ENV_task, tis_pct_pkl_name, lobular_label_fname, nb_clst=6):
    '''
    counting specific cluster's percentage distribution on all slides (numbers), for lobular/non-lobular
    '''
    
    ''' loading/processing data '''
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    
    clst_t_pct_dist_dict = {}
    for c in range(nb_clst):
        clst_t_pct_dist_dict['c-{}'.format(c+1)] = ([0] * 10, [0] * 10)
    
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
        clst_t_pct_dist_dict['c-{}'.format(c+1)] = (lob_tis_pct_dist, nonlob_tis_pct_dist)
        
    ''' plot '''
    fig = plt.figure(figsize=(10, 15))
    for c in range(nb_clst):
        lob_t_pct_dist_tuples = [[r, clst_t_pct_dist_dict['c-{}'.format(c+1)][0][i], 'lobular cases'] for i, r in enumerate(tis_pct_range_labels) ]
        nonlob_t_pct_dist_tuples = [[r, clst_t_pct_dist_dict['c-{}'.format(c+1)][1][i], 'non-lobular cases'] for i, r in enumerate(tis_pct_range_labels) ]
        df_alllob_t_pct_dist = pd.DataFrame(lob_t_pct_dist_tuples + nonlob_t_pct_dist_tuples, 
                                            columns=['tissue_percentage', 'number_cases', 'lobular_label'])
        
        ax_c = fig.add_subplot(2, int((nb_clst+1)/2), c+1)
        ax_c = sns.displot(data=df_alllob_t_pct_dist, x='tissue_percentage', y='number_cases',
                           palette=['blue', 'springgreen'], hue='lobular_label')
        ax_c.set_title('cluster-%d' % (c + 1))
        
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    pass




