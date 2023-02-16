'''
Created on 15 Feb 2023

@author: laengs2304
'''

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from interpre.prep_tools import load_vis_pkg_from_pkl
from support.metadata import query_task_label_dict_fromcsv


def plot_lobular_clsts_avg_dist(ENV_task, tis_pct_pkl_name, lobular_label_fname, nb_clst):
    '''
    counting all clusters' average distribution on all slides, for lobular/non-lobular
    '''
    
    ''' loading/processing data '''
    lobular_label_dict = query_task_label_dict_fromcsv(ENV_task, lobular_label_fname)
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    lobular_tis_pcts, nb_lob_cases = [0.0] * nb_clst, 0
    non_lobular_tis_pcts, nb_nlob_cases = [0.0] * nb_clst, 0
    for slide_id in slide_tis_pct_dict.keys():
        tissue_pct_dict = slide_tis_pct_dict[slide_id]
        if lobular_label_dict[slide_id] == 0:
            # non-lobular cases
            nb_nlob_cases += 1
            for c in range(nb_clst):
                non_lobular_tis_pcts[c] += tissue_pct_dict[c]
        if lobular_label_dict[slide_id] == 1:
            nb_lob_cases += 1
            for c in range(nb_clst):
                lobular_tis_pcts[c] += tissue_pct_dict[c]
                
    nd_lob_tis_pct = np.array(lobular_tis_pcts)
    nd_nonlob_tis_pct = np.array(non_lobular_tis_pcts)
    nd_lob_tis_pct = nd_lob_tis_pct / nb_lob_cases
    nd_nonlob_tis_pct = nd_nonlob_tis_pct / nb_nlob_cases
    
    ''' plot '''
    lob_t_pct_tuples = [[c+1, nd_lob_tis_pct[c]] for c in range(nb_clst) ]
    nonlob_t_pct_tuples = [[c+1, nd_nonlob_tis_pct[c]] for c in range(nb_clst) ]
    df_lob_t_pct = pd.DataFrame(lob_t_pct_tuples, columns=['cluster', 'tissue_percentage'])
    df_nonlob_t_pct = pd.DataFrame(nonlob_t_pct_tuples, columns=['cluster', 'tissue_percentage'])
    
    fig = plt.figure(figsize=(5, 10))
    ax_1 = fig.add_subplot(2, 1, 1)
    ax_2 = fig.add_subplot(2, 1, 2)
    ax_1 = sns.barplot(x='cluster', y='tissue_percentage', color='blue', data=df_lob_t_pct)
    ax_2 = sns.barplot(x='cluster', y='tissue_percentage', color='green', data=df_nonlob_t_pct)
    ax_1.set_title('tissue percentage of clusters, for lobular cases')
    ax_2.set_title('tissue percentage of clusters, for non-lobular cases')
    
    plt.tight_layout()
    plt.show()
    
    
def plot_lobular_sp_clst_pct_dist(ENV_task, slide_tis_pct_dict, lobular_label_path):
    '''
    counting specific cluster's percentage distribution on all slides (numbers), for lobular/non-lobular
    '''

if __name__ == '__main__':
    pass