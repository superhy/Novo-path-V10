'''
Created on 24 Mar 2024

@author: super
'''

import os


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from support import metadata
from support.files import parse_caseid_from_slideid
from interpre import prep_stat_vis
from interpre.prep_tools import load_vis_pkg_from_pkl


def plot_clst_group_props_sort_by_henning_frac(ENV_task, slide_group_props_dict, slide_frac_dict, 
                                               gp_name_list=['A', 'B', 'C', 'D', 'N'],
                                               color_dict={'A': 'green', 'B': 'blue', 'C': 'orange', 
                                                           'D': 'red', 'N': 'lightgrey'}):
    '''
    stack bar plot
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    sns.set_style('whitegrid')  # setup seaborn fashion
    
    # x-axis sorted by fraction score from Henning
    sorted_slides = sorted(slide_frac_dict, key=slide_frac_dict.get)
    plt.figure(figsize=(15, 6))
    
    handles_dict = {}  # Initialize an empty dictionary to store unique handles for each cluster group
    first_gp_drawn = set()  # Set to keep track of which group names have been drawn
    
    x_axis_fracs = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    x_ticks = []  # List to store s_idx positions where x-axis labels will be shown
    x_labels = [] # List to store the corresponding labels for the x_ticks
    
    # Find s_idx for the first slide_id that exceeds each value in x_axis_fracs
    for frac in x_axis_fracs:
        for s_idx, slide_id in enumerate(sorted_slides):
            if slide_frac_dict[slide_id] >= frac:
                x_ticks.append(s_idx)
                x_labels.append(str(frac))  # Use the fraction value as the label
                break  # Move to the next fraction value once the first exceeding slide_id is found
    
    for s_idx, slide_id in enumerate(sorted_slides):
        if slide_id not in slide_group_props_dict.keys():
            continue
    
        bottom = 0
        for gp_name in gp_name_list:
            proportion = slide_group_props_dict[slide_id][gp_name]
            color = color_dict.get(gp_name, 'lightgrey')
            if f'group: {gp_name}' not in first_gp_drawn:
                handle = plt.bar(s_idx, proportion, bottom=bottom, color=color, label=f'group: {gp_name}')
                first_gp_drawn.add(f'group: {gp_name}')  # Mark this group name as drawn
                handles_dict[f'group: {gp_name}'] = handle  # Store the handle in the dictionary
            else:
                plt.bar(s_idx, proportion, bottom=bottom, color=color)  # Don't add label to avoid duplicate legend entries
            bottom += proportion
        
    # Create a legend using the handles and group names stored in the dictionary
    plt.legend([handle[0] for handle in handles_dict.values()], handles_dict.keys(), title='Cluster groups')
     
    plt.xticks(x_ticks, x_labels)  # Set custom x-axis tick positions and labels
    plt.ylabel('Group Proportions')
    plt.xlabel('Slides with pathologist\'s P62 fraction score, shows fraction scores(%): low -> high')
    plt.legend(title='Cluster groups')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.tight_layout()
    
    plt.xlim(0, 250)
    plt.ylim(0, 1.001)
    
    fig_name = f'clst_gp-{len(gp_name_list)}_props_sort_by_henning_frac.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'clst_group_props visualisation saved at {save_path}')
    plt.show()
    
def _plot_c_group_props_by_henning_frac(ENV_task, slide_tile_label_dict_filename, clst_gps):
    '''
    run stack bar plot for clst groups proportion
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    slide_tile_label_dict = load_vis_pkg_from_pkl(stat_store_dir, slide_tile_label_dict_filename)
    slide_ids = list(slide_tile_label_dict.keys())
    
    # load Henning's slide_frac_dict
    slide_frac_dict = {}
    case_frac_dict = metadata.load_percentages_from_csv(ENV_task)
    for s_id in slide_ids:
        case_id = parse_caseid_from_slideid(s_id)
        slide_frac_dict[s_id] = case_frac_dict[case_id]
    
    slide_group_props_dict = prep_stat_vis.proportion_clst_gp_on_each_slides(slide_tile_label_dict, clst_gps)
    plot_clst_group_props_sort_by_henning_frac(ENV_task, slide_group_props_dict, slide_frac_dict)
    

def plot_clsts_agts_corr_henning_frac(ENV_task, slide_agt_score_dict, slide_frac_dict, 
                                      gp_or_sp):
    '''
    plot the correlation between clusters' aggregation score and henning fraction score
    
    Args:
        gp_or_sp: name of cluster group or label of selected cluster
            I will give a group name list, of not in this list, 
            we know the input is for single cluster label
    PS:
        attention: slide_agt_score_dict's type must map to gp_or_sp
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    gp_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    
    data = []
    for slide_id, group_scores in slide_agt_score_dict.items():
        if gp_or_sp in group_scores:
            aggregation_score = group_scores[gp_or_sp]
            frac_score = slide_frac_dict.get(slide_id, 0)
            data.append((slide_id, aggregation_score, frac_score))
    
    # generate Pandas DataFrame
    df = pd.DataFrame(data, columns=['Slide ID', 'Aggregation Score', 'Frac Score'])
    
    plt.figure(figsize=(10, 6))
    # plot
    sns.regplot(x='Frac Score', y='Aggregation Score', data=df, order=2, scatter_kws={'s': 50})
    
    plt.title('Aggregation Score vs Frac Score for {}'.format(gp_or_sp))
    if gp_or_sp in gp_name_list:
        y_title_str = f'group: {gp_or_sp}'
    else:
        y_title_str = f'cluster: {gp_or_sp}'
    plt.ylabel(f'Aggregation Score for {y_title_str}')
    plt.xlabel('Pathologist\'s Frac Score')
    plt.ylim(-0.05, 0.3)
    plt.xlim(0, 5.0)
    
    fig_name = f'gp-{gp_or_sp}_agt_corr_henning_frac.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'clst-gp: {gp_or_sp} or sp-clst corr frac-score visualisation saved at {save_path}')
    # plt.show()
    
def _plot_c_gp_agts_corr_henning_frac(ENV_task, c_gp_aggregation_filename):
    '''
    run correlation curve for agt score <-> henning frac on clst groups
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    gp_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    slide_gp_agt_score_dict, name_gps  = load_vis_pkg_from_pkl(stat_store_dir, 
                                                               c_gp_aggregation_filename)
    slide_ids = list(slide_gp_agt_score_dict.keys())
    
    # load Henning's slide_frac_dict
    slide_frac_dict = {}
    case_frac_dict = metadata.load_percentages_from_csv(ENV_task)
    for s_id in slide_ids:
        case_id = parse_caseid_from_slideid(s_id)
        slide_frac_dict[s_id] = case_frac_dict[case_id]
        
    for i in range(len(name_gps)):
        print(f'plot correlation with frac-score for group: \n{name_gps[gp_name_list[i]]} >>>')
        clst_gp = gp_name_list[i]
        plot_clsts_agts_corr_henning_frac(ENV_task, slide_gp_agt_score_dict, slide_frac_dict, 
                                          gp_or_sp=clst_gp)
        
def _plot_sp_c_agts_corr_henning_frac(ENV_task, sp_c_aggregation_filename):
    '''
    run correlation curve for agt score <-> henning frac on specific clst
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    slide_spc_agt_score_dict = load_vis_pkg_from_pkl(stat_store_dir, sp_c_aggregation_filename)
    slide_ids = list(slide_spc_agt_score_dict.keys())
    sp_clsts = list(next(iter(slide_spc_agt_score_dict.values())).keys())
    
    # load Henning's slide_frac_dict
    slide_frac_dict = {}
    case_frac_dict = metadata.load_percentages_from_csv(ENV_task)
    for s_id in slide_ids:
        case_id = parse_caseid_from_slideid(s_id)
        slide_frac_dict[s_id] = case_frac_dict[case_id]
        
    for sp_c in sp_clsts:
        print(f'plot correlation with frac-score for cluster: \n{sp_c} >>>')
        plot_clsts_agts_corr_henning_frac(ENV_task, slide_spc_agt_score_dict, slide_frac_dict, 
                                          gp_or_sp=sp_c)
        
def plot_agt_dist_h_l_henning_frac(ENV_task,slide_agt_score_dict, slide_frac_dict, 
                                   c_gp, frac_thd):
    '''
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    # prepare the data
    scores_high_frac = []  # aggregation_score list for frac_score >= frac_thd
    scores_low_frac = []   # aggregation_score list for frac_score < frac_thd

    for slide_id, group_scores in slide_agt_score_dict.items():
        # check if slide_id has specific c_gp's aggregation_score we want
        if c_gp in group_scores:
            aggregation_score = group_scores[c_gp]
            frac_score = slide_frac_dict.get(slide_id, 0)
            
            # split different frac_score sets
            if frac_score >= frac_thd:
                scores_high_frac.append(aggregation_score)
            else:
                scores_low_frac.append(aggregation_score)
    
    # plot
    plt.figure(figsize=(8, 6))
    sns.kdeplot(scores_high_frac, color="red", fill=True, alpha=0.3, 
                label=f'Slides with fraction >= {frac_thd}')
    sns.kdeplot(scores_low_frac, color="blue", fill=True, alpha=0.3, 
                label=f'Slides with fraction < {frac_thd}')


    plt.title(f'Aggregation Score Distribution for Group {c_gp}')
    plt.xlabel('Aggregation Score')
    plt.ylabel('Density')
    plt.legend()
    
    fig_name = f'gp-{c_gp}_dist_h-l-frac-thd-{frac_thd}.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'clst-gp: {c_gp} or sp-clst corr frac-score visualisation saved at {save_path}')
    # plt.show()

def _plot_c_gp_agts_dist_h_l_frac(ENV_task, c_gp_aggregation_filename, frac_thd=0.2):
    '''
    h_l_frac means higher or lower a threshold on fraction 
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    gp_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    slide_gp_agt_score_dict, name_gps  = load_vis_pkg_from_pkl(stat_store_dir, 
                                                               c_gp_aggregation_filename)
    slide_ids = list(slide_gp_agt_score_dict.keys())
    
    # load Henning's slide_frac_dict
    slide_frac_dict = {}
    case_frac_dict = metadata.load_percentages_from_csv(ENV_task)
    for s_id in slide_ids:
        case_id = parse_caseid_from_slideid(s_id)
        slide_frac_dict[s_id] = case_frac_dict[case_id]
        
    for i in range(len(name_gps)):
        print(f'plot agt distribution higher/lower frac-score for group: \n{name_gps[gp_name_list[i]]} >>>')
        clst_gp = gp_name_list[i]
        plot_agt_dist_h_l_henning_frac(ENV_task, slide_gp_agt_score_dict, slide_frac_dict,
                                       c_gp=clst_gp, frac_thd=frac_thd)

if __name__ == '__main__':
    pass