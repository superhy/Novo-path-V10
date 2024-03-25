'''
Created on 24 Mar 2024

@author: super
'''

import os

from interpre.prep_tools import load_vis_pkg_from_pkl
import matplotlib.pyplot as plt
import seaborn as sns
from support import metadata
from support.files import parse_caseid_from_slideid
from interpre import prep_stat_vis


def plot_clst_group_props_sort_by_henning_frac(ENV_task, slide_group_props_dict, slide_frac_dict, 
                                               gp_name_list=['A', 'B', 'C', 'D', 'N'],
                                               color_dict={'A': 'green', 'B': 'blue', 'C': 'orange', 
                                                           'D': 'red', 'N': 'grey'}):
    '''
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    sns.set()  # setup seaborn fashion
    
    # x-axis sorted by fraction score from Henning
    sorted_slides = sorted(slide_frac_dict, key=slide_frac_dict.get)
    plt.figure(figsize=(10, 6))
    
    handles, labels = [], []
    for s_idx, slide_id in enumerate(sorted_slides):
        if slide_id not in slide_group_props_dict.keys():
            continue
        
        bottom = 0
        for gp_name in gp_name_list:
            proportion = slide_group_props_dict[slide_id][gp_name]
            color = color_dict.get(gp_name, 'grey')
            handle = plt.bar(s_idx, proportion, bottom=bottom, color=color, label=f'c-group: {gp_name}')
            handles.append(handle)
            labels.append(f'c-group: {gp_name}')
            bottom += proportion
            
    # create legend for all cluster group names
    unique_labels, unique_handles = [], []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    plt.legend(unique_handles, unique_labels)
            
    plt.ylabel('Group Proportions')
    plt.xlabel('Slide ID')
    plt.legend(title='Cluster groups')
    plt.tight_layout()
    
    fig_name = f'clst_gp-{len(gp_name_list)}_props_sort_by_henning_frac.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'clst_group_props visualisation saved at {save_path}')
    
    # plt.show()
    
def _plot_c_group_props_by_henning_frac(ENV_task, slide_tile_label_dict_filename, clst_gps):
    '''
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

if __name__ == '__main__':
    pass