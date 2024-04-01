'''
Created on 24 Mar 2024

@author: super
'''

import gc
import os

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats.stats import pearsonr

from interpre import prep_stat_vis
from interpre.prep_tools import load_vis_pkg_from_pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from support import metadata, tools
from support.files import parse_caseid_from_slideid


def plot_clst_group_props_sort_by_henning_frac(ENV_task, slide_group_props_dict, slide_frac_dict, 
                                               gp_name_list=['A', 'B', 'C', 'D', 'N'],
                                               color_dict={'A': 'green', 'B': 'blue', 'C': 'orange', 
                                                           'D': 'red', 'N': 'lightgrey'}):
    '''
    stack bar plot
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    sns.set_style('whitegrid')  # setup seaborn fashion
    
    # Set global font size
    plt.rcParams.update({'font.size': 15})
    # x-axis sorted by fraction score from Henning
    sorted_slides = sorted(slide_frac_dict, key=slide_frac_dict.get)
    plt.figure(figsize=(18, 4.5))
    
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
    
    plt.xlim(0, 251)
    plt.ylim(0, 1.001)
    
    fig_name = f'clst_gp-{len(gp_name_list)}_props_sort_by_henning_frac.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'clst_group_props visualisation saved at {save_path}')
    # plt.show()
    
def plot_c_gps_props_dist_in_slides(ENV_task, slide_group_props_dict, gp_name, color):
    '''
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    # Prepare the data: extract proportion values for the specified group_name across all slide_ids
    proportions = []
    for slide_id, groups in slide_group_props_dict.items():
        if gp_name in groups:
            proportions.append(groups[gp_name])  # Append the proportion value of the specified group

    # Convert the list to a DataFrame for seaborn compatibility
    data = pd.DataFrame(proportions, columns=['Proportion'])

    # Set global font size
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(8, 5))
    # Use seaborn's displot to plot the distribution with a KDE curve
    sns.histplot(data, x='Proportion', bins=80, kde=True, color=color)
    # sns.displot(data, x='Proportion', kind='kde')

    # Add title and labels for clarity
    plt.title(f'Group: {gp_name}\'s proportion in slides')
    plt.xlabel('Proportion Value')
    plt.ylabel('Density')
    plt.tight_layout()

    fig_name = f'clst_gp-{gp_name}_props_dist_in_slides.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'Visualisation of clst_group\' props distribution in slides, saved at {save_path}')
    # plt.show()
    
def plot_c_gp_props_box_in_diff_slides(ENV_task, slide_group_props_dict, gp_name,
                                       slide_parts, s_part_names):
    '''
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    # preparing the data
    data = []
    for part, part_name in zip(slide_parts, s_part_names):
        for slide_id in part:
            if slide_id in slide_group_props_dict and gp_name in slide_group_props_dict[slide_id]:
                data.append([slide_group_props_dict[slide_id][gp_name], part_name])

    # produce the dataframe
    df = pd.DataFrame(data, columns=['Proportion', 'Part Name'])

    # plot
    # Set global font size
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(4, 8))
    # sns.boxplot(x='Part Name', y='Proportion', data=df, linewidth=2, 
    #             boxprops=dict(facecolor='none') )
    sns.boxplot(x='Part Name', y='Proportion', data=df, linewidth=2)

    # put title and x,y axis
    plt.title(f'Group {gp_name} proportion in\ndiff-parts of slides')
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.ylabel('Proportion Value')
    plt.tight_layout()

    fig_name = f'clst_gp-{gp_name}_props_box_diff_parts_s.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'Visualisation of clst_group\' props box in different part of slides, saved at {save_path}')
    # plt.show()
    
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
    
def _plot_c_gps_props_dist_in_slides(ENV_task, slide_tile_label_dict_filename, clst_gps, gp_names,
                                     colors=['green', 'blue', 'orange', 'red']):
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
    
    for i, gp_name in enumerate(gp_names):
        plot_c_gps_props_dist_in_slides(ENV_task, slide_group_props_dict, gp_name, colors[i])
    
def _plot_c_gp_props_box_in_diff_slides_025(ENV_task, slide_tile_label_dict_filename, clst_gps, gp_names):
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
    
    slide_parts, s_part_names = [[], [], []], ['frac < 0.2', '0.2 <= frac <= 0.5', '0.5 < frac']
    for slide_id, frac_value in slide_frac_dict.items():
        if frac_value < 0.2:
            slide_parts[0].append(slide_id)
        elif frac_value >= 0.2 and frac_value <= 0.5:
            slide_parts[1].append(slide_id)
        else:
            slide_parts[2].append(slide_id)
    
    for i, gp_name in enumerate(gp_names):
        plot_c_gp_props_box_in_diff_slides(ENV_task, slide_group_props_dict, gp_name, 
                                           slide_parts, s_part_names)
    

def plot_clsts_agts_corr_henning_frac(ENV_task, slide_agt_score_dict, slide_frac_dict, 
                                      gp_or_sp,
                                      colors = ['lightseagreen', 'royalblue', 'blueviolet'],
                                      xlims = [(0, 0.2), (0.2, 1.0), (1.0, 5.0)],
                                      ylims = [(-0.001, 0.1), (-0.001, 0.2), (-0.001, 0.3)]):
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
    # colors = ['green', 'blue', 'red']  # Colors for the subplots
    gp_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'All']
    
    data = []
    for slide_id, group_scores in slide_agt_score_dict.items():
        if gp_or_sp in group_scores:
            aggregation_score = group_scores[gp_or_sp]
            frac_score = slide_frac_dict.get(slide_id, 0)
            data.append((slide_id, aggregation_score, frac_score))
    
    df = pd.DataFrame(data, columns=['Slide ID', 'Aggregation Score', 'Frac Score'])
    corr, p_value = pearsonr(df['Aggregation Score'], df['Frac Score'])
    print(f"{gp_or_sp}: {corr:.2f}, p_value: {p_value}")
    
    # Set global font size
    plt.rcParams.update({'font.size': 15})
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=False)
    
    # xlims = [(0, 0.2), (0.2, 1.0), (1.0, 5.0)]
    # ylims = [(-0.001, 0.1), (-0.001, 0.2), (-0.001, 0.3)]
    if gp_or_sp in gp_name_list:
        y_title_str = f'group: {gp_or_sp}'
    else:
        y_title_str = f'cluster: {gp_or_sp}'
        
    for i, (xlim, ylim) in enumerate(zip(xlims, ylims)):
        ax = axes[i]
        sns.regplot(x='Frac Score', y='Aggregation Score', data=df, ax=ax, order=2, scatter_kws={'s': 50}, color=colors[i])
        
        ax.set_xlim(xlim) # Set custom xlim for each subplot
        ax.set_ylim(ylim) # Set custom ylim for each subplot
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Set x-axis tick labels to have 2 decimal places
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Set y-axis tick labels to have 2 decimal places
        xlim_0, xlim_1 = xlim[0], xlim[1]
        ax.set_title(f'Fraction range: {xlim_0} to {xlim_1}')
        ax.set_xlabel('Pathologist\'s Fraction Score')
        
        if i == 0:  # Only add y-label to the first subplot
            ax.yaxis.label.set_visible(True)
            ax.set_ylabel(f'Aggregation Score for {y_title_str}')
        else:
            ax.yaxis.label.set_visible(False)
    
    plt.suptitle(f'Aggregation Score vs Fraction Score for {y_title_str}')  # Overall title for all subplots
    # embed the text of pearson score to the figures
    fig.text(0.6, 0.84, f'Pearson Correlation: {corr:.2f}', 
             ha='center', va='top', fontsize=20, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout(rect=[0, 0.01, 1, 0.99])  # Adjust layout to make room for the overall title
    
    fig_name = f'clst_gp-{gp_or_sp}_agt_corr_henning_frac.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'clst-gp: {gp_or_sp} or sp-clst corr frac-score visualisation saved at {save_path}')
    # plt.show()
    
def _plot_c_gp_agts_corr_henning_frac(ENV_task, c_gp_aggregation_filename,
                                      colors = ['lightseagreen', 'royalblue', 'blueviolet'],
                                      xlims = [(0, 0.2), (0.2, 1.0), (1.0, 5.0)],
                                      ylims = [(-0.001, 0.1), (-0.001, 0.2), (-0.001, 0.3)]):
    '''
    run correlation curve for agt score <-> henning frac on clst groups
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    slide_gp_agt_score_dict, name_gps  = load_vis_pkg_from_pkl(stat_store_dir, 
                                                               c_gp_aggregation_filename)
    gp_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'] if len(name_gps) > 1 else ['All']
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
                                          gp_or_sp=clst_gp,
                                          colors=colors,
                                          xlims=xlims, ylims=ylims)
        
def _plot_sp_c_agts_corr_henning_frac(ENV_task, sp_c_aggregation_filename,
                                      colors = ['lightseagreen', 'royalblue', 'blueviolet'],
                                      xlims = [(0, 0.2), (0.2, 1.0), (1.0, 5.0)],
                                      ylims = [(-0.001, 0.1), (-0.001, 0.2), (-0.001, 0.3)]):
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
                                          gp_or_sp=sp_c,
                                          colors=colors,
                                          xlims=xlims, ylims=ylims)
        
def plot_agt_dist_h_l_henning_frac(ENV_task,slide_agt_score_dict, slide_frac_dict, 
                                   c_gp, frac_thd,
                                   colors = ['salmon', 'lightseagreen']):
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
    
    # Set global font size
    plt.rcParams.update({'font.size': 15})
    
    # plot
    plt.figure(figsize=(6, 6))
    sns.kdeplot(scores_high_frac, color=colors[0], fill=True, alpha=0.3, 
                label=f'Slides with fraction >= {frac_thd}')
    sns.kdeplot(scores_low_frac, color=colors[1], fill=True, alpha=0.3, 
                label=f'Slides with fraction < {frac_thd}')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # plt.xlim(-0.1, 0.8)
    plt.title(f'Aggregation Score Distribution for Group {c_gp}')
    plt.xlabel('Aggregation Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    fig_name = f'gp-{c_gp}_dist_h-l-frac-thd-{frac_thd}.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'clst-gp: {c_gp} or sp-clst corr frac-score visualisation saved at {save_path}')
    # plt.show()

def _plot_c_gp_agts_dist_h_l_frac(ENV_task, c_gp_aggregation_filename, 
                                  frac_thd=0.2, colors = ['salmon', 'lightseagreen']):
    '''
    h_l_frac means higher or lower a threshold on fraction 
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    slide_gp_agt_score_dict, name_gps  = load_vis_pkg_from_pkl(stat_store_dir, 
                                                               c_gp_aggregation_filename)
    gp_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'] if len(name_gps) > 1 else ['All']
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
                                       c_gp=clst_gp, frac_thd=frac_thd, colors=colors)

        
def plot_agt_score_heatmap(ENV_task, slide_agt_score_dict, slide_frac_dict, label_sort, 
                           y_markers=[], frac_markers=[]):
    '''
    Args:
        label_sort = [[clst_labels], [gp_labels], ['All']]
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    # Extract all agt_score for normalization
    all_agt_scores = []
    for slide_id, scores in slide_agt_score_dict.items():
        for clst_label, agt_score in scores.items():
            all_agt_scores.append(agt_score)
    
    all_agt_scores_array = np.array(all_agt_scores)
    normalized_agt_scores = tools.normalization(all_agt_scores_array)
    
    # Reconstruct slide_agt_score_dict using normalized agt_score
    normalized_slide_spc_agt_score_dict = {}
    pos = 0  # index of normalized_agt_scores
    for slide_id, scores in slide_agt_score_dict.items():
        normalized_scores = {}
        for clst_label in scores:
            normalized_scores[clst_label] = normalized_agt_scores[pos]
            pos += 1
        normalized_slide_spc_agt_score_dict[slide_id] = normalized_scores
    
    sorted_slides = sorted(slide_frac_dict, key=slide_frac_dict.get)
    frac_i, x_marker_ts = 0, []
    for pos, s_id in enumerate(sorted_slides):
        if frac_i >= len(frac_markers):
            break
        if slide_frac_dict[s_id] > frac_markers[frac_i]:
            x_marker_ts.append((pos, frac_markers[frac_i]))
            frac_i += 1
    print(x_marker_ts)
    
    data = []
    # for slide_id, scores in slide_agt_score_dict.items():
    for slide_id in sorted_slides:
        if slide_id in slide_agt_score_dict.keys():
            # print(slide_id, slide_frac_dict[slide_id])
            # scores = slide_agt_score_dict[slide_id]
            scores = normalized_slide_spc_agt_score_dict[slide_id]
            print(slide_id)
            # for clst_label, agt_score in scores.items():
            for clst_label in label_sort:
                agt_score = scores[clst_label]
                data.append({'Slide ID': slide_id, 'Cluster Label': clst_label, 'Aggregation Score': agt_score})
    df = pd.DataFrame(data)
    
    if df.duplicated(subset=['Slide ID', 'Cluster Label']).any():
        print("Warning: Duplicated rows found. Please check the input data.")
        df = df.drop_duplicates(subset=['Slide ID', 'Cluster Label'])
    
    # transfer the data format for heatmap
    heatmap_data = df.pivot("Cluster Label", "Slide ID", "Aggregation Score").reindex(index=label_sort, 
                                                                                      columns=sorted_slides)
    print(heatmap_data)
    
    plt.rcParams.update({'font.size': 40})
    # plot
    # print(len(list(slide_agt_score_dict.items())[0][1]) )
    plt.figure(figsize=(60, 0.75 * len(list(slide_agt_score_dict.items())[0][1]) ))
    ax = sns.heatmap(heatmap_data, annot=False, cmap="rocket_r", linewidths=0.5, linecolor='white',
                     cbar_kws={'shrink': 1, 'pad': 0.01})
    plt.title('Aggregation score (normalised) heatmap')
    
    plt.yticks([])
    plt.ylabel('')
    plt.xticks([])
    ax.set_xlabel('Slides with pathologist\'s P62 fraction score, left -> right with fraction scores(%): low -> high', 
                  labelpad=80)
    
    total_rows = len(label_sort)
    for marker in y_markers:
        relative_marker_position = (total_rows - marker) / total_rows
        if marker > 0:
            ax.axhline(y=marker, color='black', linewidth=5)
        left_extension = Line2D([-0.01, 0], [relative_marker_position, relative_marker_position],
                                color='black', linewidth=5, transform=ax.transAxes, clip_on=False)
        ax.add_line(left_extension)
    for pos, frac_score in x_marker_ts:
        relative_x_position = pos / len(sorted_slides)
        # ax.axvline(x=pos, color='black', linewidth=0.5, ymin=1, ymax=1.2)  # down x-tick
        ax.text(x=pos-0.5, y=total_rows+1, s=frac_score, ha='center', va='top')
        down_extension = Line2D([relative_x_position, relative_x_position], [-0.025, 0], color='black', 
                                linewidth=5, transform=ax.transAxes, clip_on=False)
        ax.add_line(down_extension)
        # ax.text(x=pos, y=total_rows + 0.5, s=frac_score, ha='center', va='top', 
        #         transform=ax.get_xaxis_transform())
        
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(8)
        spine.set_edgecolor('black')
    # plt.xlim(0, len(sorted_slides))
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.2)
    plt.tight_layout()
    
    fig_name = f'spc-gps-{len(label_sort)}_agt_score_heatmap-{y_markers}.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'spc (and gps) aggregation score heatmap visualisation saved at {save_path}')
    # plt.show()
    
def _plot_spc_agt_heatmap_by_henning_frac(ENV_task, sp_c_aggregation_filename):
    '''
    run stack bar plot for clst groups proportion
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    slide_spc_agt_score_dict = load_vis_pkg_from_pkl(stat_store_dir, sp_c_aggregation_filename)
    slide_ids = list(slide_spc_agt_score_dict.keys())
    
    label_sort = list(slide_spc_agt_score_dict[slide_ids[0]].keys() )
    
    # load Henning's slide_frac_dict
    slide_frac_dict = {}
    case_frac_dict = metadata.load_percentages_from_csv(ENV_task)
    for s_id in slide_ids:
        case_id = parse_caseid_from_slideid(s_id)
        slide_frac_dict[s_id] = case_frac_dict[case_id]
        
    plot_agt_score_heatmap(ENV_task, slide_spc_agt_score_dict, slide_frac_dict, label_sort)
    
def combine_dicts(dicts):
    '''
    '''
    combined_dict = {}
    for d in dicts:
        for slide_id, scores in d.items():
            if slide_id not in combined_dict:
                combined_dict[slide_id] = scores
            else:
                combined_dict[slide_id].update(scores)
    return combined_dict

def _plot_spc_gps_agt_heatmap_by_henning_frac(ENV_task, spc_gps_agt_filename_list,
                                              given_y_markers=None):
    '''
    run stack bar plot for clst groups proportion
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    slide_agt_score_dicts = []
    for agt_fname in spc_gps_agt_filename_list:
        slide_lbl_agt_dict = load_vis_pkg_from_pkl(stat_store_dir, agt_fname)
        if isinstance(slide_lbl_agt_dict, tuple):
            label_prefix = agt_fname[agt_fname.find('rad'): agt_fname.find('rad') + 4]
            slide_lbl_agt_dict = slide_lbl_agt_dict[0]
            # avoid repeat label
            updated_dict = {}
            for slide_id, lbl_agt_scores in slide_lbl_agt_dict.items():
                updated_lbl_agt_scores = {label_prefix + label: agt_score for label, agt_score in lbl_agt_scores.items()}
                updated_dict[slide_id] = updated_lbl_agt_scores
            slide_lbl_agt_dict = updated_dict
            del updated_dict
            
        # print(slide_lbl_agt_dict['23910-158_Sl064-C57-P62'].keys())
        slide_agt_score_dicts.append(slide_lbl_agt_dict)
    
    gc.collect()
        
    slide_ids = list(slide_agt_score_dicts[0].keys())
        
    label_sort, y_markers = [], [0]
    for agt_dict in slide_agt_score_dicts:
        clst_gp_labels = list(agt_dict[slide_ids[0]].keys())
        # print(agt_dict[slide_ids[0]].keys())
        label_sort.extend(clst_gp_labels)
        y_markers.append(y_markers[-1] + len(clst_gp_labels))
    if given_y_markers is not None:
        y_markers = given_y_markers
    # print(label_sort)
    # print(y_markers)
    
    slide_spc_gps_agt_score_dict = combine_dicts(slide_agt_score_dicts)
    # print(slide_spc_gps_agt_score_dict[slide_ids[0]].keys())

    # load Henning's slide_frac_dict
    slide_frac_dict = {}
    case_frac_dict = metadata.load_percentages_from_csv(ENV_task)
    for s_id in slide_ids:
        case_id = parse_caseid_from_slideid(s_id)
        slide_frac_dict[s_id] = case_frac_dict[case_id]
      
    frac_markers = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]  
    plot_agt_score_heatmap(ENV_task, slide_spc_gps_agt_score_dict, slide_frac_dict, 
                           label_sort, y_markers, frac_markers)

if __name__ == '__main__':
    pass