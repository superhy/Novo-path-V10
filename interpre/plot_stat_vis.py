'''
Created on 24 Mar 2024

@author: super
'''

import csv
import gc
import os

from PIL import Image, ImageDraw, ImageFont
from cmapy import cmap
from matplotlib import patches
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import pearsonr

from interpre import prep_stat_vis
from interpre.prep_stat_vis import save_slide_group_props_to_csv
from interpre.prep_tools import load_vis_pkg_from_pkl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import umap

import seaborn as sns
from support import metadata, tools
from support.files import parse_caseid_from_slideid, \
    parse_23910_clinicalid_from_slideid


# from scipy.stats.stats import pearsonr
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
    plt.xlabel('Slides with P62 preliminary dark-area fraction score, shows fraction scores(%): low -> high')
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
    save_slide_group_props_to_csv(ENV_task, slide_group_props_dict)
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
        
def plot_agt_dist_h_l_henning_frac2(ENV_task,slide_agt_score_dict, slide_frac_dict, 
                                   c_gp, frac_thd,
                                   colors = ['salmon', 'lightseagreen', ]):
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
    
    fig_name = f'gp-{c_gp}_dist_h-l-frac2-thd-{frac_thd}.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'clst-gp: {c_gp} or sp-clst corr frac2-score visualisation saved at {save_path}')
    # plt.show()
    
def plot_agt_dist_h_l_henning_frac3(ENV_task,slide_agt_score_dict, slide_frac_dict, 
                                   c_gp, frac_thd_tuple,
                                   colors = ['magenta', 'salmon', 'lightseagreen']):
    '''
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    # prepare the data
    scores_high_frac = [] # aggregation_score list for frac_score >= frac_thd[1]
    scores_mid_frac = [] # aggregation_score list for frac_thd[0] <= frac_score < frac_thd[1]
    scores_low_frac = [] # aggregation_score list for frac_score < frac_thd[0]

    for slide_id, group_scores in slide_agt_score_dict.items():
        # check if slide_id has specific c_gp's aggregation_score we want
        if c_gp in group_scores:
            aggregation_score = group_scores[c_gp]
            frac_score = slide_frac_dict.get(slide_id, 0)
            
            # split different frac_score sets
            if frac_score >= frac_thd_tuple[2]:
                scores_high_frac.append(aggregation_score)
            elif frac_score >= frac_thd_tuple[1] and frac_score < frac_thd_tuple[2]:
                scores_mid_frac.append(aggregation_score)
            elif frac_score >= frac_thd_tuple[0] and frac_score < frac_thd_tuple[1]:
                scores_low_frac.append(aggregation_score)
    
    # Set global font size
    plt.rcParams.update({'font.size': 15})
    
    # plot
    plt.figure(figsize=(6, 6))
    sns.kdeplot(scores_high_frac, color=colors[0], fill=True, alpha=0.3, 
                label=f'Slides with fraction >= {frac_thd_tuple[2]}')
    sns.kdeplot(scores_mid_frac, color=colors[1], fill=True, alpha=0.3, 
                label=f'{frac_thd_tuple[1]} <= Slides with fraction < {frac_thd_tuple[2]}')
    sns.kdeplot(scores_low_frac, color=colors[2], fill=True, alpha=0.3, 
                label=f'{frac_thd_tuple[0]} <= Slides with fraction < {frac_thd_tuple[1]}')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # plt.xlim(-0.0, 1.0)
    plt.title(f'Aggregation Score Distribution for Group {c_gp}')
    plt.xlabel('Aggregation Score')
    plt.xticks(rotation=30)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    fig_name = f'gp-{c_gp}_dist_h-l-frac3-thd-{frac_thd_tuple}.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'clst-gp: {c_gp} or sp-clst corr frac3-score visualisation saved at {save_path}')
    # plt.show()

def _plot_c_gp_agts_dist_h_l_frac2(ENV_task, c_gp_aggregation_filename, 
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
        plot_agt_dist_h_l_henning_frac2(ENV_task, slide_gp_agt_score_dict, slide_frac_dict,
                                       c_gp=clst_gp, frac_thd=frac_thd, colors=colors)
        
def _plot_c_gp_agts_dist_h_l_frac3(ENV_task, c_gp_aggregation_filename, 
                                   frac_thd_tuple=(0, 0.2, 0.5), 
                                   colors = ['magenta', 'salmon', 'lightseagreen']):
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
        plot_agt_dist_h_l_henning_frac3(ENV_task, slide_gp_agt_score_dict, slide_frac_dict,
                                        c_gp=clst_gp, frac_thd_tuple=frac_thd_tuple, colors=colors)

        
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
    # print(df)
    
    if df.duplicated(subset=['Slide ID', 'Cluster Label']).any():
        print("Warning: Duplicated rows found. Please check the input data.")
        df = df.drop_duplicates(subset=['Slide ID', 'Cluster Label'])
    
    # transfer the data format for heatmap
    heatmap_data = df.pivot(index="Cluster Label", 
                            columns="Slide ID", 
                            values="Aggregation Score").reindex(index=label_sort, columns=sorted_slides)
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
    ax.set_xlabel('Slides with P62 preliminary dark-area fraction score, left -> right with fraction scores(%): low -> high', 
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
    
    
def load_slide_ids_from_vis_pkg(ENV_task, any_vis_pkg_name):
    '''
    load the the slide_ids with statistic results
    and generate cohort slide_id -> marta's case_id dict to mapping marta's results
    '''
    
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    slide_spc_agt_score_dict = load_vis_pkg_from_pkl(stat_store_dir, any_vis_pkg_name)
    slide_ids = list(slide_spc_agt_score_dict.keys())
    
    cohort_s_marta_p_dict, marta_p_cohort_s_dict = {}, {}
    for slide_id in slide_ids:
        cohort_s_id = parse_caseid_from_slideid(slide_id)
        cohort_p_id = parse_23910_clinicalid_from_slideid(slide_id)
        if cohort_p_id.startswith('HV'):
            marta_p_id = cohort_p_id.replace('-', '_')
        else:
            marta_p_id = f'P_{cohort_p_id}'
        
        cohort_s_marta_p_dict[cohort_s_id] = marta_p_id
        marta_p_cohort_s_dict[marta_p_id] = cohort_s_id
        
    return cohort_s_marta_p_dict, marta_p_cohort_s_dict

def _rewrite_csv_with_slide_id(ENV_task, old_csv_name, marta_p_cohort_s_dict):
    '''
    rewrite the csv file for aligning the column id of Yang and Marta
    '''
    
    csv_file_path = os.path.join(ENV_task.META_FOLDER, old_csv_name)
    output_csv_path = os.path.join(ENV_task.META_FOLDER, old_csv_name.replace('.csv', '_slideid.csv') )
    
    updated_rows = []
    
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        # check if 'case_ID' exist in csv file
        if 'case_ID' not in reader.fieldnames:
            raise ValueError("CSV does not contain 'case_ID' column.")
        
        # change the column title
        fieldnames = [fn if fn != 'case_ID' else 'slide_id' for fn in reader.fieldnames]
        
        # check every line
        for row in reader:
            p_id = row['case_ID']
            if p_id in marta_p_cohort_s_dict:
                # replace patient_id to slide_id
                row['slide_id'] = marta_p_cohort_s_dict[p_id]
                del row['case_ID']
                updated_rows.append(row)
                print(row)
            # if p_id not in the dict，just skip
    ''' sort by slide_id '''
    updated_rows.sort(key=lambda x: x['slide_id'])
    
    # write the new CSV file
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    print(f'rewrote csv file for p_id -> slide_id, \nfrom {csv_file_path} to {output_csv_path}...')

def plot_corr_between_items_from_2csv(ENV_task, cohort_s_marta_p_dict, 
                                      x_csv_filename, y_csv_filename,
                                      x_col_n, y_col_n,
                                      x_thd_la=0.5):
    '''
    plot the correlation for values of 2 columns from 2 csv files
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    x_idx, y_idx = 'slide_id', 'case_ID'
    shown_name_dict = {'A': 'P62-group-A',
                       'B': 'P62-group-B',
                       'C': 'P62-group-C',
                       'D': 'P62-group-D',
                       'all': 'P62-all_sensi_area',
                       '0': 'Fibrosis-cluster_0',
                       '1': 'Fibrosis-cluster_1',
                       '2': 'Fibrosis-cluster_2',
                       '3': 'Fibrosis-cluster_3',
                       '4': 'Fibrosis-cluster_4',
                       '5': 'Fibrosis-cluster_5',
                       '6': 'Fibrosis-cluster_6'}
    
    # Load CSV files
    x_df = pd.read_csv(os.path.join(ENV_task.META_FOLDER, x_csv_filename) )
    y_df = pd.read_csv(os.path.join(ENV_task.META_FOLDER, y_csv_filename) )
    # print(x_df)
    # print(y_df)
    
    # Filter rows based on cohort_s_marta_p_dict mappings
    filtered_x_df = x_df[x_df[x_idx].isin(cohort_s_marta_p_dict.keys())]
    filtered_y_df = y_df[y_df[y_idx].isin(cohort_s_marta_p_dict.values())]
    # print(filtered_x_df)
    # print(filtered_y_df)
    
    # Create a new DataFrame to store matched rows based on cohort_s_marta_p_dict
    matched_df = pd.DataFrame(columns=[x_col_n, y_col_n])
    
    # Iterate through the filtered_x_df and find matching rows in filtered_y_df based on the dictionary mapping
    for x_id in filtered_x_df[x_idx]:
        y_id = cohort_s_marta_p_dict[x_id] if x_idx != y_idx else x_id
        if y_id in filtered_y_df[y_idx].values:
            x_value = filtered_x_df[filtered_x_df[x_idx] == x_id][x_col_n].values[0]
            y_value = filtered_y_df[filtered_y_df[y_idx] == y_id][y_col_n].values[0]
            if pd.isna(x_value) or pd.isna(y_value) or x_value < x_thd_la:
                print('found NaN, skip...')
                continue
            matched_df = matched_df.append({x_col_n: x_value, y_col_n: y_value}, ignore_index=True)
    # print(matched_df, type(matched_df))
    
    # Get fake names for the axes if provided
    x_label = shown_name_dict.get(x_col_n, x_col_n)
    y_label = shown_name_dict.get(y_col_n, y_col_n)
    
    plt.figure(figsize=(6, 6))
    # Plotting with regplot
    sns.regplot(data=matched_df, x=x_col_n, y=y_col_n)
    plt.title(f'Correlation between {x_label} and {y_label}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    print(matched_df[x_col_n])
    print(matched_df[y_col_n])
    # Calculate Pearson correlation coefficient
    pearson_coef, p_value = pearsonr(matched_df[x_col_n], matched_df[y_col_n])
    print(f'Pearson correlation coefficient: {pearson_coef}')
    # Annotate the plot with the Pearson correlation coefficient
    plt.annotate(f'Pearson = {pearson_coef:.2f}', xy=(0.1, 0.9), xycoords='axes fraction')
    
    fig_name = f'corr_{x_label}-to-{y_label}.png'
    save_path = os.path.join(stat_store_dir, fig_name)
    plt.savefig(save_path)
    print(f'correlation of {x_label} and {y_label} visualisation saved at {save_path}')
    # plt.show()
    
''' ------------- correlation calculate ------------- '''
def merge_csv_files(csv_file_paths):
    '''
    merge different csv information into a big dataframe, using slide_id as the unique ID
    '''
    
    combined_df = None
    for file_path in csv_file_paths:
        # read CSV
        df = pd.read_csv(file_path)
        # ensure slide_id as the only index
        df.set_index('slide_id', inplace=True)
        
        if combined_df is None:
            # initialise combined_df
            combined_df = df
        else:
            # merge according to slide_id
            combined_df = combined_df.join(df, how='left', rsuffix='_dup')

    # remove the possible repeated columns
    combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('_dup')]
    # set slide_id back as a normal index
    combined_df.reset_index(inplace=True)
    print(f'combined DataFrame as:\n{combined_df}')
    
    return combined_df

def give_column_names(df):
    '''
    replace some stupid column names
    '''
    
    shown_name_dict = {'A': 'P62-group-A',
                       'B': 'P62-group-B',
                       'C': 'P62-group-C',
                       'D': 'P62-group-D',
                       'all': 'P62-all_sensi_area',
                       '0': 'Fibrosis-cluster_0',
                       '1': 'Fibrosis-cluster_1',
                       '2': 'Fibrosis-cluster_2',
                       '3': 'Fibrosis-cluster_3',
                       '4': 'Fibrosis-cluster_4',
                       '5': 'Fibrosis-cluster_5',
                       '6': 'Fibrosis-cluster_6',
                       'ballooning_percentage': 'Henning_dark_p62_frac'}
    
    new_columns = [shown_name_dict.get(col, col) for col in df.columns]
    df.columns = new_columns
    print('replaced the nick name for some columns')
    
    return df

# def calculate_pearson_correlation(df):
#     '''
#     '''
#
#     # exclude the column of slide_id and NaN lines
#     if 'slide_id' in df.columns:
#         df = df.drop('slide_id', axis=1)
#     df = df.dropna()
#
#     # calculation of pearson_coef
#     correlation_matrix = df.corr()
#     # setup cross value as 0
#     for i in range(len(correlation_matrix)):
#         correlation_matrix.iloc[i, i] = 0
#     print('calculated the Pearson correlation score')    
#
#     return correlation_matrix
def calculate_pearson_correlation(df):
    '''
    '''
    # exclude the column of slide_id and NaN lines
    if 'slide_id' in df.columns:
        df = df.drop('slide_id', axis=1)
    df = df.dropna()
    
    cols = df.columns
    corr_matrix = pd.DataFrame(data=np.zeros((len(cols), len(cols))), columns=cols, index=cols)
    p_value_matrix = pd.DataFrame(data=np.zeros((len(cols), len(cols))), columns=cols, index=cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j:
                corr, p_value = pearsonr(df[cols[i]].dropna(), df[cols[j]].dropna())
                corr_matrix.loc[cols[i], cols[j]] = corr
                p_value_matrix.loc[cols[i], cols[j]] = p_value
            else:
                corr_matrix.loc[cols[i], cols[j]] = 1
                p_value_matrix.loc[cols[i], cols[j]] = 0

    return corr_matrix, p_value_matrix, len(df)

class HandlerCrossInBox(HandlerBase):
    '''
    create the cross for plotting legend
    '''
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Ensure the box is more square-like by adjusting the dimensions
        box_size = min(width, height) * 1.2
        center = 0.5 * width - xdescent, 0.5 * height - ydescent
        # Create a rectangle as the box
        box = patches.Rectangle([center[0] - box_size / 2, center[1] - box_size / 2], 
                                box_size, box_size, edgecolor='black', facecolor='none', lw=1, transform=trans)
        # Adjust cross lines to fit within the new box size, with finer lines
        line1 = mlines.Line2D([center[0] - box_size / 3, center[0] + box_size / 3],
                              [center[1] - box_size / 3, center[1] + box_size / 3], color='black', lw=1, transform=trans)
        line2 = mlines.Line2D([center[0] - box_size / 3, center[0] + box_size / 3],
                              [center[1] + box_size / 3, center[1] - box_size / 3], color='black', lw=1, transform=trans)
        
        return [box, line1, line2]

def plot_invert_tri_corr_heatmap(correlation_matrix, p_value_matrix, fig_path, title_str=''):
    '''
    '''
    # Create a heat map using an inverted triangle matrix
    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=-1)
    plt.figure(figsize=(15, 10))
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)  # colour panel
    cmap = 'bwr'
    ax = sns.heatmap(correlation_matrix, mask=mask, annot=False, fmt=".2f", cmap=cmap, center=0,
                     square=True, linewidths=.5, cbar_kws={"shrink": .5, "pad": 0.2})
    if p_value_matrix is not None:
        plt.title(f'Correlation heatmap cross all measurements (with 95% CIs){title_str}')
    else:
        plt.title(f'Correlation heatmap cross all measurements {title_str}')
    
    # set position of x and y axis
    ax.set_xticks(np.arange(correlation_matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(correlation_matrix.shape[0]) + 0.5)
    ax.set_xticklabels(correlation_matrix.columns, rotation=90, ha='left')
    ax.set_yticklabels(correlation_matrix.index, rotation=0)
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    
    if p_value_matrix is not None:
        # shows cross legend
        legend_handle = patches.Rectangle((0, 0), 1, 1, color='white', edgecolor='none')  # Invisible, used for spacing
        plt.legend([legend_handle], ['p-value>0.05'], handler_map={patches.Rectangle: HandlerCrossInBox()},
                   loc='upper left', bbox_to_anchor=(1.3, 1))
    
        # mark the p-value > 0.05 as a "X"
        for i in range(p_value_matrix.shape[0]):
            for j in range(p_value_matrix.shape[1]):
                if p_value_matrix.iloc[i, j] > 0.05 and not mask[i, j]:
                    # cross from left-top to right-bottom
                    ax.plot([j, j+1], [i, i+1], color='black', lw=1)
                    # cross from right-top to left-bottom
                    ax.plot([j+1, j], [i, i+1], color='black', lw=1)
                    # frame
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=1))
                
    plt.tight_layout()
    plt.savefig(fig_path)
    # plt.show()
    
def _plot_pearson_corr_heatmap_p62_fibrosis(ENV_task, csv_file_names, do_p_val=True):
    '''
    '''
    
    csv_file_paths = []
    for csv_name in csv_file_names:
        csv_file_paths.append(os.path.join(ENV_task.META_FOLDER, csv_name))
    
    combined_df = merge_csv_files(csv_file_paths)
    updated_df = give_column_names(combined_df)
    correlation_matrix, p_value_matrix, N = calculate_pearson_correlation(updated_df)
    title_str = f', N = {N}'
    
    if do_p_val is False:
        p_value_matrix = None
        fig_path = os.path.join(ENV_task.STATISTIC_STORE_DIR,
                            f'p62_fibrosis_corr_pearson_map-no_p_val.png')
    else:
        fig_path = os.path.join(ENV_task.STATISTIC_STORE_DIR,
                            f'p62_fibrosis_corr_pearson_map.png')
    plot_invert_tri_corr_heatmap(correlation_matrix, p_value_matrix, fig_path, title_str)
    
def _plot_pearson_corr_heatmap_p62thd_fibrosis(ENV_task, csv_file_names, l_thd, do_p_val=True):
    '''
    '''
    
    csv_file_paths = []
    for csv_name in csv_file_names:
        csv_file_paths.append(os.path.join(ENV_task.META_FOLDER, csv_name))
    
    combined_df = merge_csv_files(csv_file_paths)
    updated_df = give_column_names(combined_df)
    filtered_df = updated_df[updated_df['Henning_dark_p62_frac'] >= l_thd]
    correlation_matrix, p_value_matrix, N = calculate_pearson_correlation(filtered_df)
    title_str = f', N = {N}, on Henning\'s fraction > {l_thd}%'
    
    if do_p_val is False:
        p_value_matrix = None
        fig_path = os.path.join(ENV_task.STATISTIC_STORE_DIR,
                            f'p62-l{l_thd}_fibrosis_corr_pearson_map-no_p_val.png')
    else:
        fig_path = os.path.join(ENV_task.STATISTIC_STORE_DIR,
                            f'p62-l{l_thd}_fibrosis_corr_pearson_map.png')
    plot_invert_tri_corr_heatmap(correlation_matrix, p_value_matrix, fig_path, title_str)
    
''' ------------ combine the org-heatmap into one picture ------------ '''

def load_data(csv_path):
    """ Load and clean the CSV file containing slide data. """
    data = pd.read_csv(csv_path)
    data['slide_id'] = data['slide_id'].apply(lambda x: x.strip())  # Clean slide IDs
    return data

def find_matching_images(directory):
    """ Find and yield pairs of matched images from given directory. """
    images_dict = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'org' in file:
                base_name = file.replace('org', '{type}')
            elif 'hard' in file:
                base_name = file.replace('hard', '{type}')
            else:
                continue

            if base_name not in images_dict:
                images_dict[base_name] = {}

            type_key = 'org' if 'org' in file else 'hard'
            images_dict[base_name][type_key] = os.path.join(root, file)

    for base_name, paths in images_dict.items():
        if 'org' in paths and 'hard' in paths:
            yield (paths['org'], paths['hard'], base_name.format(type='combine'))

def combine_images_vertically(image_path1, image_path2, text, output_path):
    """ Combine two images vertically and append a text area to the right. """
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    font_size = 512  # Font size is 512 pixels
    text_width = font_size * 8  # Assuming the maximum width of text could be the width of 8 characters at this font size

    # Calculate the total width and height needed for the combined image
    total_width = max(img1.width, img2.width) + text_width
    total_height = img1.height + img2.height

    new_img = Image.new('RGB', (total_width, total_height), "white")
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, img1.height))

    # Draw text with a very large font
    draw = ImageDraw.Draw(new_img)
    try:
        # Try to load a clearer, large font
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # If unable to load the font, fallback to the default font
        font = ImageFont.load_default()

    # Start drawing text to the right of the widest image
    text_x = max(img1.width, img2.width) + 50  # 50 pixels padding from the images
    text_y = 50  # Start 50 pixels from the top
    line_spacing = 600  # Adjust line spacing for large text

    for line in text.split('\n'):
        draw.text((text_x, text_y), line, fill="black", font=font)
        text_y += line_spacing  # Move to the next line position

    new_img.save(output_path)
    print(f'Saved combined image at: {output_path}')
    
def extract_slide_id_from_filename(filename):
    """ extract slide ID from filename"""
    parts = filename.split('_')
    if len(parts) > 1:
        slide_id_part = parts[1]  # like: Sl006-C5-P62-combine
        slide_id = slide_id_part.split('-')[0]  # get Sl006
        return slide_id
    return None

def process_images(input_directory, output_directory, data):
    """ Process each image pair and save the combined image. """
    for org_path, hard_path, combined_name in find_matching_images(input_directory):
        # extract slide_id
        slide_id = extract_slide_id_from_filename(combined_name)
        
        print(f"Processing: {combined_name}")  # Debug information
        print(f"Extracted slide_id: {slide_id}")

        if slide_id in data['slide_id'].values:
            slide_data = data[data['slide_id'] == slide_id].iloc[0] * 100
            text = 'Cluster Prop:\n' + '\n'.join([f"{col} = {slide_data[col]:.2f}%" for col in ['A', 'B', 'C', 'D', 'all']])
            output_path = os.path.join(output_directory, combined_name)
            combine_images_vertically(org_path, hard_path, text, output_path)
        else:
            print(f"Slide ID {slide_id} not found in CSV.")

def _combine_org_heatmap_proptext(ENV_task):
    # Set paths
    csv_path = os.path.join(ENV_task.META_FOLDER, 'slide_clusters_props-yang.csv')
    input_directory = os.path.join(ENV_task.HEATMAP_STORE_DIR, 'clst_assim_map-8x-asm02')
    output_directory = os.path.join(ENV_task.HEATMAP_STORE_DIR, 'clst_combine_map-8x-asm02')

    # Load data and process images
    slide_data = load_data(csv_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    process_images(input_directory, output_directory, slide_data)
    
    
''' --------- umap --------- '''
    
def visualize_slide_data(csv_file_paths, feature_columns, index_column, norm=True):
    '''
    Visualize slide data using UMAP projection and color coding based on a specific index column.

    :param csv_file_paths: List of paths to the CSV files
    :param feature_columns: List of columns to extract as features
    :param index_column: Column name to use for color coding in the plot
    :param norm: Boolean, whether to norm feature columns
    :return: None, shows a plot
    '''
    # Merge CSV files into a big single dataframe
    combined_df = merge_csv_files(csv_file_paths)
    
    # Select the relevant features and the index column
    data = combined_df[feature_columns + [index_column]]

    # Normalize the feature data if requested
    if norm:
        scaler = MinMaxScaler()
        data[feature_columns] = scaler.fit_transform(data[feature_columns])

    # Prepare UMAP reduction
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = umap_reducer.fit_transform(data[feature_columns])
    
    # Create a plot
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=data[index_column], 
                              palette='viridis', s=100, legend='full')
    scatter.set_title('UMAP Projection of Slide Data')
    scatter.legend(title=index_column, title_fontsize='13', labelspacing=1.05, fontsize='11')
    plt.show()

if __name__ == '__main__':
    pass



