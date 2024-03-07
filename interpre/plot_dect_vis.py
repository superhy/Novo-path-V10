'''
Created on 25 Nov 2022

@author: Yang Hu
'''
import os

from scipy.stats.kde import gaussian_kde
from scipy.stats.stats import pearsonr

from interpre.draw_maps import draw_original_image, draw_attention_heatmap
from interpre.prep_dect_vis import redu_K_embeds_distribution
from interpre.prep_tools import load_vis_pkg_from_pkl
from interpre.statistics import df_p62_s_clst_assim_tis_pct_ball_dist, \
    df_p62_s_clst_and_assim_t_p_ball_corr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from models.functions_clustering import load_clustering_pkg_from_pkl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from support.files import parse_slideid_from_filepath, parse_caseid_from_slideid
from support.metadata import query_task_label_dict_fromcsv, \
    load_percentages_from_csv
from wsi.slide_tools import slide_to_scaled_np_image


def draw_scaled_slide_imgs(ENV_task):
    '''
    draw all the scaled original slides images in the tissue folder
    '''
    slide_tissue_dir = ENV_task.SLIDE_FOLDER
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    slide_files = os.listdir(slide_tissue_dir)
    
    org_scaled_tissue_folder = 'scaled_org'
    if not os.path.exists(os.path.join(heat_store_dir, org_scaled_tissue_folder) ):
        os.makedirs(os.path.join(heat_store_dir, org_scaled_tissue_folder))
        print('create file dir {}'.format(os.path.join(heat_store_dir, org_scaled_tissue_folder)) )
    
    for slide_f in slide_files:
        np_slide_img, _, _, _, _ = slide_to_scaled_np_image(os.path.join(slide_tissue_dir, slide_f))
        slide_id = parse_slideid_from_filepath(slide_f)
        draw_original_image(os.path.join(heat_store_dir, org_scaled_tissue_folder), np_slide_img, 
                            (slide_id, 'scaled-{}'.format(str(ENV_task.SCALE_FACTOR)) ) )
        print('keep the scaled original tissue for: ', slide_id)
        

def _plot_draw_scaled_slide_imgs(ENV_task):
    draw_scaled_slide_imgs(ENV_task)
    
    
def _plot_activation_kde_dist(ENV_task, ENV_label_hv, act_scores_pkl_name, 
                              act_type=0, cut_top=None, legend_loc='best', conj_s_range=None):
    '''
    plot the activation scores' distribution, with different tag of group, like HV, 0, X
    
    Args:
        act_type: can only be 0 or 1, 0: tryk-mil fine-turned activation, 1: original
        cut_top: always cut the highest [cut_top] values from the act_scores of each slide
    '''
    heat_store_dir = ENV_task.HEATMAP_STORE_DIR
    label_dict = query_task_label_dict_fromcsv(ENV_label_hv)
    act_score_dict = load_vis_pkg_from_pkl(heat_store_dir, act_scores_pkl_name)[act_type]
    
    name_dict = {'P62': 'Ballooning'}
    name_activations_dict = {'P62': {'HV': [], '0-1': [], '2': []}
                             }
    
    activations = name_activations_dict[ENV_task.STAIN_TYPE]
    for slide_id, act_scores in act_score_dict.items():
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id not in label_dict.keys():
            slide_tag = '0-1'
        else:
            if label_dict[case_id] == 1: slide_tag = '2'
            else: slide_tag = 'HV'
        if cut_top is not None and cut_top > 0:
            activations[slide_tag].extend(np.partition(act_scores, -cut_top)[-cut_top:])
        else:
            activations[slide_tag].extend(act_scores)
    for tag in activations.keys():
        activations[tag] = np.asarray(activations[tag])
        
    sns.set(style="ticks")
    
    plt.figure(figsize=(5, 4))
    for tag, activation in activations.items():
        print(tag, np.sum(activation > 0.5), 
              np.min(activation), np.max(activation),
              np.average(activation), np.median(activation) )
        label_name = f'{name_dict[ENV_task.STAIN_TYPE]} {tag}' if tag != 'HV' else 'Healthy volunteers'
        sns.kdeplot(activation, fill=True, clip=(0, np.max(activation)),
                    label=label_name)
       
    if conj_s_range is not None: 
        # calculate the conjunction point of 'HV' and '2'
        kde_HV = gaussian_kde(np.array(activations['HV']))
        kde_2 = gaussian_kde(np.array(activations['2']))
        x = np.linspace(conj_s_range[0], conj_s_range[1], 1000) # set a searching points scope
        kde_vals_HV = kde_HV(x)
        kde_vals_2 = kde_2(x)
        # the conjunction is the lowest points with the difference of 'HV' and '2'
        index = np.argmin(np.abs(kde_vals_HV - kde_vals_2))
        cross_point = x[index]
        # draw
        plt.axvline(x=cross_point, color='gray', linestyle='--')
        plt.text(cross_point, 0, f'{cross_point:.2f}', color='black', ha='center')
    
    plt.xlim(0, 1.0)
    plt.title('Activation dist for diff-groups')
    plt.xlabel('Activation')
    plt.ylabel('Density')
    
    current_ticks = plt.xticks()[0]
    if 0.5 not in current_ticks:
        new_ticks = np.sort(np.append(current_ticks, 0.5))
    else:
        new_ticks = current_ticks
    plt.xticks(new_ticks)
    
    plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.savefig(os.path.join(heat_store_dir, 'groups_activation_distribution-kde-{}.png'.format('ft' if act_type == 0 else 'org') ) )
    print('store the picture in {}'.format(os.path.join(heat_store_dir, 
                                                        'groups_activation_distribution-kde-?.png')) )
    plt.clf()
    
def plot_K_embeds_scatter(ENV_task, l_m_h_tuples, group_names, legend_loc='best', act_type=0):
    '''
    
    Args:
        l_m_h_tuples: low_features, mid_features, high_features
        group_names: mapping to l_m_h_tuples, 
            which are Healthy volunteers, Ballooning 0-1, Ballooning 2
        legend_loc:
        act_type: just for mapping the name
    '''
    stat_store_dir = ENV_task.STATISTIC_STORE_DIR
    
    if len(group_names) != 3:
        raise ValueError("tuples or groups\' number is not right")
    
    low_features, mid_features, high_features = l_m_h_tuples
    
    # make DataFrame
    df_low = pd.DataFrame(low_features, columns=['x', 'y'])
    df_low['group'] = group_names[0]

    df_mid = pd.DataFrame(mid_features, columns=['x', 'y'])
    df_mid['group'] = group_names[1]

    df_high = pd.DataFrame(high_features, columns=['x', 'y'])
    df_high['group'] = group_names[2]
    
    # merge the DataFrame
    df = pd.concat([df_low, df_mid, df_high])
    
    # plot the scatter 
    sns.scatterplot(data=df, x='x', y='y', hue='group', s=2.5)
    plt.title("Representative tiles feature scatter")
    plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.savefig(os.path.join(stat_store_dir, 'K_l-m-h_scatter-{}.png'.format('org' if act_type == 0 else 'ft') ) )
    print('store the picture in {}'.format(os.path.join(stat_store_dir, 
                                                        'K_l-m-h_scatter-?.png')) )
    plt.clf()
    
def _plot_groups_K_embeds_scatter(ENV_task, ENV_label_hv, K_t_embeds_pkl_name,
                                  group_names = ['Healthy volunteers', 'Ballooning 0-1', 'Ballooning 2'], 
                                  legend_loc='best'):
    '''
    '''
    if K_t_embeds_pkl_name.startswith('K_t_org'):
        act_type = 0
    else:
        act_type = 1
    
    slide_K_t_embeds_dict = load_vis_pkg_from_pkl(ENV_task.STATISTIC_STORE_DIR,
                                                  K_t_embeds_pkl_name)
    low_features, mid_features, high_features = redu_K_embeds_distribution(ENV_label_hv, 
                                                                           slide_K_t_embeds_dict)
    plot_K_embeds_scatter(ENV_task, (low_features, mid_features, high_features), 
                          group_names, legend_loc, act_type)
       
    
def _plot_scores_heatmaps(ENV_task, ENV_annotation, heatmap_pkl_name,
                          _env_data_tumor_mask_dir=None):
    """
    plot the original heatmap with attention scores for all tiles
    """
    
    ''' prepare some parames  '''
    _env_heatmap_store_dir = ENV_task.HEATMAP_STORE_DIR
    _env_task_name = ENV_task.TASK_NAME
    
    slide_heatmap_dict = load_vis_pkg_from_pkl(_env_heatmap_store_dir,
                                               heatmap_pkl_name)
    label_dict = query_task_label_dict_fromcsv(ENV_annotation)
    
    for slide_id in slide_heatmap_dict.keys():
        case_id = parse_caseid_from_slideid(slide_id)
        slide_label = label_dict[case_id]
        
        heatmap_info_dict = slide_heatmap_dict[slide_id]
        org_image = heatmap_info_dict['original']
        heat_med_style = heatmap_info_dict['heat_med_style']
        heat_med_grad = heatmap_info_dict['heat_med_grad']
        
        alg_name = heatmap_pkl_name[heatmap_pkl_name.find('heatmap') + 8:-14]
        
        single_multi_dir = os.path.join(_env_heatmap_store_dir, 'att_map')
        attention_dir = os.path.join(single_multi_dir, 'attention_dx')
        cam_dir = os.path.join(single_multi_dir, 'cam')
        
        if not os.path.exists(os.path.join(_env_heatmap_store_dir, '{}//1'.format(attention_dir))):
            os.makedirs(os.path.join(_env_heatmap_store_dir, '{}//1'.format(attention_dir)))
            print('create file dir {}'.format(os.path.join(_env_heatmap_store_dir, '{}//1'.format(attention_dir))))
        if not os.path.exists(os.path.join(_env_heatmap_store_dir, '{}//0'.format(attention_dir))):
            os.makedirs(os.path.join(_env_heatmap_store_dir, '{}//0'.format(attention_dir)))
            print('create file dir {}'.format(os.path.join(_env_heatmap_store_dir, '{}//0'.format(attention_dir))))
        if not os.path.exists(os.path.join(_env_heatmap_store_dir, '{}//1'.format(cam_dir))):
            os.makedirs(os.path.join(_env_heatmap_store_dir, '{}//1'.format(cam_dir)))
            print('create file dir {}'.format(os.path.join(_env_heatmap_store_dir, '{}//1'.format(cam_dir))))
        if not os.path.exists(os.path.join(_env_heatmap_store_dir, '{}//0'.format(cam_dir))):
            os.makedirs(os.path.join(_env_heatmap_store_dir, '{}//0'.format(cam_dir)))
            print('create file dir {}'.format(os.path.join(_env_heatmap_store_dir, '{}//0'.format(cam_dir))))
            
        slide_troi_contours = None
        draw_attention_heatmap(org_image, 
                               _env_heatmap_store_dir,
                               None, slide_troi_contours, 
                               (os.path.join('{}//'.format(attention_dir) + str(slide_label), slide_id + '-org'), alg_name))
        print('draw original image in: {} for slide:{}'.format(os.path.join(_env_heatmap_store_dir, '{}//'.format(attention_dir)), slide_id))
#         draw_attention_heatmap(heat_cam_style, 
#                                _env_heatmap_store_dir,
#                                org_image, slide_troi_contours,
#                                (os.path.join('{}//'.format(attention_dir) + str(slide_label), slide_id + '-camst'), alg_name))
#         print('draw cam_style heatmap in: {} for slide:{}'.format(os.path.join(_env_heatmap_store_dir, '{}//'.format(attention_dir)), slide_id))
        draw_attention_heatmap(heat_med_style, 
                               _env_heatmap_store_dir,
                               org_image, None,
                               (os.path.join('{}//'.format(attention_dir) + str(slide_label), slide_id + '-medst'), alg_name))
        print('draw med_style heatmap in: {} for slide:{}'.format(os.path.join(_env_heatmap_store_dir, '{}//'.format(attention_dir)), slide_id))
        
        if heat_med_grad is not None:
            draw_attention_heatmap(heat_med_grad, 
                                   _env_heatmap_store_dir,
                                   org_image, None,
                                   (os.path.join('{}//'.format(attention_dir) + str(slide_label), slide_id + '-gcol'), alg_name))
            print('+ draw med gradient color heatmap in: {} for slide:{}'.format(os.path.join(_env_heatmap_store_dir, '{}//'.format(attention_dir)), slide_id))
            

def _plot_topK_scores_heatmaps(ENV_task, ENV_annotation, heatmap_pkl_name, folder_sfx=''):
    """
    plot the original heatmap with attention scores for top K tiles
    """
    
    ''' prepare some parames  '''
    _env_heatmap_store_dir = ENV_task.HEATMAP_STORE_DIR
    _env_task_name = ENV_task.TASK_NAME
    
    slide_topk_heatmap_dict = load_vis_pkg_from_pkl(_env_heatmap_store_dir,
                                                    heatmap_pkl_name)
    label_dict = query_task_label_dict_fromcsv(ENV_annotation)
    
    for slide_id in slide_topk_heatmap_dict.keys():
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id in label_dict.keys():
            slide_label = label_dict[case_id]
        else:
            slide_label = 'm'
        
        heatmap_info_dict = slide_topk_heatmap_dict[slide_id]
        org_image = heatmap_info_dict['original']
        heat_hard_cv2 = heatmap_info_dict['heat_hard_cv2']
        heat_soft_cv2 = heatmap_info_dict['heat_soft_cv2']
        
        alg_name = heatmap_pkl_name[heatmap_pkl_name.find('K_map_') + 7:-15]
        
        folder_prefix = heatmap_pkl_name.split('_')[0]
        single_multi_dir = os.path.join(_env_heatmap_store_dir, f'{folder_prefix}_map_{folder_sfx}')
        attention_dir = os.path.join(single_multi_dir, f'{folder_prefix}_dx')
        
        if not os.path.exists(os.path.join(_env_heatmap_store_dir, '{}//1'.format(attention_dir))):
            os.makedirs(os.path.join(_env_heatmap_store_dir, '{}//1'.format(attention_dir)))
            print('create file dir {}'.format(os.path.join(_env_heatmap_store_dir, '{}//1'.format(attention_dir))))
        if not os.path.exists(os.path.join(_env_heatmap_store_dir, '{}//0'.format(attention_dir))):
            os.makedirs(os.path.join(_env_heatmap_store_dir, '{}//0'.format(attention_dir)))
            print('create file dir {}'.format(os.path.join(_env_heatmap_store_dir, '{}//0'.format(attention_dir))))
        if not os.path.exists(os.path.join(_env_heatmap_store_dir, '{}//m'.format(attention_dir))):
            os.makedirs(os.path.join(_env_heatmap_store_dir, '{}//m'.format(attention_dir)))
            print('create file dir {}'.format(os.path.join(_env_heatmap_store_dir, '{}//m'.format(attention_dir))))
            
        # slide_troi_contours = None
        draw_attention_heatmap(attention_dir, org_image, None, None, 
                               (os.path.join('{}//'.format(attention_dir) + str(slide_label), slide_id + '-org'), alg_name))
        print('draw original image in: {} for slide:{}'.format(os.path.join(_env_heatmap_store_dir, '{}//'.format(attention_dir)), slide_id))

        if heat_hard_cv2 is not None:
            draw_attention_heatmap(attention_dir, heat_hard_cv2, org_image, None,
                                   (os.path.join('{}//'.format(attention_dir) + str(slide_label), slide_id + '-hard'), alg_name))
            print('1. draw hard heatmap in: {} for slide:{}'.format(os.path.join(_env_heatmap_store_dir, '{}//'.format(attention_dir)), slide_id))
        
        draw_attention_heatmap(attention_dir, heat_soft_cv2, org_image, None,
                               (os.path.join('{}//'.format(attention_dir) + str(slide_label), slide_id + '-soft'), alg_name))
        print('2. draw soft heatmap in: {} for slide:{}'.format(os.path.join(_env_heatmap_store_dir, '{}//'.format(attention_dir)), slide_id))
    
        
def _plot_spatial_sensi_clusters_assims(ENV_task, ENV_annotation, spatmap_pkl_name, draw_org=False):
    '''
    plot the original image and the spatial map of the sensitive clusters as well as the assimilated tiles (optional)
    '''
    _env_heatmap_store_dir = ENV_task.HEATMAP_STORE_DIR
    _env_task_name = ENV_task.TASK_NAME
    
    slide_topk_heatmap_dict = load_vis_pkg_from_pkl(_env_heatmap_store_dir,
                                                    spatmap_pkl_name)
    label_dict = query_task_label_dict_fromcsv(ENV_annotation)
    # print(len(slide_topk_heatmap_dict.keys()))
    # print(slide_topk_heatmap_dict.keys())
    
    for slide_id in slide_topk_heatmap_dict.keys():
        case_id = parse_caseid_from_slideid(slide_id)
        if case_id in label_dict.keys():
            slide_label = label_dict[case_id]
        else:
            slide_label = 'm'
            
        org_img, heat_s_clst_col = slide_topk_heatmap_dict[slide_id]
            
        alg_name = spatmap_pkl_name[spatmap_pkl_name.find('-spat') + 6:-15]
        if spatmap_pkl_name.find('-a-spat') != -1:
            single_multi_dir = os.path.join(_env_heatmap_store_dir, 'clst_assim_map')
            spat_dir = os.path.join(single_multi_dir, 'clst_assim_dx')
        else:
            single_multi_dir = os.path.join(_env_heatmap_store_dir, 'clst_map')
            spat_dir = os.path.join(single_multi_dir, 'clst_dx')
            
        if not os.path.exists(os.path.join(_env_heatmap_store_dir, '{}//1'.format(spat_dir))):
            os.makedirs(os.path.join(_env_heatmap_store_dir, '{}//1'.format(spat_dir)))
            print('create file dir {}'.format(os.path.join(_env_heatmap_store_dir, '{}//1'.format(spat_dir))))
        if not os.path.exists(os.path.join(_env_heatmap_store_dir, '{}//0'.format(spat_dir))):
            os.makedirs(os.path.join(_env_heatmap_store_dir, '{}//0'.format(spat_dir)))
            print('create file dir {}'.format(os.path.join(_env_heatmap_store_dir, '{}//0'.format(spat_dir))))
        if not os.path.exists(os.path.join(_env_heatmap_store_dir, '{}//m'.format(spat_dir))):
            os.makedirs(os.path.join(_env_heatmap_store_dir, '{}//m'.format(spat_dir)))
            print('create file dir {}'.format(os.path.join(_env_heatmap_store_dir, '{}//m'.format(spat_dir))))
        
        if draw_org is True:
            # if draw the original image
            draw_attention_heatmap(spat_dir, org_img, None, None, 
                                   (os.path.join('{}//'.format(spat_dir) + str(slide_label), slide_id + '-org'), alg_name))
            print('draw original image in: {} for slide:{}'.format(os.path.join(_env_heatmap_store_dir, '{}//'.format(spat_dir)), 
                                                               slide_id))
            
        draw_attention_heatmap(spat_dir, heat_s_clst_col, org_img, None,
                               (os.path.join('{}//'.format(spat_dir) + str(slide_label), slide_id + '-hard'), alg_name))
        print('draw clst(assim) heatmap in: {} for slide:{}'.format(os.path.join(_env_heatmap_store_dir,'{}//'.format(spat_dir)),
                                                                    slide_id))
        
def df_plot_s_clst_assim_ball_dist_box(ENV_task, s_clst_t_p_pkl_name, 
                                       assim_t_p_pkl_name, biom_label_fname):
    '''
    plot the tissue percentage of sensitive clusters and assimilated patches
    for ballooning low (score = 0) or ballooning high (score = 2)
    '''
    palette_dict = {'ballooning-0': 'lightcoral',
                    'ballooning-2': 'turquoise'
                }
    
    ''' loading/processing data '''
    # biom_label_dict = query_task_label_dict_fromcsv(ENV_task, biom_label_fname)
    s_clst_slide_t_p_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, s_clst_t_p_pkl_name)
    assim_slide_t_p_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, assim_t_p_pkl_name)
    df_tis_pct_dist_elemts = df_p62_s_clst_assim_tis_pct_ball_dist(ENV_task, s_clst_slide_t_p_dict,
                                                                   assim_slide_t_p_dict, biom_label_fname)
    
    order = ['sensitive clusters', 'assimilated patches', 'both']
    df_tis_pct_dist_elemts['patch_type'] = pd.Categorical(df_tis_pct_dist_elemts['patch_type'],
                                                          categories=order, ordered=True)
    
    fig = plt.figure(figsize=(5, 6))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1 = sns.boxplot(x='patch_type', y='tissue_percentage', palette=palette_dict,
                       data=df_tis_pct_dist_elemts, hue='ballooning_label')
    ax_1.set_title('tis-pct of sensitive/assimilated patches \n in ballooning low/high slides')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR,
                             'sensi_clst_assim_tp-ballooning-p-box.png') )
    print('store the picture in {}'.format(os.path.join(ENV_task.HEATMAP_STORE_DIR, 'sensi_clst_assim_tp-ballooning-p-box.png') ) )
    plt.close(fig)
    
def df_plot_s_clst_assim_ball_corr_box(ENV_task, s_clst_t_p_pkl_name, assim_t_p_pkl_name):
    '''
    plot the tissue percentage of sensitive clusters and assimilated patches
    for different ballooning scores (include health volunteer)
    '''
    palette_dict = {'sensitive clusters': 'lightcoral',
                    'assimilated patches': 'turquoise',
                    'both': 'dodgerblue'
                }
    
    ''' loading/processing data '''
    # biom_label_dict = query_task_label_dict_fromcsv(ENV_task, biom_label_fname)
    s_clst_slide_t_p_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, s_clst_t_p_pkl_name)
    assim_slide_t_p_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, assim_t_p_pkl_name)
    df_tis_pct_corr_elemts = df_p62_s_clst_and_assim_t_p_ball_corr(ENV_task, 
                                                                   s_clst_slide_t_p_dict,
                                                                   assim_slide_t_p_dict)
    
    order = ['Health volunteers', 'Ballooning-0', 'Ballooning-1', 'Ballooning-2']
    df_tis_pct_corr_elemts['ballooning_label'] = pd.Categorical(df_tis_pct_corr_elemts['ballooning_label'],
                                                                categories=order, ordered=True)
    
    fig = plt.figure(figsize=(10, 5))
    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1 = sns.boxplot(x='ballooning_label', y='tissue_percentage', palette=palette_dict,
                       data=df_tis_pct_corr_elemts, hue='patch_type')
    ax_1.set_title('tis-pct of sensitive/assimilated patches \n in slides with different ballooning scores')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR,
                             'sensi_clst_assim_tp-ballooning-s-box.png') )
    print('store the picture in {}'.format(os.path.join(ENV_task.HEATMAP_STORE_DIR, 'sensi_clst_assim_tp-ballooning-s-box.png') ) )
    plt.close(fig)
    
    
''' --------- clustering distribution visualise for hierarchical clustering results --------- '''
    
def plot_clsts_tis_pct_abs_nb_box(ENV_task, ENV_annotation, tis_pct_pkl_name, branch_prefix, 
                                  avail_labels=['Healthy volunteers', 'Ballooning 0-1', 'Ballooning 2'], 
                                  tis_pct=True):
    '''
    
    Args:
        ENV_task:
        ENV_annotation: could be env_flinc_p62.ENV_FLINC_P62_BALL_HV
        tis_pct_pkl_name: the distribution of tissue percentage or absolute number, pkl file name
        branch_prefix: indicate the clusters' family
        avail_labels: label names shown on figure
        tis_pct: if Ture, showing tissue percentage; if False, showing absolute number
    '''
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    label_dict = query_task_label_dict_fromcsv(ENV_annotation)
    
    if len(avail_labels) < 3:
        raise ValueError("avail_labels must has at least 3 values!")
    
    data = []
    for slide_id, (tissue_pct_dict, abs_num_dict) in slide_tis_pct_dict.items():
        # get case_id and load the label
        case_id = parse_caseid_from_slideid(slide_id)
        # label = label_dict.get(case_id, 'mid')  # if label not exist, set as 'mid'
        # label = avail_labels[0] if label == 0 else avail_labels[1] if label == 1 else avail_labels[2]
        if case_id not in label_dict.keys():
            label_name = avail_labels[1]
        elif label_dict[case_id] == 0:
            label_name = avail_labels[0]
        else:
            label_name = avail_labels[2]

        # via branch_prefix, select the target clusters
        for cluster_name, value in (tissue_pct_dict if tis_pct else abs_num_dict).items():
            if cluster_name.startswith(branch_prefix):
                data.append({'cluster_name': cluster_name, 'value': value, 'label': label_name})

    # create the dataframe
    df_clst_label = pd.DataFrame(data)
    df_clst_label['label'] = pd.Categorical(df_clst_label['label'], categories=avail_labels, ordered=True)
    print(df_clst_label)
    # setup the width on x-axis
    unique_clusters = df_clst_label['cluster_name'].nunique()
    chart_width = max(5, unique_clusters * 1)
    
    colors = ['dodgerblue', 'turquoise', 'lightcoral']
    palette = {label: color for label, color in zip(avail_labels, colors)}
    
    dist_target = ENV_annotation.TASK_NAME
    # boxplot
    plt.figure(figsize=(chart_width, 6))
    sns.boxplot(x='cluster_name', y='value', hue='label', data=df_clst_label, palette=palette)
    plt.title(f"Cluster distribution to {dist_target} for cluster family '{branch_prefix}'")
    if tis_pct:
        plt.ylim(bottom=-0.0005, top=0.025)
    else:
        plt.ylim(bottom=-5, top=250)
    plt.xlabel('Cluster Name')
    plt.ylabel('Tissue Percentage' if tis_pct else 'Number of tiles')
    plt.legend(title='Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    fig_name = f'tissue_percenatge_clsts-{dist_target}_family-{branch_prefix}.png' if tis_pct \
        else f'absolute_number_clsts-{dist_target}_family-{branch_prefix}.png'
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR, fig_name ) )
    print('store the picture in {}'.format(os.path.join(ENV_task.HEATMAP_STORE_DIR, 
                                                        f'tissue_percenatge(absolute_number)_clsts_family-{branch_prefix}')) )
    plt.clf()
    
def plot_clst_gp_tis_pct_abs_nb_box(ENV_task, ENV_annotation, tis_pct_pkl_name, gp_prefixs, 
                                    avail_labels=['Healthy volunteers', 'Ballooning 0-1', 'Ballooning 2'], 
                                    tis_pct=True):
    '''
    
    Args:
        ENV_task:
        ENV_annotation: could be env_flinc_p62.ENV_FLINC_P62_BALL_HV
        tis_pct_pkl_name: the distribution of tissue percentage or absolute number, pkl file name
        gp_prefixs: indicate the cluster groups' families
        avail_labels: label names shown on figure
        tis_pct: if Ture, showing tissue percentage; if False, showing absolute number
    '''
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    label_dict = query_task_label_dict_fromcsv(ENV_annotation)
    
    if len(avail_labels) < 3:
        raise ValueError("avail_labels must has at least 3 values!")
    
    data = []
    for gp_prefix in gp_prefixs:
        for slide_id, (tissue_pct_dict, abs_num_dict) in slide_tis_pct_dict.items():
            # get case_id and load the label
            case_id = parse_caseid_from_slideid(slide_id)
            # label = label_dict.get(case_id, 'mid')  # if label not exist, set as 'mid'
            # label = avail_labels[0] if label == 0 else avail_labels[1] if label == 1 else avail_labels[2]
            if case_id not in label_dict.keys():
                label_name = avail_labels[1]
            elif label_dict[case_id] == 0:
                label_name = avail_labels[0]
            else:
                label_name = avail_labels[2]

            # calculate the sum of cluster distribution in each group
            total_value = sum(value for cluster_name, value in (tissue_pct_dict if tis_pct else abs_num_dict).items() \
                                if cluster_name.startswith(gp_prefix))
            data.append({'group_prefix': gp_prefix, 'value': total_value, 'label': label_name})

    # create the dataframe
    df_clst_label = pd.DataFrame(data)
    df_clst_label['label'] = pd.Categorical(df_clst_label['label'], categories=avail_labels, ordered=True)
    print(df_clst_label)
    # setup the width on x-axis
    chart_width = max(5, len(gp_prefixs) * 1.5)
    
    colors = ['dodgerblue', 'turquoise', 'lightcoral']
    palette = {label: color for label, color in zip(avail_labels, colors)}
    
    dist_target = ENV_annotation.TASK_NAME
    # boxplot
    plt.figure(figsize=(chart_width, 6))
    sns.boxplot(x='group_prefix', y='value', hue='label', data=df_clst_label, palette=palette)
    plt.title(f"Cluster group distribution for groups '{gp_prefixs[0]} ~ {gp_prefixs[-1]}'")
    if tis_pct:
        # plt.ylim(bottom=-0.005, top=0.25 * (8 / len(gp_prefixs) ))
        plt.ylim(bottom=-0.005, top=0.25 )
    else:
        # plt.ylim(bottom=-50, top=2500 * (8 / len(gp_prefixs) ))
        plt.ylim(bottom=-50, top=2500 )
    plt.xlabel('Cluster Name')
    plt.ylabel('Tissue Percentage' if tis_pct else 'Number of tiles')
    plt.legend(title='Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    gp_name = f'{gp_prefixs[0]}~{gp_prefixs[-1]}_groups'
    fig_name = f'tissue_percenatge_c-groups-{dist_target}_{gp_name}.png' if tis_pct \
        else f'absolute_number_c-groups-{dist_target}_{gp_name}.png'
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR, fig_name ) )
    print('store the picture in {}'.format(os.path.join(ENV_task.HEATMAP_STORE_DIR, 
                                                        f'tissue_percenatge(absolute_number)_c-groups-{gp_name}')) )
    plt.clf()
    
    
# def visualize_avg_cluster_difference_barplot(ENV_task, ENV_annotation, label_dict_neg, 
#                                              tis_pct_pkl_name, gp_prefixs, 
#                                              avail_labels = ['Healthy volunteers', 'Ballooning 0-1', 'Ballooning 2'], 
#                                              tis_pct=True):
#     '''
# TODO:
#     '''
#     slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
#     label_dict = query_task_label_dict_fromcsv(ENV_annotation)
#
#     data = []
#     for gp_prefix in gp_prefixs:
#         total_values_pos = {label: 0 for label in avail_labels}
#         total_values_neg = {label: 0 for label in avail_labels}
#         count_values_pos = {label: 0 for label in avail_labels}
#         count_values_neg = {label: 0 for label in avail_labels}
#
#         for slide_id, (tissue_pct_dict, abs_num_dict) in slide_tis_pct_dict.items():
#             # 获取对应的case_id和label
#             case_id = parse_caseid_from_slideid(slide_id)
#             label = label_dict.get(case_id, 'mid')
#             label_neg = label_dict_neg.get(case_id, 'mid')
#
#             # 转换为对应的标签名
#             label_name = avail_labels[0] if label == 0 else avail_labels[1] if label == 1 else 'mid'
#             label_neg_name = avail_labels[0] if label_neg == 0 else avail_labels[1] if label_neg == 1 else 'mid'
#
#             # 累计每个gp_prefix下的clusters总和和计数
#             for cluster_name, value in (tissue_pct_dict if tis_pct else abs_num_dict).items():
#                 if cluster_name.startswith(gp_prefix):
#                     total_values_pos[label_name] += value
#                     count_values_pos[label_name] += 1
#                     total_values_neg[label_neg_name] += value
#                     count_values_neg[label_neg_name] += 1
#
#         # 计算平均值差异并添加到数据集
#         for label in avail_labels:
#             if count_values_pos[label] > 0 and count_values_neg[label] > 0:
#                 avg_value_pos = total_values_pos[label] / count_values_pos[label]
#                 avg_value_neg = total_values_neg[label] / count_values_neg[label]
#                 difference = avg_value_pos - avg_value_neg
#                 data.append({'group_prefix': gp_prefix, 'value': difference, 'label': label})
#
#     # create dataframe
#     df = pd.DataFrame(data)
#
#     # make color
#     colors = ['dodgerblue', 'turquoise', 'lightcoral']
#     palette = {label: color for label, color in zip(avail_labels, colors)}
#
#     # setup the width on x-axis
#     chart_width = max(5, len(gp_prefixs) * 1.5)
#
#     # plot
#     plt.figure(figsize=(chart_width, 6))
#     sns.barplot(x='value', y='group_prefix', hue='label', data=df, palette=palette, orient='h')
#     plt.title("Average Cluster Difference Distribution")
#     plt.xlabel('Average Difference in Tissue Percentage' if tis_pct else 'Average Difference in Absolute Number')
#     plt.ylabel('Group Prefix')
#     plt.legend(title='Label')
#
#     plt.show()

def plot_clst_gp_tis_pct_abs_nb_ball_df_stea(ENV_task, ENV_ball_ant, 
                                             stea_csv_filename, tis_pct_pkl_name,
                                             gp_prefixs, hl_prefixs=None, tis_pct=True, def_color='skyblue'):
    '''
    '''
    
    def custom_lineplot(x, y, **kwargs):
        data = kwargs.pop('data')
        kwargs.pop('color', None)
        if hl_prefixs is None:
            color = def_color
        else:
            hl_clst_gp_names = [f'clsts_group {hl_pf}' for hl_pf in hl_prefixs]
            color = 'lightsalmon' if data['Cluster Group'].iloc[0] in hl_clst_gp_names else def_color
        sns.lineplot(x=x, y=y, data=data, color=color, **kwargs)
    
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    ballooning_label_dict = query_task_label_dict_fromcsv(ENV_ball_ant) 
    steatosis_label_full_dict = query_task_label_dict_fromcsv(ENV_task, stea_csv_filename) 
    
    # # initialize the data structure
    # cluster_group_totals  = {prefix: {'b-high & s-3': [], 'b-high & s-2': [], 
    #                                'b-high & s-1': [], 'b-high & s-0': []} for prefix in gp_prefixs}

    data = []
    for slide_id, (tissue_pct_dict, abs_num_dict) in slide_tis_pct_dict.items():
        case_id = parse_caseid_from_slideid(slide_id)
        clst_value_dict = tissue_pct_dict if tis_pct else abs_num_dict

        if ballooning_label_dict.get(case_id) == 1:  # Ballooning high
            stea_score = steatosis_label_full_dict.get(case_id)
            if stea_score is not None:
                for gp_prefix in gp_prefixs:
                    total_value = sum(value for cluster_name, value in clst_value_dict.items() if cluster_name.startswith(gp_prefix))
                    data.append({
                        'Cluster Group': f'clsts_group {gp_prefix}', 
                        'Steatosis': f'b-high & s-{stea_score}', 
                        'Value': total_value,
                        'Slide ID': slide_id
                    })
    df = pd.DataFrame(data)
    steatosis_order = ['b-high & s-3', 'b-high & s-2', 'b-high & s-1', 'b-high & s-0']
    df['Steatosis'] = pd.Categorical(df['Steatosis'], categories=steatosis_order, ordered=True)
    print(df)

    # use FacetGrid to generate sub-plots
    g = sns.FacetGrid(df, col='Cluster Group', col_wrap=len(gp_prefixs), 
                      height=4, aspect=0.5, sharey=True)
    # g.map(sns.lineplot, 'Steatosis', 'Value', marker='o')
    g.map_dataframe(custom_lineplot, 'Steatosis', 'Value', marker='o')

    # setup title and label
    g.set_titles(col_template="{col_name}")
    g.set_ylabels("Tissue Percentage" if tis_pct else "Number of tiles")
    for ax in g.axes:
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # plt.suptitle('Tissue Percentage by Ballooning and Steatosis Score', fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    
    gp_name = f'{gp_prefixs[0]}~{gp_prefixs[-1]}_groups'
    fig_name = f'tis-pct_df-ball-stea_{gp_name}.png' if tis_pct \
        else f'abs-nb_df-ball-stea_{gp_name}.png'
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR, fig_name) )
    print('store the picture in {}'.format(os.path.join(ENV_task.HEATMAP_STORE_DIR, fig_name)) )
    plt.clf()
    
def plot_clst_gp_tis_pct_abs_nb_ball_df_lob(ENV_task, ENV_ball_ant,
                                            lob_csv_filename, tis_pct_pkl_name,
                                            gp_prefixs, hl_prefixs=None, tis_pct=True, def_color='skyblue'):
    '''
    '''
    
    def custom_lineplot(x, y, **kwargs):
        data = kwargs.pop('data')
        kwargs.pop('color', None)
        if hl_prefixs is None:
            color = def_color
        else:
            hl_clst_gp_names = [f'clsts_group {hl_pf}' for hl_pf in hl_prefixs]
            color = 'lightsalmon' if data['Cluster Group'].iloc[0] in hl_clst_gp_names else def_color
        sns.lineplot(x=x, y=y, data=data, color=color, **kwargs)
    
    slide_tis_pct_dict = load_vis_pkg_from_pkl(ENV_task.HEATMAP_STORE_DIR, tis_pct_pkl_name)
    ballooning_label_dict = query_task_label_dict_fromcsv(ENV_ball_ant) 
    lob_inf_label_full_dict = query_task_label_dict_fromcsv(ENV_task, lob_csv_filename) 
    
    # # initialize the data structure
    # cluster_group_totals  = {prefix: {'b-high & s-3': [], 'b-high & s-2': [], 
    #                                'b-high & s-1': [], 'b-high & s-0': []} for prefix in gp_prefixs}

    data = []
    for slide_id, (tissue_pct_dict, abs_num_dict) in slide_tis_pct_dict.items():
        case_id = parse_caseid_from_slideid(slide_id)
        clst_value_dict = tissue_pct_dict if tis_pct else abs_num_dict

        if ballooning_label_dict.get(case_id) == 1:  # Ballooning high
            lob_score = lob_inf_label_full_dict.get(case_id)
            if lob_score is not None:
                for gp_prefix in gp_prefixs:
                    total_value = sum(value for cluster_name, value in clst_value_dict.items() if cluster_name.startswith(gp_prefix))
                    data.append({
                        'Cluster Group': f'clsts_group {gp_prefix}', 
                        'Lob-inf': f'b-high & i-{lob_score}', 
                        'Value': total_value,
                        'Slide ID': slide_id
                    })
    df = pd.DataFrame(data)
    steatosis_order = ['b-high & i-3', 'b-high & i-2', 'b-high & i-1', 'b-high & i-0']
    df['Lob-inf'] = pd.Categorical(df['Lob-inf'], categories=steatosis_order, ordered=True)
    print(df)

    # use FacetGrid to generate sub-plots
    g = sns.FacetGrid(df, col='Cluster Group', col_wrap=len(gp_prefixs), 
                      height=4, aspect=0.5, sharey=True)
    # g.map(sns.lineplot, 'Lob-inf', 'Value', marker='o')
    g.map_dataframe(custom_lineplot, 'Lob-inf', 'Value', marker='o')

    # setup title and label
    g.set_titles(col_template="{col_name}")
    g.set_ylabels("Tissue Percentage" if tis_pct else "Number of tiles")
    for ax in g.axes:
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # plt.suptitle('Tissue Percentage by Ballooning and Steatosis Score', fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    
    gp_name = f'{gp_prefixs[0]}~{gp_prefixs[-1]}_groups'
    fig_name = f'tis-pct_df-ball-lob_{gp_name}.png' if tis_pct \
        else f'abs-nb_df-ball-lob_{gp_name}.png'
    plt.savefig(os.path.join(ENV_task.HEATMAP_STORE_DIR, fig_name ) )
    print('store the picture in {}'.format(os.path.join(ENV_task.HEATMAP_STORE_DIR, fig_name)) )
    plt.clf()
    
def plot_clst_gp_tis_pct_abs_nb_ball_df_fib(ENV_task, ENV_ball_ant,
                                            fib_csv_filename, tis_pct_pkl_name,
                                            gp_prefixs, hl_prefixs=None, tis_pct=True, def_color='skyblue'):
    '''
    TODO:
    '''
    pass

''' -------------------------------------------------------- '''

def plot_sc_at_tp_nb_ball_grades_scatter(ENV_task, _someargs_, only_sc=True):
    '''
    PS:
        sc_at -> sensi_c_assim_t
        tp_nb -> tissue_percentage and abs_number
        
    TODO: finish this function, args are not filled all
    '''
    
def plot_sc_at_tis_pct_ball_grades_box(ENV_task, _someargs_):
    '''
    PS:
        sc_at -> sensi_c_assim_t
        
    TODO: finish this function, args are not filled all
    '''
    
def plot_sc_at_abs_nb_ball_grades_box(ENV_task, _someargs_):
    '''
    PS:
        sc_at -> sensi_c_assim_t
        
    TODO: finish this function, args are not filled all
    '''
    
def plot_he_rpt_henning_label_parcats(ENV_task,
                                      score_csv_name, percentage_label_csv_name):
    '''
    '''
    score_csv_path = os.path.join(ENV_task.META_FOLDER, score_csv_name)
    percentage_label_csv_path = os.path.join(ENV_task.META_FOLDER, percentage_label_csv_name)
    
    # Read CSV files
    df_score = pd.read_csv(score_csv_path)
    df_percentage_label = pd.read_csv(percentage_label_csv_path)
    
    # Merge the two DataFrames on 'slide_id'
    merged_df = pd.merge(df_score, df_percentage_label, on='slide_id', suffixes=('_score', '_label'))
    print(merged_df)
    
    # Create a Parallel Categories Diagram (Parcats) to show the label transitions
    fig = go.Figure(data=[
        go.Parcats(
            dimensions=[
                {'label': 'H&E report grade', 'values': merged_df[df_score.columns[1]],
                 'categoryorder': 'array', 'categoryarray': [0, 1, 2]},
                {'label': 'pathologist fraction grade', 'values': merged_df[df_percentage_label.columns[1]],
                 'categoryorder': 'array', 'categoryarray': [0, 1, 2]}
            ],
            line=dict(
                color=merged_df[df_score.columns[1]],  # Use the score for coloring
                colorscale='Viridis',  # You can choose other color scales
                # colorscale=[[0, 'dodgerblue'], [0.5, 'lightcoral'], [1, 'turquoise']],  # setup the color scale
                shape='hspline'  # Make the lines smooth
            ),
            hoverinfo='all'
        )
    ])
    
    # Update layout of the figure
    fig.update_layout(
        margin=dict(l=40, b=75, r=80, t=20, pad=0),
        title='Ballooning Score to Ballooning Percentage Label Transition',
        title_font_size=25,
        font_size=20,
        autosize=False,
        width=900, 
        height=900,
        title_x=0.5, 
        title_y=0.05
    )
    
    # fig.show()
    fig.write_html('fig_he_rpt_percenatge_trans.html', auto_open=True)

def plot_cross_labels_parcats(ENV_task, 
                              ball_s_csv_filename, stea_s_csv_filename, lob_s_csv_filename):
    '''
    plot the parcat to show the transformation of labels in cases
    '''
    ball_label_dict = query_task_label_dict_fromcsv(ENV_task, ball_s_csv_filename)
    stea_label_dict = query_task_label_dict_fromcsv(ENV_task, stea_s_csv_filename)
    lob_label_dict = query_task_label_dict_fromcsv(ENV_task, lob_s_csv_filename)
    case_ids = sorted(ball_label_dict.keys(), key=ball_label_dict.get )
    ball_s_list, stea_s_list, lob_s_list = [], [], []
    for id in case_ids:
        if (id not in ball_label_dict.keys()) or (id not in stea_label_dict.keys()) or (id not in lob_label_dict.keys()):
            continue
        ball_s_list.append(ball_label_dict[id])
        stea_s_list.append(stea_label_dict[id])
        lob_s_list.append(lob_label_dict[id])
    
    # load labels data
    df = pd.DataFrame({
        'Steatosis': stea_s_list,
        'Ballooning': ball_s_list,
        'Inflammation': lob_s_list
    })
    
    # plot the parcats
    fig = go.Figure(data=[go.Parcats(
        dimensions=[
            {'label': 'Ballooning', 'values': df['Ballooning'], 
             'categoryorder': 'array', 'categoryarray': [0, 1, 2]},
            {'label': 'Steatosis', 'values': df['Steatosis'], 
             'categoryorder': 'array', 'categoryarray': [0, 1, 2, 3]},
            {'label': 'Inflammation', 'values': df['Inflammation'], 
             'categoryorder': 'array', 'categoryarray': [0, 1, 2, 3]}
        ],
        line=dict(
            color=df['Ballooning'],  # use this label to change the color
            colorscale=[[0, 'dodgerblue'], [0.5, 'lightcoral'], [1, 'turquoise']],  # setup the color scale
            shape='hspline'  # smooth line
        )
    )])
    
    fig.update_layout(
        margin=dict(l=18, b=75, r=20, t=20, pad=0),
        title_text="Transformation of case distribution",
        title_font_size=25,
        font_size=20,
        autosize=False,
        width=1200, 
        height=900,
        title_x=0.5, 
        title_y=0.05
        )
    # fig.show()
    fig.write_html('fig_label_trans.html', auto_open=True)
    
def plot_cross_labels_parcats_lmh(ENV_task, 
                                  ball_s_csv_filename, stea_s_csv_filename, lob_s_csv_filename):
    '''
    plot the parcat to show the transformation of labels in cases
    but just with labels of low, middle, high
    '''
    ball_label_dict = query_task_label_dict_fromcsv(ENV_task, ball_s_csv_filename)
    stea_label_dict = query_task_label_dict_fromcsv(ENV_task, stea_s_csv_filename)
    lob_label_dict = query_task_label_dict_fromcsv(ENV_task, lob_s_csv_filename)
    case_ids = sorted(ball_label_dict.keys(), key=ball_label_dict.get )
    ball_s_list, stea_s_list, lob_s_list = [], [], []
    high_ball_s, high_stea_s, high_lob_s = 2, 3, 3
    for id in case_ids:
        if (id not in ball_label_dict.keys()) or (id not in stea_label_dict.keys()) or (id not in lob_label_dict.keys()):
            continue
        if ball_label_dict[id] == 0:
            ball_l = 'low'
        elif ball_label_dict[id] == high_ball_s:
            ball_l = 'high'
        else:
            ball_l = 'mid'
        if stea_label_dict[id] == 0:
            stea_l = 'low'
        elif stea_label_dict[id] == high_stea_s:
            stea_l = 'high'
        else:
            stea_l = 'mid'
        if lob_label_dict[id] == 0:
            lob_l = 'low'
        elif lob_label_dict[id] == high_lob_s:
            lob_l = 'high'
        else:
            lob_l = 'mid'
        ball_s_list.append(ball_l)
        stea_s_list.append(stea_l)
        lob_s_list.append(lob_l)
        
    # setup a color dict
    label_to_num = {'low': 0, 'mid': 1, 'high': 2}
    ball_s_numeric = [label_to_num[label] for label in ball_s_list]
    
    # load labels data
    df = pd.DataFrame({
        'Steatosis': stea_s_list,
        'Ballooning': ball_s_list,
        'Inflammation': lob_s_list,
        'Color': ball_s_numeric
    })
    
    # plot the parcats
    fig = go.Figure(data=[go.Parcats(
        dimensions=[
            {'label': 'Ballooning', 'values': df['Ballooning'], 
             'categoryorder': 'array', 'categoryarray': ['low', 'mid', 'high']},
            {'label': 'Steatosis', 'values': df['Steatosis'], 
             'categoryorder': 'array', 'categoryarray': ['low', 'mid', 'high']},
            {'label': 'Inflammation', 'values': df['Inflammation'], 
             'categoryorder': 'array', 'categoryarray': ['low', 'mid', 'high']}
        ],
        line=dict(
            color=df['Color'],  # use this label to change the color
            colorscale=[[0, 'dodgerblue'], [0.5, 'lightcoral'], [1, 'turquoise']],  # setup the color scale
            shape='hspline'  # smooth line
        )
    )])
    
    fig.update_layout(
        margin=dict(l=18, b=75, r=20, t=20, pad=0),
        title_text="Transformation of case distribution",
        title_font_size=25,
        font_size=20,
        autosize=False,
        width=1200, 
        height=900,
        title_x=0.5, 
        title_y=0.05
        )
    # fig.show()
    fig.write_html('fig_label_trans_lmh.html', auto_open=True)
    
def plot_henning_fraction_dist(ENV_task, percentage_csv_name=None, auto_bins=False):
    '''
    visualisation the distribution of bio-marker percentage from Henning and Marta's results
    '''
    percentage_dict = load_percentages_from_csv(ENV_task, percentage_csv_name)
    percentage_values = np.asarray(list(percentage_dict.values()))
    
    plt.figure(figsize=(12, 3))
    data_le_0_2 = percentage_values[percentage_values <= 0.2]
    data_bt_0_2_1 = percentage_values[(0.2 < percentage_values) & (percentage_values <= 1)]
    data_gt_1 = percentage_values[percentage_values > 1]
    sns.histplot(data_le_0_2, color='orange', kde=False, binwidth=0.05)
    sns.histplot(data_bt_0_2_1, color='skyblue', kde=False, binwidth=0.05)
    sns.histplot(data_gt_1, color='red', kde=False, binwidth=0.05)
    # sns.histplot(percentage_values, kde=True, color='skyblue', line_kws={'color': 'blue'}) # shutdown KDE
    # sns.kdeplot(data_le_0_2, bw_method='scott', cut=0, clip=(0, 1), color='black')
    
    orange_patch = mpatches.Patch(color='orange', label='≤ 0.2')
    skyblue_patch = mpatches.Patch(color='skyblue', label='0.2 - 1')
    red_patch = mpatches.Patch(color='red', label='> 1')
    plt.legend(handles=[orange_patch, skyblue_patch, red_patch])
        
    plt.xlim(-0.05, )
    # plt.ylim(0, 50)
    plt.title(f'Distribution of {ENV_task.STAIN_TYPE} Percentage')
    plt.xlabel(f'{ENV_task.STAIN_TYPE} percentage')
    plt.ylabel('Frequency')
    # using histplot and turn on the KDE fitting
    
    plt.tight_layout()
    plt.show()
    
''' -------------------------------------------------------------------------------- '''
    
def calculate_entropy(values, bins=80):
    """
    calculate the entropy for a distribution
    """
    # split into many bins to statistic the probability in different ranges
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    probabilities = hist * np.diff(bin_edges)
    # entropy
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-6))  # avoid log(0)
    return entropy

def gini_coefficient(values):
    """ 
    calculate Gini coefficient
    """
    # sort the values
    sorted_values = sorted(values)
    n = len(values)
    cumulative_sum = sum((i+1)*val for i, val in enumerate(sorted_values))
    # use formulate
    gini = (2 * cumulative_sum) / (n * sum(sorted_values)) - (n + 1) / n
    return gini
    
def plot_tis_pct_dist_clsts_in_slides(ENV_task, clst_hiera_pkl_name, tis_pct_pkl_name,
                                      on_fraction_thd=0.0, higher_thd=True, color='cornflowerblue'):
    '''
    count the distribution of each (sub)cluster's tissue percentage on all slides
    PS: best here to plot for each label of (sub)cluster one by one
    
    Args:
        on_fraction_thd: cannot be < 0.0
    '''
    meta_folder = ENV_task.META_FOLDER
    model_store_dir = ENV_task.MODEL_FOLDER
    heatmap_store_dir = ENV_task.HEATMAP_STORE_DIR
    clst_hiera_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clst_hiera_pkl_name)
    # load the cluster labels and remove the repeated
    unique_clst_labels = set([res for res, _, _, _ in clst_hiera_res_pkg])
    slide_tis_pct_dict = load_vis_pkg_from_pkl(heatmap_store_dir, tis_pct_pkl_name)
    
    tissue_pct_by_label = {label: [] for label in unique_clst_labels}
    # load the tissue percentage in slides
    if on_fraction_thd > 0.0:
        ballooning_df = pd.read_csv(os.path.join(meta_folder, 'P62_ballooning_pct.csv'))
        if higher_thd:
            filtered_ballooning_df = ballooning_df[ballooning_df['ballooning_percentage'] >= on_fraction_thd]
            for slide_id, (tis_pct_dict, _) in slide_tis_pct_dict.items():
                case_id = parse_caseid_from_slideid(slide_id)
                if case_id in filtered_ballooning_df['slide_id'].values:
                    for label in unique_clst_labels:
                        if label in tis_pct_dict:
                            tissue_pct_by_label[label].append(tis_pct_dict[label])
        else:
            filtered_ballooning_df = ballooning_df[ballooning_df['ballooning_percentage'] < on_fraction_thd]
            for slide_id, (tis_pct_dict, _) in slide_tis_pct_dict.items():
                case_id = parse_caseid_from_slideid(slide_id)
                if case_id in filtered_ballooning_df['slide_id'].values:
                    for label in unique_clst_labels:
                        if label in tis_pct_dict:
                            tissue_pct_by_label[label].append(tis_pct_dict[label])
    else:
        for slide_id, (tis_pct_dict, _) in slide_tis_pct_dict.items():
            for label in unique_clst_labels:
                if label in tis_pct_dict:
                    tissue_pct_by_label[label].append(tis_pct_dict[label])
                
    img_save_folder = os.path.join(heatmap_store_dir, 'clst_tis-pct_dist_in_slides')
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)
        print(f'folder newly created: {img_save_folder}...')
    else:
        print(f'folder already exists: {img_save_folder}...')
        
    # make plot and save the figures for each label one by one
    for label, percentages in tissue_pct_by_label.items():
        if on_fraction_thd == 0.0:
            plt.figure(figsize=(10, 10))  
        else:
            plt.figure(figsize=(5, 10))
        # print(len(percentages))
        percentages = np.array(percentages) * 100
        nb_bins = 80 if on_fraction_thd==0.0 else 30
        sns.histplot(percentages, bins=nb_bins, kde=True, color=color)
        
        # calculate the gini score (close to 0 -> more average) and entropy (higher -> more average)
        gini = gini_coefficient(percentages)
        entropy = calculate_entropy(percentages, bins=nb_bins)
        
        plt.title(f'Tissue Percentage Distribution for {label}')
        plt.xlabel('Tissue Percentage (%)')
        plt.ylabel('Density')
        plt.text(0.95, 0.95, f'Gini: {gini:.2f}\nEntropy: {entropy:.2f}', 
                 transform=plt.gca().transAxes, fontsize=20, horizontalalignment='right',
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.tight_layout()
    
        h_l_str = 'h' if higher_thd else 'l'
        filter_suffix = '' if on_fraction_thd == 0.0 else f'_{h_l_str}-{on_fraction_thd}'
        plot_f_name = f"{img_save_folder}/tissue_percentage_distribution-{label}{filter_suffix}.png"
        plt.savefig(plot_f_name)
        plt.close()
    
        print(f'plot saved: {plot_f_name}')
        
def plot_tis_pct_henning_fraction_correlation(ENV_task, clst_hiera_pkl_name, tis_pct_pkl_name,
                                              higher_fraction_thd=0.0, color='salmon'):
    '''
    statistic and plot the correlation between tissue percentage of each cluster
        and henning's fraction score, to find the ballooning sensitive clusters
        
    PS: best here to plot for each label of (sub)cluster one by one
    '''
    meta_folder = ENV_task.META_FOLDER
    model_store_dir = ENV_task.MODEL_FOLDER
    heatmap_store_dir = ENV_task.HEATMAP_STORE_DIR
    clst_hiera_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clst_hiera_pkl_name)
    # load the cluster labels and remove the repeated
    unique_clst_labels = set([res for res, _, _, _ in clst_hiera_res_pkg])
    slide_tis_pct_dict = load_vis_pkg_from_pkl(heatmap_store_dir, tis_pct_pkl_name)
    
    henning_fraction_df = pd.read_csv(os.path.join(meta_folder, 'P62_ballooning_pct.csv'))
    print(henning_fraction_df)
    filtered_henning_fraction_df = henning_fraction_df[henning_fraction_df['ballooning_percentage'] > higher_fraction_thd]
    correlation_results = {}
    
    img_save_folder = os.path.join(heatmap_store_dir, 'clst_tis-pct_henning-fract_corr')
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)
        print(f'folder newly created: {img_save_folder}...')
    else:
        print(f'folder already exists: {img_save_folder}...')
    
    # for each cluster, calculate the correlation and plot the scatter
    for label in unique_clst_labels:
        tissue_pcts = []
        henning_fractions = []
    
        # load tissue percentage and henning fractions of all slides
        if higher_fraction_thd == 0.0:
            for slide_id, (tis_pct_dict, _) in slide_tis_pct_dict.items():
                if label in tis_pct_dict:
                    tissue_pct = tis_pct_dict[label]
                    # slide_id in csv actually is case_id
                    case_id = parse_caseid_from_slideid(slide_id)
                    ballooning_pct = henning_fraction_df[henning_fraction_df['slide_id'] == case_id]['ballooning_percentage'].values[0]
        
                    tissue_pcts.append(tissue_pct)
                    henning_fractions.append(ballooning_pct)
        else:
            for slide_id, (tis_pct_dict, _) in slide_tis_pct_dict.items():
                case_id = parse_caseid_from_slideid(slide_id)
                # only pick the slides with henning's fraction higher than 0.2
                if case_id in filtered_henning_fraction_df['slide_id'].values:
                    if label in tis_pct_dict:
                        tissue_pct = tis_pct_dict[label]
                        ballooning_pct = filtered_henning_fraction_df[filtered_henning_fraction_df['slide_id'] == case_id]['ballooning_percentage'].values[0]
        
                        tissue_pcts.append(tissue_pct)
                        henning_fractions.append(ballooning_pct)
    
        tissue_pcts = np.array(tissue_pcts) * 100 # use %
        # pearson score
        correlation, _ = pearsonr(tissue_pcts, henning_fractions)
        correlation_results[label] = correlation
    
        # scatter and regression line
        if higher_fraction_thd == 0.0:
            plt.figure(figsize=(10, 8))
        else:
            plt.figure(figsize=(5, 8))
        sns.regplot(x=tissue_pcts, y=henning_fractions, color=color)
        if higher_fraction_thd == 0.0:
            plt.title(f'Tissue Percentage vs. Pathologist P62 Fraction for {label}')
        else:
            plt.title(f'Tis Pct vs. Pat P62 Frat for {label}')
        plt.xlabel('Tissue Percentage')
        plt.ylabel('Pathologist P62 Fraction')
        if higher_fraction_thd == 0.0:
            plt.ylim(0, 2.0)
        else:
            plt.ylim(0, 5.0)
        
        # embed the text of pearson score to the figures
        plt.text(0.05, 0.95, f'Pearson Correlation: {correlation:.2f}', 
                 transform=plt.gca().transAxes, fontsize=20, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.tight_layout()
        
        filter_suffix = '' if higher_fraction_thd == 0.0 else f'_h-{higher_fraction_thd}'
        save_path = os.path.join(img_save_folder, f'tis-pct_vs_fraction_{label}{filter_suffix}.png') 
        plt.savefig(save_path)
        plt.close()
        print(f"figure for {label} saved: {save_path}")
    
    # print the correlation results
    print("Correlation results:")
    for label, correlation in correlation_results.items():
        print(f"{label}: {correlation:.2f}")
        
''' ----------------- some statistic and results printing functions  ----------------- '''

def stat_dist_gini_ent_clsts_in_slides(ENV_task, clst_hiera_pkl_name, tis_pct_pkl_name,
                                       on_fraction_thd=0.0, higher_thd=True):
    '''
    statistic the gini and entropy results for clusters, and return
    '''
    meta_folder = ENV_task.META_FOLDER
    model_store_dir = ENV_task.MODEL_FOLDER
    heatmap_store_dir = ENV_task.HEATMAP_STORE_DIR
    clst_hiera_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clst_hiera_pkl_name)
    # load the cluster labels and remove the repeated
    unique_clst_labels = set([res for res, _, _, _ in clst_hiera_res_pkg])
    slide_tis_pct_dict = load_vis_pkg_from_pkl(heatmap_store_dir, tis_pct_pkl_name)
    
    tissue_pct_by_label = {label: [] for label in unique_clst_labels}
    # load the tissue percentage in slides
    if on_fraction_thd > 0.0:
        ballooning_df = pd.read_csv(os.path.join(meta_folder, 'P62_ballooning_pct.csv'))
        if higher_thd:
            filtered_ballooning_df = ballooning_df[ballooning_df['ballooning_percentage'] >= on_fraction_thd]
            for slide_id, (tis_pct_dict, _) in slide_tis_pct_dict.items():
                case_id = parse_caseid_from_slideid(slide_id)
                if case_id in filtered_ballooning_df['slide_id'].values:
                    for label in unique_clst_labels:
                        if label in tis_pct_dict:
                            tissue_pct_by_label[label].append(tis_pct_dict[label])
        else:
            filtered_ballooning_df = ballooning_df[ballooning_df['ballooning_percentage'] < on_fraction_thd]
            for slide_id, (tis_pct_dict, _) in slide_tis_pct_dict.items():
                case_id = parse_caseid_from_slideid(slide_id)
                if case_id in filtered_ballooning_df['slide_id'].values:
                    for label in unique_clst_labels:
                        if label in tis_pct_dict:
                            tissue_pct_by_label[label].append(tis_pct_dict[label])
    else:
        for slide_id, (tis_pct_dict, _) in slide_tis_pct_dict.items():
            for label in unique_clst_labels:
                if label in tis_pct_dict:
                    tissue_pct_by_label[label].append(tis_pct_dict[label])
               
    clst_gini_ent_dict= {}
    for label, percentages in tissue_pct_by_label.items():
        percentages = np.array(percentages) * 100
        nb_bins = 80 if on_fraction_thd==0.0 else 30
        # calculate the gini score (close to 0 -> more average) and entropy (higher -> more average)
        gini = gini_coefficient(percentages)
        entropy = calculate_entropy(percentages, bins=nb_bins)
        
        clst_gini_ent_dict[label] = (gini, entropy)
        
    return clst_gini_ent_dict

def stat_f_corr_clsts_in_slides(ENV_task, clst_hiera_pkl_name, tis_pct_pkl_name,
                                higher_fraction_thd=0.0):
    '''
    statistic the correlation pearson scores for clusters, and return
    '''
    meta_folder = ENV_task.META_FOLDER
    model_store_dir = ENV_task.MODEL_FOLDER
    heatmap_store_dir = ENV_task.HEATMAP_STORE_DIR
    clst_hiera_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clst_hiera_pkl_name)
    # load the cluster labels and remove the repeated
    unique_clst_labels = set([res for res, _, _, _ in clst_hiera_res_pkg])
    slide_tis_pct_dict = load_vis_pkg_from_pkl(heatmap_store_dir, tis_pct_pkl_name)
    
    henning_fraction_df = pd.read_csv(os.path.join(meta_folder, 'P62_ballooning_pct.csv'))
    print(henning_fraction_df)
    filtered_henning_fraction_df = henning_fraction_df[henning_fraction_df['ballooning_percentage'] > higher_fraction_thd]
    clst_f_corr_dict = {}
    
    for label in unique_clst_labels:
        tissue_pcts = []
        henning_fractions = []
    
        # load tissue percentage and henning fractions of all slides
        if higher_fraction_thd == 0.0:
            for slide_id, (tis_pct_dict, _) in slide_tis_pct_dict.items():
                if label in tis_pct_dict:
                    tissue_pct = tis_pct_dict[label]
                    # slide_id in csv actually is case_id
                    case_id = parse_caseid_from_slideid(slide_id)
                    ballooning_pct = henning_fraction_df[henning_fraction_df['slide_id'] == case_id]['ballooning_percentage'].values[0]
        
                    tissue_pcts.append(tissue_pct)
                    henning_fractions.append(ballooning_pct)
        else:
            for slide_id, (tis_pct_dict, _) in slide_tis_pct_dict.items():
                case_id = parse_caseid_from_slideid(slide_id)
                # only pick the slides with henning's fraction higher than 0.2
                if case_id in filtered_henning_fraction_df['slide_id'].values:
                    if label in tis_pct_dict:
                        tissue_pct = tis_pct_dict[label]
                        ballooning_pct = filtered_henning_fraction_df[filtered_henning_fraction_df['slide_id'] == case_id]['ballooning_percentage'].values[0]
        
                        tissue_pcts.append(tissue_pct)
                        henning_fractions.append(ballooning_pct)
    
        tissue_pcts = np.array(tissue_pcts) * 100 # use %
        # pearson score
        correlation, _ = pearsonr(tissue_pcts, henning_fractions)
        clst_f_corr_dict[label] = correlation
        
    return clst_f_corr_dict

def filter_clst_gini_ent_thd(clst_gini_ent_dict, gini_thd, ent_thd, higher_this_gini=True, higher_this_ent=True):
    '''
    pick(filter) the key clsts with gini entropy threshold
    return the pick clst_label_list and gini_ent_dict
    '''
    picked_clsts, picked_clsts_gini_ent_dict = [], {}
    for label, (gini, entropy) in clst_gini_ent_dict.items():
        gini_right, ent_right = False, False
        
        if higher_this_gini is True:
            gini_right = (gini >= gini_thd)
        else:
            gini_right = (gini < gini_thd)
        
        if higher_this_ent is True:
            ent_right = (entropy >= ent_thd)
        else:
            ent_right = (entropy < ent_thd)
            
        if gini_right is True and ent_right is True:
            picked_clsts.append(label)
            picked_clsts_gini_ent_dict[label] = (gini, entropy)
    
    return picked_clsts, picked_clsts_gini_ent_dict

def filter_clst_f_corr_thd(clst_f_corr_dict, fcorr_thd, higher_this_fcorr=True):
    '''
    pick(filter) the key clsts with fraction correlation threshold
    return the pick clst_label_list and f_corr_dict
    '''
    picked_clsts, picked_clsts_f_corr_dict = [], {}
    for label, correlation in clst_f_corr_dict.items():
        f_corr_right = False
        if higher_this_fcorr is True:
            f_corr_right = (correlation >= fcorr_thd)
        else:
            f_corr_right = (correlation < fcorr_thd)
        
        if f_corr_right is True:
            picked_clsts.append(label)
            picked_clsts_f_corr_dict[label] = correlation
            
    picked_clsts, picked_clsts_f_corr_dict

if __name__ == '__main__':
    pass


