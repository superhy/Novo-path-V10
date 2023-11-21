'''
Created on 25 Nov 2022

@author: Yang Hu
'''
import os
import numpy as np

from interpre.draw_maps import draw_original_image, draw_attention_heatmap
from interpre.prep_tools import load_vis_pkg_from_pkl
from interpre.statistics import df_p62_s_clst_assim_tis_pct_ball_dist, \
    df_p62_s_clst_and_assim_t_p_ball_corr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from support.files import parse_slideid_from_filepath, parse_caseid_from_slideid
from support.metadata import query_task_label_dict_fromcsv
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
    
    
def _plot_activation_kde_dist(ENV_task, ENV_label_hv,
                              act_scores_pkl_name, act_type=0, cut_top=None):
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
        
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(5, 5))
    for tag, activation in activations.items():
        print(tag, np.sum(activation > 0.5), 
              np.min(activation), np.max(activation),
              np.average(activation), np.median(activation) )
        label_name = f'{name_dict[ENV_task.STAIN_TYPE]} {tag}' if tag != 'HV' else 'Healthy volunteers'
        sns.kdeplot(activation, fill=True, clip=(0, np.max(activation)),
                    label=label_name)
    
    plt.xlim(0, 1.0)
    plt.title('Activation dist for diff-groups')
    plt.xlabel('Activation')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(heat_store_dir, 'groups_activation_distribution-kde-{}.png'.format('ft' if act_type == 0 else 'org') ) )
    print('store the picture in {}'.format(os.path.join(heat_store_dir, 
                                                        'groups_activation_distribution-kde-?.png')) )
    plt.clf()
       
    
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
    
        
def _plot_spatial_sensi_clusters_assims(ENV_task, ENV_annotation, spatmap_pkl_name):
    '''
    plot the original image and the spatial map of the sensitive clusters as well as the assimilated tiles (optional)
    '''
    _env_heatmap_store_dir = ENV_task.HEATMAP_STORE_DIR
    _env_task_name = ENV_task.TASK_NAME
    
    slide_topk_heatmap_dict = load_vis_pkg_from_pkl(_env_heatmap_store_dir,
                                                    spatmap_pkl_name)
    label_dict = query_task_label_dict_fromcsv(ENV_annotation)
    
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
    

if __name__ == '__main__':
    pass


