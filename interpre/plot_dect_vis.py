'''
Created on 25 Nov 2022

@author: Yang Hu
'''
import os

from interpre.draw_maps import draw_original_image, draw_attention_heatmap
from interpre.prep_tools import load_vis_pkg_from_pkl
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
    
    
def _plot_attention_heatmaps(ENV_task, ENV_annotation,
                             heatmap_pkl_name,
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
            

def _plot_topK_attention_heatmaps(ENV_task, ENV_annotation, heatmap_pkl_name):
    """
    plot the original heatmap with attention scores for all tiles
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
        
        alg_name = heatmap_pkl_name[heatmap_pkl_name.find('topK_map_') + 10:-15]
        
        single_multi_dir = os.path.join(_env_heatmap_store_dir, 'topk_map')
        attention_dir = os.path.join(single_multi_dir, 'topk_att_dx')
        
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

        draw_attention_heatmap(attention_dir, heat_hard_cv2, org_image, None,
                               (os.path.join('{}//'.format(attention_dir) + str(slide_label), slide_id + '-hard'), alg_name))
        print('1. draw hard heatmap in: {} for slide:{}'.format(os.path.join(_env_heatmap_store_dir, '{}//'.format(attention_dir)), slide_id))
        
        draw_attention_heatmap(attention_dir, heat_soft_cv2, org_image, None,
                               (os.path.join('{}//'.format(attention_dir) + str(slide_label), slide_id + '-soft'), alg_name))
        print('2. draw soft heatmap in: {} for slide:{}'.format(os.path.join(_env_heatmap_store_dir, '{}//'.format(attention_dir)), slide_id))
        

if __name__ == '__main__':
    pass