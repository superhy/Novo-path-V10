'''
@author: Yang Hu
'''

import os
import pickle
import sys

import numpy as np
from support import env_monuseg, env_gtex_seg
from support.env import ENV
from wsi import filter_tools
from wsi import slide_tools
from wsi import tiles_tools
from wsi.tiles_tools import parse_slideid_from_filepath

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



sys.path.append("..")       
def generate_tiles_list_pkl_filepath(slide_filepath, tiles_list_pkl_dir):
    """
    generate the filepath of pickle 
    """
    
    slide_id = parse_slideid_from_filepath(slide_filepath)
    tiles_list_pkl_filename = slide_id + '-(tiles_list)' + '.pkl'
    if not os.path.exists(tiles_list_pkl_dir):
        os.makedirs(tiles_list_pkl_dir)
    
    pkl_filepath = os.path.join(tiles_list_pkl_dir, tiles_list_pkl_filename)
    
    return pkl_filepath

 
def recovery_tiles_list_from_pkl(pkl_filepath):
    """
    load tiles list from [.pkl] file on disk
    (this function is for some other module)
    """
    with open(pkl_filepath, 'rb') as f_pkl:
        tiles_list = pickle.load(f_pkl)
    return tiles_list


def parse_filesystem_slide(slide_dir):
    slide_path_list = []
    for root, dirs, files in os.walk(slide_dir):
        for f in files:
            if f.endswith('.svs') or f.endswith('.tiff') or f.endswith('.tif'):
                slide_path = os.path.join(root, f)
                slide_path_list.append(slide_path)
                
    return slide_path_list

        
def slide_tiles_split_keep_object_seg(slides_folder):
    """
    conduct the whole pipeline of slide's tiles split, by Sequential process
    store the tiles Object [.pkl] on disk
    
    without train/test separation, for segmentation task only  
    
    Args:
        slides_folder: the folder path of slides ready for segmentation
    """
    
    ''' load all slides '''
    slide_path_list = parse_filesystem_slide(slides_folder)
    for i, slide_path in enumerate(slide_path_list):
        np_small_img, large_w, large_h, small_w, small_h = slide_tools.slide_to_scaled_np_image(slide_path)
        np_small_filtered_img = filter_tools.apply_image_filters(np_small_img)
        
        shape_set_img = (large_w, large_h, small_w, small_h)
        tiles_list = tiles_tools.get_slide_tiles(np_small_filtered_img, shape_set_img, slide_path,
                                                 ENV.TILE_W_SIZE, ENV.TILE_H_SIZE,
                                                 t_p_threshold=ENV.TP_TILES_THRESHOLD, load_small_tile=False)
        
        print('generate tiles for slide: %s, keep [%d] tile objects in (.pkl) list.' % (slide_path, len(tiles_list)))
        if len(tiles_list) == 0:
            continue
        
        f_suffix = slide_path.split('.')[-1]
        if slide_path.find('images') != -1:
            pkl_path = slide_path.replace('images', 'tiles')
        else:
            pkl_path = slide_path.replace('slides', 'tiles')
        pkl_path = pkl_path.replace('.' + f_suffix, '-tiles.pkl')
        
        with open(pkl_path, 'wb') as f_pkl:
            pickle.dump(tiles_list, f_pkl)
        return pkl_path
    
def slide_tiles_split_keep_object_cls(ENV_task, test_num_set: tuple=None):
    '''
    conduct the whole pipeline of slide's tiles split, by Sequential process
    store the tiles Object [.pkl] on disk
    
    with train/test separation, for classification task
    
    Args:
        ENV_task:
        test_num_set: list of number of test samples for 
        delete_old_files = True, this is a parameter be deprecated but setup as [True] default
    '''
    
    
            
def _run_monuseg_slide_tiles_split(ENV_task):
    '''
    '''
    slides_folder = os.path.join(ENV_task.SEG_TRAIN_FOLDER_PATH, 'images')
    _ = slide_tiles_split_keep_object_seg(slides_folder)
    
def _run_gtexseg_slide_tiles_split(ENV_task):
    '''
    '''
    slides_folder = ENV_task.SEG_TRAIN_FOLDER_PATH
    _ = slide_tiles_split_keep_object_seg(slides_folder)
    

if __name__ == '__main__': 
#     _run_monuseg_slide_tiles_split(env_monuseg.ENV_MONUSEG)
    _run_gtexseg_slide_tiles_split(env_gtex_seg.ENV_GTEX_SEG) 
    
    
