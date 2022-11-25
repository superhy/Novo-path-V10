'''
Created on 25 Nov 2022

@author: Yang Hu
'''
import os

from interpre.draw_maps import draw_original_image
from support.files import parse_slideid_from_filepath
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

if __name__ == '__main__':
    pass