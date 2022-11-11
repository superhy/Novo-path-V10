'''
@author: Yang Hu
'''

import os
import cv2

import numpy as np
from wsi import image_tools



def show_heatmap_on_image(img: np.ndarray,
                          heatmap: np.ndarray):
    '''
    '''
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        img = np.float32(img) / 255

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def draw_original_image(default_dir, image, im_info: tuple=None):
    '''
    '''
    im_filename = im_info[0] + '_' + im_info[1] + '.png'
    im_filepath = os.path.join(default_dir, im_filename)
    image = np.asarray(image)
    
    image = image_tools.convert_rgb_to_bgr(image)
    cv2.imwrite(im_filepath, image)
    
def draw_attention_heatmap(map_dir, 
                           visualization, original_image,
                           troi_contours=None, im_info: tuple=None):
    '''
    '''
    im_filename = im_info[0] + '_' + im_info[1] + '.png'
    im_filepath = os.path.join(map_dir, im_filename)
    
    if im_info[0].find('org') != -1:
        visualization = image_tools.convert_rgb_to_bgr(visualization)
    
    if original_image is not None:
        original_image = image_tools.convert_rgb_to_bgr(original_image)
        visualization = show_heatmap_on_image(original_image, visualization)
    
    if troi_contours is not None:
            cv2.drawContours(visualization, troi_contours, -1, (0, 0, 255), 5)
    
    cv2.imwrite(im_filepath, visualization)

if __name__ == '__main__':
    pass