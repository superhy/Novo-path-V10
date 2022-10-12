'''
Created on 7 Oct 2022

@author: laengs2304
'''
import os

from wsi.filter_tools import apply_image_filters_he, apply_image_filters_psr
from wsi.image_tools import np_to_pil
from wsi.slide_tools import original_slide_and_scaled_pil_image, \
    slide_to_scaled_np_image


def test_filter_slide_img():
    # slide_filepath = 'D:/FLINC_dataset/slides/yang_psr_fib/tissues/23910-157_Sl040-C24-PSR.ndpi'
    slide_filepath = 'D:/FLINC_dataset/slides/yang_he_stea/tissues/23910-157_Sl049-C32-HE.ndpi'
    np_slide_img, _, _, _, _ = slide_to_scaled_np_image(slide_filepath)
    
    np_filtered_img = apply_image_filters_he(np_slide_img)
    # np_filtered_img = apply_image_filters_psr(np_slide_img)
    
    pil_img = np_to_pil(np_filtered_img)
    print(pil_img)
    pil_img.save('test_slide_filter.jpg')
    pil_img.show()
    
test_filter_slide_img()