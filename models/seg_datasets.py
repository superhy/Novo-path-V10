'''
@author: Yang Hu
'''
import glob
import os
import random

from PIL import Image
import PIL
import cv2
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
from wsi.image_tools import pil_to_np_rgb
from wsi.process import recovery_tiles_list_from_pkl
from wsi.slide_tools import original_slide_and_scaled_pil_image


def tnd_reshape_3channels(img):
    '''
    '''
    new_img = []
    for i in range(3):
        new_img.append(img[:, :, i])
    new_img = np.asarray(new_img)
    return new_img

class UKAIH_fat_Dataset(Dataset):
    ''' UKAIH dataset '''
    
    def __init__(self, folder_path, data_aug=False, istest=False):
        self.folder_path = folder_path
        self.images_path = glob.glob(os.path.join(self.folder_path, 'images/*.jpg'))
        self.data_aug = data_aug
        self.istest = istest
        
    def aug_flip(self, image, flip_code):
        flip = cv2.flip(image, flip_code)
        return flip
    
    def __getitem__(self, index):
        image_path = self.images_path[index]
        label_path = image_path.replace('images', 'masks')
        
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = tnd_reshape_3channels(image)
        label = label.reshape(1, label.shape[0], label.shape[1])
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        
        if label.max() > 1:
            label = label / 255
            
        if self.data_aug:
            flip_code = random.choice([-1, 0, 1, 2])
            if flip_code != 2:
                image = self.aug_flip(image, flip_code)
                label = self.aug_flip(label, flip_code)
        
        if self.istest:
            return image, label, image_path.replace('images', 'prediction')
        else:
            return image, label
    
    def __len__(self):
        return len(self.images_path)
   
   
''' MoNuSeg dataset '''
   
def plt_img_mask(img, mask=None):
    
    if mask is None:
        mask = np.zeros(img.shape)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(img)
    ax2.imshow(mask)
    plt.show()
   
def load_MoNuSeg_images_masks(images_folder):
    '''
    '''
    images_path = glob.glob(os.path.join(images_folder, 'images/*.tif') )
    
    images, masks = [], []
    for i, image_path in enumerate(images_path):
        img, _ = original_slide_and_scaled_pil_image(slide_filepath=image_path, 
                                                     scale_factor=1, print_opening=True)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = tnd_reshape_3channels(img)
        img = img.astype(np.float32)
        
        label_path = image_path.replace('images', 'masks')
        label_path = label_path.replace('.tif', '_mask.bmp')
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = label.reshape(1, label.shape[0], label.shape[1])
        
        label = label.astype(np.float32)
        if label.max() > 1:
            label = label / 255
        
        images.append(img)
        masks.append(label)
        
    return images, masks
        
class MoNuSeg_Dataset(Dataset):
    
    def __init__(self, images_list, masks_list, random_crop_size=0, flip=False, transforms: transforms=None):
        self.images_list = images_list
        self.masks_list = masks_list 
        self.random_crop_size = random_crop_size 
        self.flip = flip
        self.transforms = transforms
            
    def aug_crop(self, image, crop_h_s, crop_w_s):
        crop = image[:, crop_h_s: crop_h_s + self.random_crop_size, crop_w_s: crop_w_s + self.random_crop_size]
        return crop
    
    def aug_flip(self, image, flip_code):
        flip = cv2.flip(image, flip_code)
        return flip
    
    def __getitem__(self, index):
        image = self.images_list[index]
        label = self.masks_list[index]
        
        if self.random_crop_size > 32:
            img_h, img_w = image.shape[-2], image.shape[-1]
            crop_h_s = random.randint(0, img_h - self.random_crop_size)
            crop_w_s = random.randint(0, img_w - self.random_crop_size)
            image = self.aug_crop(image, crop_h_s, crop_w_s)
            label = self.aug_crop(label, crop_h_s, crop_w_s)
        if self.flip:
            flip_code = random.choice([-1, 0, 1, 2])
            if flip_code != 2:
                image = self.aug_flip(image, flip_code)
                label = self.aug_flip(label, flip_code)
        image = image.astype(np.float32)
        label = label.astype(np.float32)
                
        if self.transforms is not None:
            image = self.transforms(image)
            label = self.transforms(label)
        
        return image, label
    
    def __len__(self):
        return len(self.images_list)
    
    
''' some with GTEx_seg dataset '''
   
def tile_img_pad_white(broken_tile_img, resize_shape):
    '''
    '''
    old_img = np.asarray(broken_tile_img)
#     print(old_img.shape)
    
    new_img = []
    for i in range(3):
        ext_img = np.ones((resize_shape, resize_shape)) * 255
        ext_img[0:old_img.shape[-2], 0:old_img.shape[-1]] = old_img[i]
        new_img.append(ext_img)
    new_img = np.asarray(new_img)
    
    return new_img


class Naive_Tiles_Dataset(Dataset):
    
    def __init__(self, tile_list, uni_tile_size,
                 preload_slide=None,
                 transforms: transforms=None):
        self.tile_list = tile_list
        self.uni_tile_size = uni_tile_size
        self.preload_slide = preload_slide
        self.transforms = transforms
        
    def query_tile(self, index):
        return self.tile_list[index]
    
    def __getitem__(self, index):
        tile = self.tile_list[index]
        if self.preload_slide is None:
            _, self.preload_slide = tile.get_pil_scaled_slide()
        tile_img = tile.get_pil_tile(self.preload_slide)
        np_tile_img = cv2.cvtColor(np.asarray(tile_img), cv2.COLOR_RGB2BGR)
        np_tile_img = tnd_reshape_3channels(np_tile_img)
        
        w_s, w_e, h_s, h_e = tile.large_w_s, tile.large_w_e, tile.large_h_s, tile.large_h_e
        
        if w_e - w_s < self.uni_tile_size or h_e - h_s < self.uni_tile_size:
            np_tile_img = tile_img_pad_white(np_tile_img, self.uni_tile_size)
        np_tile_img = np_tile_img.astype(np.float32)
            
        if self.transforms is not None:
            np_tile_img = self.transforms(np_tile_img)
        
        return np_tile_img, w_s, w_e, h_s, h_e, index
    
    def __len__(self):
        return len(self.tile_list) 


def load_slides_tile_images(tiles_folder, tile_size):
    '''
    '''
    slide_tiles_path = glob.glob(os.path.join(tiles_folder, '*.pkl'))
    
    for i, tile_pth in enumerate(slide_tiles_path):
        tiles_list = recovery_tiles_list_from_pkl(tile_pth)
        # list the tiles in this slide
        preload_slide = None
        for j, tile in enumerate(tiles_list):
            _, preload_slide = tile.get_pil_scaled_slide()
            np_tile_img = tile.get_np_tile(preload_slide)
            np_tile_img = np_tile_img.reshape(3, np_tile_img.shape[0], np_tile_img.shape[1])
            org_img_shape = [np_tile_img.shape[-2], np_tile_img.shape[-1]]
            if org_img_shape[0] < tile_size or org_img_shape[1] < tile_size:
                np_tile_img = tile_img_pad_white(np_tile_img, [tile_size, tile_size])
            print(np_tile_img)
                
#             print(np_tile_img)
              
            
if __name__ == '__main__':
    load_slides_tile_images('D:/LIVER_NASH_dataset/MoNuSeg/tiles', 512)



