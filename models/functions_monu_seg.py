'''
@author: Yang Hu
'''

import glob
import os

import cv2
import torch

from models.datasets import load_MoNuSeg_images_masks, MoNuSeg_Dataset, \
    plt_img_mask, Naive_Tiles_Dataset
from models.functions import get_data_loader, train_epoch, dice_bce_loss, \
    optimizer_adam_basic, optimizer_rmsprop_basic, bce_loss, dice_loss, \
    get_transform, bce_logits_loss, mse_loss
from models.seg_networks import store_net, UNet, reload_net
import numpy as np
from wsi.process import recovery_tiles_list_from_pkl


def train_segmentation(ENV_task, net, seg_trainset, optimizer, loss):
    '''
    '''
    seg_trainloader = get_data_loader(dataset=seg_trainset,
                                      seg_batch_size=ENV_task.MINI_BATCH,
                                      SEG_NUM_WORKERs=ENV_task.SEG_NUM_WORKER,
                                      sf=True)
    
    SEG_NUM_EPOCH = ENV_task.SEG_NUM_EPOCH
    for epoch in range(SEG_NUM_EPOCH):
        print('In training... ', end='')
        train_epoch(train_loader=seg_trainloader,
                    net=net,
                    loss=loss,
                    optimizer=optimizer,
                    epoch_info=(epoch, SEG_NUM_EPOCH))
        
        if (epoch + 1) % 500 == 0:
            init_obj_dict = {'epoch': SEG_NUM_EPOCH}
            store_filepath = store_net(store_dir=ENV_task.MODEL_FOLDER_PATH,
                                       trained_net=net, algorithm_name=ENV_task.TASK_NAME + '-{}-'.format(str(epoch + 1)),
                                       optimizer=optimizer, init_obj_dict=init_obj_dict)
    
    return store_filepath


def pred_slide_mask(slide_tiles_dataset, net, org_slide, pred_mask_path, green=False):
    '''
    '''
    slide_tiles_loader = get_data_loader(dataset=slide_tiles_dataset,
                                         seg_batch_size=1, SEG_NUM_WORKERs=0, sf=False)
    net.eval()
    
    large_w, large_h = org_slide.dimensions
    slide_mask = np.zeros((large_h, large_w))
    with torch.no_grad():
        for X, w_s, w_e, h_s, h_e in slide_tiles_loader:
            X = X.cuda()
            y_mask = net(X)
            y_mask = y_mask.data.cpu()
            for i in range(len(w_s)):
                tile_mask = np.array(y_mask[i])[0]
                tile_mask = tile_mask[0:h_e - h_s, 0:w_e - w_s]
                tile_mask[tile_mask >= 0.5] = 255
                tile_mask[tile_mask < 0.5] = 0
                w_s, w_e, h_s, h_e = int(w_s[i]), int(w_e[i]), int(h_s[i]), int(h_e[i])
                slide_mask[h_s:h_e, w_s:w_e] = tile_mask
                
    cv2.imwrite(pred_mask_path, slide_mask)
    print('Write slide mask at: {}.'.format(pred_mask_path))
    
    if green is True:
        slide_mask_green = np.ones((large_h, large_w, 3)) * 255
        slide_mask_green[slide_mask == 255] = [0, 255, 0]
        cv2.imwrite(pred_mask_path.replace('.bmp', '-green.bmp'), slide_mask_green)
        print('Write slide mask (green) at: %s' % pred_mask_path.replace('.bmp', '-green.bmp'))
    
            
def pred_segmentation(ENV_task, net):
    '''
    '''
    slide_tiles_path = glob.glob(os.path.join(ENV_task.TILES_FOLDER_PATH, '*.pkl'))
    
    for i, tile_pth in enumerate(slide_tiles_path):
        tiles_list = recovery_tiles_list_from_pkl(tile_pth)
        _, org_slide = tiles_list[0].get_pil_scaled_slide()
        slide_id = tiles_list[0].query_slideid()
        pred_mask_path = os.path.join(ENV_task.PREDICTION_FOLDER_PATH, slide_id + '-pred.bmp')
        
        tiles_dataset = Naive_Tiles_Dataset(tiles_list, ENV_task.TILE_H_SIZE, org_slide)
        pred_slide_mask(tiles_dataset, net, org_slide, pred_mask_path, green=True)


def _run_seg_train_unet(ENV_task, trained_model_name=None):
    '''
    '''
    unet = UNet(n_channels=3, n_classes=1, x_width=4)
    if trained_model_name is not None:
        trained_model_path = os.path.join(ENV_task.MODEL_FOLDER_PATH, trained_model_name)
        unet, _ = reload_net(unet, trained_model_path)
    unet = unet.cuda()
    
    print('Task: {}, network: {}'.format(ENV_task.TASK_NAME, unet.name))
    loss = mse_loss()
    optimizer = optimizer_adam_basic(unet, lr=1e-4)
    
    images, masks = load_MoNuSeg_images_masks(ENV_task.TRAIN_FOLDER_PATH)
    transforms = get_transform(ENV_task.TRANSFORMS_RESIZE)
    seg_trainset = MoNuSeg_Dataset(images, masks, random_crop_size=ENV_task.TRANSFORMS_RESIZE, flip=False)
    
    model_path = train_segmentation(ENV_task, unet, seg_trainset, optimizer, loss)
    

def _run_segmentation_slides_unet(ENV_task, trained_model_name):
    '''
    '''
    unet = UNet(n_channels=3, n_classes=1, x_width=4)
    trained_model_path = os.path.join(ENV_task.MODEL_FOLDER_PATH, trained_model_name)
    unet, _ = reload_net(unet, trained_model_path)
    unet = unet.cuda()
    
    print('Task: {}, network: {}'.format(ENV_task.TASK_NAME, unet.name))
    pred_segmentation(ENV_task, unet)


if __name__ == '__main__':
    pass

