'''
@author: Yang Hu
'''

import gc
import glob
import os
import pickle
import warnings

import PIL
import cmapy
import cv2
import torch

import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from models.datasets import plt_img_mask, Naive_Tiles_Dataset
from models.functions import get_data_loader, train_seg_epoch, dice_bce_loss, \
    optimizer_adam_basic, optimizer_rmsprop_basic, bce_loss, dice_loss, \
    get_transform, bce_logits_loss, mse_loss
from models.seg_networks import store_net, UNet, reload_net
import numpy as np
import seaborn as sns
from support import env_gtex_seg
from support.tools import Time
from wsi import image_tools
from wsi.process import recovery_tiles_list_from_pkl


def store_stat_dicts_pkl(stat_dir, stat_pkl_path, stat_dict):
    if not os.path.exists(stat_dir):
        os.makedirs(stat_dir)
        
    storage_path = os.path.join(stat_dir, stat_pkl_path) if stat_pkl_path.find(stat_dir) == -1 else stat_pkl_path
    with open(storage_path, 'wb') as f_pkl:
        pickle.dump(stat_dict, f_pkl)
        
def load_stat_dicts_from_pkl(stat_pkl_path):
    with open(stat_pkl_path, 'rb') as f_pkl:
        stat_dict = pickle.load(f_pkl)
        
    return stat_dict

def filter_countours_in_tilemask(tile_mask, threshold, solidity=0.5):
    '''
    '''
    blank_canvas = np.zeros(tile_mask.shape)
    mask = tile_mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     cv2.imwrite('test.png', filter_tile_mask)

    sp_nuclei_contours = [] #, sp_nuclei_num = [], 0
    for i, cont in enumerate(contours):
        area = cv2.contourArea(cont)
        if (area > threshold[0]) & (area < threshold[1]):
            hull = cv2.convexHull(cont)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area != 0 else 0
            if solidity >= solidity:
                sp_nuclei_contours.append(cont)
#                 sp_nuclei_num += area
#                 sp_nuclei_num += 1
                
    filter_tile_mask = cv2.drawContours(blank_canvas, sp_nuclei_contours, -1, 255, cv2.FILLED)
#     cv2.imwrite('test.png', filter_tile_mask)   
    print('get nuclei mask size: %s, number %d' % (str(threshold), len(sp_nuclei_contours) ), end=' ' )
    
    del mask
    gc.collect()
    
    return filter_tile_mask, len(sp_nuclei_contours)
    

def pred_slide_nuclei_mask(ENV_task, slide_tiles_dataset, net, org_slide, pred_mask_path,
                           mask=True, green=False, sp_nuclei=False, nuclei_color_dict={}):
    '''
    '''
    time = Time()
    slide_tiles_loader = get_data_loader(dataset=slide_tiles_dataset,
                                         seg_batch_size=1, SEG_NUM_WORKERs=0, sf=False)
    net.eval()
    
    large_w, large_h = org_slide.dimensions
    slide_mask = np.zeros((large_h, large_w), dtype=np.uint8)
    
    if len(nuclei_color_dict.keys()) == 0:
        sp_nuclei = False
    if green == False and sp_nuclei == False:
        mask = True
    
    nuclei_mask_dict, nuclei_tile_stat_dict = {}, {}
    ''' 
    prepare nuclei_name filter canvas, area counter
    
    Format:
        nuclei_tile_stat_dict: {[(tile, sp_nuclei_num)]}
    '''
    for nuclei_name in nuclei_color_dict.keys():
        nuclei_mask_dict[nuclei_name] = np.zeros((large_h, large_w), dtype=np.uint8)
        nuclei_tile_stat_dict[nuclei_name] = []
    
    tissue_area = 0
    with torch.no_grad():
        for X, w_s, w_e, h_s, h_e, idx in slide_tiles_loader:
            X = X.cuda()
            y_mask = net(X)
            y_mask = y_mask.data.cpu()
            
            for i in range(len(w_s)):
                # each mini batch
                tile = slide_tiles_dataset.query_tile(idx[i])
                tile_mask = np.asarray(y_mask[i])[0]
                tile_mask = tile_mask[0:h_e - h_s, 0:w_e - w_s]
                tile_mask[tile_mask >= 0.5] = 255
                tile_mask[tile_mask < 0.5] = 0
                w_s, w_e, h_s, h_e = int(w_s[i]), int(w_e[i]), int(h_s[i]), int(h_e[i])
                tile_mask = tile_mask.astype(np.uint8)
                slide_mask[h_s:h_e, w_s:w_e] = tile_mask

#                 tissue_area += ((h_e - h_s) * (w_e - w_s))
                for nuclei_name in nuclei_mask_dict.keys():
                    filter_tile_mask, sp_nuclei_num = filter_countours_in_tilemask(tile_mask=tile_mask, 
                                                                                    threshold=nuclei_color_dict[nuclei_name][0], 
                                                                                    solidity=nuclei_color_dict[nuclei_name][1])
                    nuclei_mask_dict[nuclei_name][h_s:h_e, w_s:w_e] = filter_tile_mask
                    nuclei_tile_stat_dict[nuclei_name].append((tile, sp_nuclei_num) )
                
                print('insert: ', ('%d ~ %d' % (h_s, h_e), '%d ~ %d' % (w_s, w_e) ), '-> to: ', (large_h, large_w) )
    
    if ENV_task.OS_NAME == 'Windows':
        cv_large_w, cv_large_h = int(large_w / 8) , int(large_h / 8)
    else:
        cv_large_w, cv_large_h = int(large_w / 4) , int(large_h / 4)
        
    if mask is True:
        slide_mask = cv2.resize(slide_mask, (cv_large_w, cv_large_h) )
        cv2.imwrite(pred_mask_path, slide_mask)
        print('Write slide mask at: %s, used time: %s.' % (pred_mask_path, str(time.elapsed())[:-5]))
    
    if green is True:
        slide_mask_green = np.ones((cv_large_h, cv_large_w, 3) ) * 255
        slide_mask_green[slide_mask == 255] = [0, 255, 0]
        cv2.imwrite(pred_mask_path.replace('mask.', 'nuclei_name.'), slide_mask_green)
        print('Write slide mask (green) at: %s' % pred_mask_path.replace('mask.', 'nuclei.') )
    
    for nuclei_name in nuclei_mask_dict.keys():
        nuclei_mask = cv2.resize(nuclei_mask_dict[nuclei_name], (cv_large_w, cv_large_h) )
#         slide_mask_nuclei = np.ones((cv_large_h, cv_large_w, 3) ) * 255
        if sp_nuclei is True:
            slide_mask_nuclei = slide_mask_green
            slide_mask_nuclei[nuclei_mask == 255] = nuclei_color_dict[nuclei_name][2]
            cv2.imwrite(pred_mask_path.replace('mask.', '{}.'.format(nuclei_name)), slide_mask_nuclei)
            print('Write nuclei_name mask (%s) at: %s' % (nuclei_name, pred_mask_path.replace('mask.', '{}.'.format(nuclei_name) ) ) )
            
        stat_dict_pkl_path = pred_mask_path.replace('mask.png', '{}.pkl'.format(nuclei_name + '-st'))
        stat_dict_pkl_path = stat_dict_pkl_path.replace('prediction', 'statistic')
        store_stat_dicts_pkl(ENV_task.STATISTIC_FOLDER_PATH, stat_dict_pkl_path, nuclei_tile_stat_dict[nuclei_name])
        print('Store the statistic pkl at: %s' % stat_dict_pkl_path )
        
    del nuclei_mask_dict
    gc.collect()
    
    return nuclei_tile_stat_dict
        
            
def pred_segmentation_nuclei_filter(ENV_task, net):
    '''
    '''
    slide_tiles_path = glob.glob(os.path.join(ENV_task.TILES_FOLDER_PATH, '*.pkl'))
    
    '''
    {
        'nuclei name': (size_threshold,
                        solidity,
                        color)
    }
    '''
    nuclei_color_dict = {'lymphocyte': ((15, 60), 0.5, [0, 0, 255])}
    
    for i, tile_pth in enumerate(slide_tiles_path):
        tiles_list = recovery_tiles_list_from_pkl(tile_pth)
        _, org_slide = tiles_list[0].get_pil_scaled_slide()
        slide_id = tiles_list[0].query_slideid()
        pred_mask_path = os.path.join(ENV_task.PREDICTION_FOLDER_PATH, slide_id + '-mask.png')
        tiles_dataset = Naive_Tiles_Dataset(tiles_list, ENV_task.TILE_H_SIZE, org_slide)
        
        mask, green, sp_nuclei = True, False, False
        _ = pred_slide_nuclei_mask(ENV_task, tiles_dataset, net, org_slide, pred_mask_path,
                                   mask=mask, green=green, sp_nuclei=sp_nuclei, nuclei_color_dict=nuclei_color_dict)
    
    
def apply_mask(heat_med, white_mask):
    new_heat = np.uint32(np.float64(heat_med) + np.float64(white_mask))
    new_heat = np.uint8(np.minimum(new_heat, 255))
    return new_heat
    
def trans_slidepath_server_to_pc(ENV_task, sever_slide_path):
    slide_filename = sever_slide_path.split('/')[-1]
    pc_slide_path = os.path.join(ENV_task.TRAIN_FOLDER_PATH, slide_filename)
    return pc_slide_path
    
def draw_attention_heatmap(ENV_task, visualization, slide_id):
    '''
    '''
    im_filename = slide_id + '-density.png'
    im_filepath = os.path.join(ENV_task.STATISTIC_FOLDER_PATH, im_filename)
#     visualization = image_tools.convert_rgb_to_bgr(visualization)
    cv2.imwrite(im_filepath, visualization)
    
def draw_sns_kdeplot(ENV_task, kdeplot_label, 
                     slide_id_1, slide_id_2, 
                     tile_sp_nuclei_num_list_1, tile_sp_nuclei_num_list_2):
    '''
    '''
    kde_label_list_1 = [kdeplot_label[slide_id_1]] * len(tile_sp_nuclei_num_list_1)
    kde_label_list_2 = [kdeplot_label[slide_id_2]] * len(tile_sp_nuclei_num_list_2)
    comp_density_frame = pd.DataFrame(
        {
            'slide_feature': kde_label_list_1 + kde_label_list_2,
            'small_cells_number': tile_sp_nuclei_num_list_1 + tile_sp_nuclei_num_list_2
        })
    
    ax = sns.kdeplot(
       data=comp_density_frame, x="small_cells_number", hue="slide_feature",
       fill=True, common_norm=False, palette="Set2",
       alpha=.5, linewidth=1,
    )
    
#     ax.axes.yaxis.set_ticks()
    ax.axes.yaxis.set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
def draw_3d_heatmap(ENV_task, H, W, np_density_heat):
    '''
    '''
    
    fig, ax_1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 6))
    
#     ax_1 = fig.add_subplot(2, 1, 1)
    X1 = np.arange(0, H, 1)
    Y1 = np.arange(0, W, 1)
    print(X1.shape, Y1.shape)
    X1, Y1 = np.meshgrid(X1, Y1)
    print(X1.shape, Y1.shape)
    Z1 = np_density_heat
    print(Z1.shape)
    surf_1 = ax_1.plot_surface(X1, Y1, Z1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
    ax_1.set_zlim(-50.0, 280.0)
    
    plt.tight_layout()
    plt.show()
    
def draw_3d_heatmap_compare(ENV_task, H1, W1, np_density_heat_1, H2, W2, np_density_heat_2):
    '''
    '''
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 12), subplot_kw={"projection": "3d"})
    
#     ax_1 = fig.add_subplot(2, 1, 1)
    X1 = np.arange(0, H1, 1)
    Y1 = np.arange(0, W1, 1)
    print(X1.shape, Y1.shape)
    X1, Y1 = np.meshgrid(X1, Y1)
    print(X1.shape, Y1.shape)
    Z1 = np_density_heat_1
    print(Z1.shape)
    ax[0].set( title='Case 1: no_abnormalities')
    surf_1 = ax[0].plot_surface(X1, Y1, Z1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
    ax[0].set_zlim(-50.0, 280.0)
    
    X2 = np.arange(0, H2, 1)
    Y2 = np.arange(0, W2, 1)
    print(X2.shape, Y2.shape)
    X2, Y2 = np.meshgrid(X2, Y2)
    print(X2.shape, Y2.shape)
    Z2 = np_density_heat_2
    print(Z2.shape)
    ax[1].set( title='Case 2: cirrhosis, inflammation')
    surf_2 = ax[1].plot_surface(X2, Y2, Z2, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
    ax[1].set_zlim(-50.0, 280.0)
    
    plt.tight_layout()
    plt.show()
    
    
def stat_comparison_2slides(ENV_task, slide_tile_stat_name_1, slide_tile_stat_name_2,
                            kdeplot_label, sp_nuclei='lymphocyte'):
    '''
    '''
    stat_dir = ENV_task.STATISTIC_FOLDER_PATH
    
    nuclei_stat_list_1 = load_stat_dicts_from_pkl(os.path.join(stat_dir, slide_tile_stat_name_1))
    nuclei_stat_list_2 = load_stat_dicts_from_pkl(os.path.join(stat_dir, slide_tile_stat_name_2))
    
    slide_id_1 = nuclei_stat_list_1[0][0].query_slideid()
    slide_id_2 = nuclei_stat_list_2[0][0].query_slideid()
    
    demo_tile_1 = nuclei_stat_list_1[0][0]
    demo_tile_2 = nuclei_stat_list_2[0][0]
    demo_tile_1.original_slide_filepath = trans_slidepath_server_to_pc(ENV_task, demo_tile_1.original_slide_filepath)
    demo_tile_2.original_slide_filepath = trans_slidepath_server_to_pc(ENV_task, demo_tile_2.original_slide_filepath)
    
    slide_np_1, _ = demo_tile_1.get_np_scaled_slide()
    H1 = round(slide_np_1.shape[0] * ENV_task.SCALE_FACTOR / ENV_task.TILE_H_SIZE)
    W1 = round(slide_np_1.shape[1] * ENV_task.SCALE_FACTOR / ENV_task.TILE_W_SIZE)
    slide_np_2, _ = demo_tile_2.get_np_scaled_slide()
    H2 = round(slide_np_2.shape[0] * ENV_task.SCALE_FACTOR / ENV_task.TILE_H_SIZE)
    W2 = round(slide_np_2.shape[1] * ENV_task.SCALE_FACTOR / ENV_task.TILE_W_SIZE)
    
    density_heat_1 = np.zeros((H1, W1, 3), dtype=np.float64)
    density_heat_2 = np.zeros((H2, W2, 3), dtype=np.float64)
    np_density_heat_1 = np.zeros((W1, H1), dtype=np.float64)
    np_density_heat_2 = np.zeros((W2, H2), dtype=np.float64)
    white_mask_1 = np.ones((H1, W1, 3), dtype=np.float64)
    white_mask_2 = np.ones((H2, W2, 3), dtype=np.float64)
    
    # count the max sp_nuclei num (one tile) around 2 slides
    tile_sp_nuclei_num_list_1, tile_sp_nuclei_num_list_2 = [], []
    for stat_tuple in nuclei_stat_list_1:
        tile_sp_nuclei_num_list_1.append(stat_tuple[1])
    for stat_tuple in nuclei_stat_list_2:
        tile_sp_nuclei_num_list_2.append(stat_tuple[1])
    max_sp_nuclei_num_1 = np.asarray(tile_sp_nuclei_num_list_1).max()
    max_sp_nuclei_num_2 = np.asarray(tile_sp_nuclei_num_list_2).max()
    print(max_sp_nuclei_num_1, max_sp_nuclei_num_2)
    max_sp_nuclei_num = max(max_sp_nuclei_num_1, max_sp_nuclei_num_2)
    avg_sp_nuclei_num = np.asarray(tile_sp_nuclei_num_list_1 + tile_sp_nuclei_num_list_2).mean()
    
    # make a density heatmap for slide 1
    for i, stat_tuple in enumerate(nuclei_stat_list_1):
        h = stat_tuple[0].h_id - 1 
        w = stat_tuple[0].w_id - 1
        if h >= H1 or w >= W1 or h < 0 or w < 0:
            warnings.warn('Out of range coordinates.')
            continue
        density_heat_1[h, w] = (stat_tuple[1] + 1) / (max_sp_nuclei_num + 1)
        np_density_heat_1[w, h] = stat_tuple[1]
        white_mask_1[h, w] = 0.0
    # make a density heatmap for slide 2
    for i, stat_tuple in enumerate(nuclei_stat_list_2):
        h = stat_tuple[0].h_id - 1 
        w = stat_tuple[0].w_id - 1
        if h >= H2 or w >= W2 or h < 0 or w < 0:
            warnings.warn('Out of range coordinates.')
            continue
        density_heat_2[h, w] = (stat_tuple[1] + 1) / (max_sp_nuclei_num + 1)
        np_density_heat_2[w, h] = stat_tuple[1]
        white_mask_2[h, w] = 0.0
    
    pil_img_type = PIL.Image.BOX
    c_panel = cmapy.cmap('Reds')
    
#     density_heat_1 = np.asarray(density_heat_1 * 255, dtype=np.uint8)
#     density_heat_2 = np.asarray(density_heat_2 * 255, dtype=np.uint8)
#     density_heat_1 = cv2.resize(density_heat_1, (slide_np_1.shape[1], slide_np_1.shape[0]), interpolation=cv2.INTER_NEAREST )
#     density_heat_2 = cv2.resize(density_heat_2, (slide_np_2.shape[1], slide_np_2.shape[0]), interpolation=cv2.INTER_NEAREST )
#     white_mask_1 = cv2.resize(white_mask_1, (slide_np_1.shape[1], slide_np_1.shape[0]), interpolation=cv2.INTER_NEAREST )
#     white_mask_2 = cv2.resize(white_mask_2, (slide_np_2.shape[1], slide_np_2.shape[0]), interpolation=cv2.INTER_NEAREST )
    
    density_heat_1 = image_tools.np_to_pil(density_heat_1).resize((slide_np_1.shape[1], slide_np_1.shape[0]), pil_img_type)
    density_heat_2 = image_tools.np_to_pil(density_heat_2).resize((slide_np_2.shape[1], slide_np_2.shape[0]), pil_img_type)
    white_mask_1 = image_tools.np_to_pil(white_mask_1).resize((slide_np_1.shape[1], slide_np_1.shape[0]), PIL.Image.BOX)
    white_mask_2 = image_tools.np_to_pil(white_mask_2).resize((slide_np_2.shape[1], slide_np_2.shape[0]), PIL.Image.BOX)
    density_heat_1 = np.float64(density_heat_1) / 255
    density_heat_1 = cv2.applyColorMap(np.uint8(255 * density_heat_1), c_panel)
    density_heat_1 = apply_mask(density_heat_1, white_mask_1)
    density_heat_2 = np.float64(density_heat_2) / 255
    density_heat_2 = cv2.applyColorMap(np.uint8(255 * density_heat_2), c_panel)
    density_heat_2 = apply_mask(density_heat_2, white_mask_2)
    print('generate {} density (tile-level) heatmap for slide: {} and {}'.format(sp_nuclei, slide_id_1, slide_id_2) )
#     draw_attention_heatmap(ENV_task, density_heat_1, slide_id_1)
#     draw_attention_heatmap(ENV_task, density_heat_2, slide_id_2)
    print('sp nuclei number average: %.2f, max: %d' % (avg_sp_nuclei_num, max_sp_nuclei_num) )
#     print('plot density heatmap for {} and {}'.format(slide_id_1, slide_id_2))
    
    draw_sns_kdeplot(ENV_task, kdeplot_label, 
                     slide_id_1, slide_id_2, 
                     tile_sp_nuclei_num_list_1, tile_sp_nuclei_num_list_2)

#     draw_3d_heatmap(ENV_task, H1, W1, np_density_heat_1)
#     draw_3d_heatmap(ENV_task, H2, W2, np_density_heat_2)
#     draw_3d_heatmap_compare(ENV_task, H1, W1, np_density_heat_1, H2, W2, np_density_heat_2)
    
        
def _run_segmentation_slides_unet(ENV_task, trained_model_name):
    '''
    '''
    unet = UNet(n_channels=3, n_classes=1, x_width=4)
    trained_model_path = os.path.join(ENV_task.MODEL_FOLDER_PATH, trained_model_name)
    unet, _ = reload_net(unet, trained_model_path)
    unet = unet.cuda()
    
    print('Task: {}, network: {}'.format(ENV_task.TASK_NAME, unet.name))
    pred_segmentation_nuclei_filter(ENV_task, unet)

def _run_stat_sp_nuclei_comparison(ENV_task):
    '''
    '''
    slide_tile_stat_name_1 = 'GTEX-1H1DE-0526-lymphocyte-st.pkl'
    slide_tile_stat_name_2 = 'GTEX-132Q8-1626-lymphocyte-st.pkl'
    
    kdeplot_label = {'GTEX-1H1DE-0526': 'no_abnormalities',
                     'GTEX-132Q8-1626': 'cirrhosis, inflammation'}
    
    stat_comparison_2slides(ENV_task, slide_tile_stat_name_1, slide_tile_stat_name_2,
                            kdeplot_label, sp_nuclei='lymphocyte')

    
''' ------------------------- test functions ------------------------ '''
    
def _test_countours():
    
    mask_path = 'D:\\LIVER_NASH_dataset\\MoNuSeg\\train\\prediction\\TCGA-18-5592-01Z-00-DX1-pred.bmp'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(type(mask), mask.shape, mask, mask.dtype)
    
    nuclei_color_dict = {'lymphocyte': ((15, 50), 0.5, [0, 0, 255])}
    print(len({}))
    
    filter_countours_in_tilemask(mask, (30, 100))
    
def test_np():
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 0, 0.25)
    print(X.shape, Y.shape)
    X, Y = np.meshgrid(X, Y)
    print(X.shape, Y.shape)
#     print(X)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    print(Z.shape)
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    plt.show()
    
def test_seaborn():
    
    tips = sns.load_dataset("tips")
    
    print(tips)
    
    sns.kdeplot(
       data=tips, x="total_bill", hue="size",
       fill=True, common_norm=False, palette="crest",
       alpha=.5, linewidth=0,
    )
    plt.show()

if __name__ == '__main__':
#     _test_countours()
#     test_np()
#     test_seaborn()
    _run_stat_sp_nuclei_comparison(env_gtex_seg.ENV_GTEX_SEG)
#     pass
    
    
    
    