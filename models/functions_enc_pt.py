'''
@author: Yang Hu
'''
import gc
import random

from models import functions
from models.datasets import load_slides_tileslist, Simple_Tile_Dataset
from models.networks import ViT_D6_H8, store_net, ViT_D3_H4_T, ViT_D4_H6,\
    ViT_Region_4_6
from support.env import devices


def safe_random_sample(pickpool, K):
    
    if len(pickpool) > K:
        return random.sample(pickpool, K)
    else:
        return pickpool


def tissue_tiles_sampling(ENV_task):
    '''
    Return:
        train_tiles_list: a list for all tiles for training from slides mixed up
    '''
    sample_N = ENV_task.NUM_SLIDE_SAMPLES
    
    slide_tiles_dict = load_slides_tileslist(ENV_task, for_train=True)
    train_tiles_list = []
    for slide_id in slide_tiles_dict.keys():
        slide_tiles_list = slide_tiles_dict[slide_id]
        slide_sample_tiles_list = safe_random_sample(slide_tiles_list, sample_N)
        train_tiles_list.extend(slide_sample_tiles_list)
        
    return train_tiles_list


''' -------------------- main functions for pre-training --------------------- '''
def pretrain_dino(ENV_task, vit_net):
    '''
    '''
    pretrain_epoch = ENV_task.NUM_ENC_SSPT_EPOCH
    
    alg_name = 'PT-Dino_{}-{}x{}'.format(ENV_task.TASK_NAME, ENV_task.VIT_SHAPE, ENV_task.VIT_SHAPE)
    
    vit_net = vit_net.cuda()
    learner = vit_net.get_dino_learner()
    optimizer = functions.optimizer_adam_basic(learner, lr=ENV_task.LR_TILE)
    if vit_net.name.startswith('ViT-Region'):
        transform = functions.get_fixsize_transform()
    else:
        transform = functions.get_zoom_transform()
    
    print('On-standby: {} training'.format(alg_name))
    print('Network: {}, train on -> {}'.format(vit_net.name, devices))
    
    pretrain_tiles_set = Simple_Tile_Dataset(tiles_list=[],
                                             transform=transform)
    
    for epoch in range(pretrain_epoch):
        # reload a new set of tiles for pretraining
        train_tiles_list = tissue_tiles_sampling(ENV_task)
        
        pretrain_tiles_set.refresh_samples(train_tiles_list)
        pretrain_tile_dataloader = functions.get_data_loader(pretrain_tiles_set,
                                                             batch_size=ENV_task.MINI_BATCH_TILE,
                                                             num_workers=ENV_task.TILE_DATALOADER_WORKER,
                                                             sf=False)
        epoch_log = functions.dino_epoch(learner=learner, train_loader=pretrain_tile_dataloader,
                                         optimizer=optimizer, epoch_info=(epoch, pretrain_epoch))
        print(epoch_log)
        
        if (epoch + 1) % ENV_task.SSPT_RECORD_PULSE == 0:
            pretrain_model_path = store_net(ENV_task.MODEL_FOLDER,
                                            vit_net, alg_name + '[{}]'.format(str(epoch + 1)), optimizer)
            print('store the dino pretrained model at: {}'.format(pretrain_model_path))
        
        del pretrain_tile_dataloader
    

def pretrain_mae(ENV_task, vit_net):
    '''
    '''
    pretrain_epoch = ENV_task.NUM_ENC_SSPT_EPOCH
    
    alg_name = 'PT-MAE_{}-{}x{}'.format(ENV_task.TASK_NAME, ENV_task.VIT_SHAPE, ENV_task.VIT_SHAPE)
    
    vit_net = vit_net.cuda()
    learner = vit_net.get_mae_learner()
    optimizer = functions.optimizer_adam_basic(learner, lr=ENV_task.LR_TILE)
    transform = functions.get_zoom_transform()
    
    print('On-standby: {} training'.format(alg_name))
    print('Network: {}, train on -> {}'.format(vit_net.name, devices))
    
    pretrain_tiles_set = Simple_Tile_Dataset(tiles_list=[],
                                             transform=transform)
    
    for epoch in range(pretrain_epoch):
        # reload a new set of tiles for pretraining
        train_tiles_list = tissue_tiles_sampling(ENV_task)
        
        pretrain_tiles_set.refresh_samples(train_tiles_list)
        pretrain_tile_dataloader = functions.get_data_loader(pretrain_tiles_set,
                                                             batch_size=ENV_task.MINI_BATCH_TILE,
                                                             num_workers=ENV_task.TILE_DATALOADER_WORKER,
                                                             sf=False)
        epoch_log = functions.mae_epoch(learner=learner, train_loader=pretrain_tile_dataloader,
                                        optimizer=optimizer, epoch_info=(epoch, pretrain_epoch))
        print(epoch_log)
        
        if (epoch + 1) % ENV_task.SSPT_RECORD_PULSE == 0:
            pretrain_model_path = store_net(ENV_task.MODEL_FOLDER,
                                            vit_net, alg_name + '[{}]'.format(str(epoch + 1)), optimizer)
            print('store the mae pretrained model at: {}'.format(pretrain_model_path))
        
        del pretrain_tile_dataloader
    

''' --------------------- functions for calling --------------------- '''
        
def _run_pretrain_4_6_dino(ENV_task):
    vit_net = ViT_D4_H6(image_size=ENV_task.TRANSFORMS_RESIZE,
                        patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    pretrain_dino(ENV_task, vit_net)
        
def _run_pretrain_6_8_dino(ENV_task):
    vit_net = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                        patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    pretrain_dino(ENV_task, vit_net)
    
def _run_pretrain_reg_6_8_dino(ENV_task):
    vit_reg_net = ViT_Region_4_6(image_size=144, patch_size=int(144/ENV_task.VIT_SHAPE), channels=3)
    pretrain_dino(ENV_task, vit_reg_net)

    
def _run_pretrain_6_8_mae(ENV_task):
    vit_net = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                        patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    pretrain_mae(ENV_task, vit_net)    


def _run_pretrain_3_4_t_dino(ENV_task):
    ''' only used for test on PC '''
    vit_net = ViT_D3_H4_T(image_size=ENV_task.TRANSFORMS_RESIZE,
                          patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    pretrain_dino(ENV_task, vit_net)


if __name__ == '__main__':
    pass
