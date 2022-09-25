'''
@author: Yang Hu
'''
import os

from models.datasets import UKAIH_fat_Dataset
from support import env_ukaih_fat
from models.functions import get_data_loader, train_epoch, bce_logits_loss, \
    optimizer_rmsprop_basic, test_epoch, bce_loss, dice_loss, \
    optimizer_adam_basic, dice_bce_loss, mse_loss
from models.seg_networks import UNet, store_net, reload_net


def train_segmentation(ENV_task, net, seg_trainset, seg_testset, 
                       optimizer, loss, record_best=True):
    '''
    '''
    seg_trainloader = get_data_loader(dataset=seg_trainset,
                                      seg_batch_size=ENV_task.MINI_BATCH, 
                                      SEG_NUM_WORKERs=ENV_task.SEG_NUM_WORKER,
                                      sf=True)
    seg_testloader = get_data_loader(dataset=seg_testset,
                                     seg_batch_size=ENV_task.MINI_BATCH, 
                                     SEG_NUM_WORKERs=ENV_task.SEG_NUM_WORKER, 
                                     sf=False)
    
    SEG_NUM_EPOCH = ENV_task.SEG_NUM_EPOCH
    best_loss, store_finalpath = 1e6, None
    for epoch in range(SEG_NUM_EPOCH):
        print('In training... ', end='')
        train_epoch(train_loader=seg_trainloader, 
                    net=net, 
                    loss=loss, 
                    optimizer=optimizer, 
                    epoch_info=(epoch, SEG_NUM_EPOCH))
        if record_best:
            print('In testing... ', end='')
            test_loss = test_epoch(test_loader=seg_testloader, 
                                   net=net, 
                                   loss=loss, 
                                   prediction=False)
            if test_loss < best_loss:
                best_loss = test_loss
                init_obj_dict = {'epoch': epoch + 1,
                                 'test_loss': test_loss}
                store_filepath = store_net(store_dir=ENV_task.MODEL_FOLDER_PATH, 
                                           trained_net=net, algorithm_name=ENV_task.TASK_NAME + '-',
                                           optimizer=optimizer, init_obj_dict=init_obj_dict)
                store_finalpath = store_filepath
                print('record model at: {}'.format(store_filepath))
    
    return store_finalpath

def pred_segmentation_nuclei_filter(ENV_task, net, model_path, loss, seg_testset):
    '''
    '''
    seg_testloader = get_data_loader(dataset=seg_testset,
                                     seg_batch_size=ENV_task.MINI_BATCH, 
                                     SEG_NUM_WORKERs=ENV_task.SEG_NUM_WORKER, 
                                     sf=False)
    if not os.path.exists(ENV_task.PREDICTION_FOLDER_PATH):
        print('Create prediction folder: {}'.format(ENV_task.PREDICTION_FOLDER_PATH))
        os.makedirs(ENV_task.PREDICTION_FOLDER_PATH)
        
    net, _ = reload_net(model_net=net, model_filepath=model_path)
#     net = net.cuda()
    _ = test_epoch(seg_testloader, net, loss, prediction=True)
    print('Output prediction at: {}'.format(ENV_task.PREDICTION_FOLDER_PATH) )

def _run_seg_train_unet(ENV_task):
    '''
    '''
    unet = UNet(n_channels=3, n_classes=1)
    unet = unet.cuda()
    
    print('Task: {}, network: {}'.format(ENV_task.TASK_NAME, unet.name ))
    loss = mse_loss()
    optimizer = optimizer_adam_basic(unet, lr=1e-4)
    
    seg_trainset = UKAIH_fat_Dataset(folder_path=ENV_task.TRAIN_FOLDER_PATH, data_aug=True)
    seg_testset = UKAIH_fat_Dataset(folder_path=ENV_task.TEST_FOLDER_PATH, data_aug=False, istest=True)
    
    model_path = train_segmentation(ENV_task, unet, 
                                    seg_trainset, seg_testset, 
                                    optimizer, loss)
    if model_path is not None:
        pred_segmentation_nuclei_filter(ENV_task, unet, model_path, loss, seg_testset)
    
    

if __name__ == '__main__':
    pass
    
    
    
    