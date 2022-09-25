'''
@author: Yang Hu
'''
from support.parames import parames_task
from support.env import ENV

ENV_UKAIH_FAT = parames_task(
    project_name=ENV.PROJECT_NAME,
    slide_type=ENV.SLIDE_TYPE,
    scale_factor=ENV.SCALE_FACTOR,
    tile_size=ENV.TILE_H_SIZE,
    tp_tiles_threshold=ENV.TP_TILES_THRESHOLD,
    debug_mode=False,
    task_name='UKAIH_fat_seg',
                        server_root='',
                        pc_root='D:/LIVER_NASH_dataset/UK_AIH_fat_annotations', 
                        train_folder_name='train', 
                        test_folder_name='train',
                        pred_folder_name = 'prediction',
                        model_folder_name = 'seg_models',
                        seg_batch_size=4,
                        seg_num_worker=4,
                        seg_num_epoch=200)