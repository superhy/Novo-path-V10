'''
@author: Yang Hu
'''

from env import ENV
from parames import parames_task


ENV_FLINC_P62_STEA = parames_task(
    project_name=ENV.PROJECT_NAME,
    slide_type=ENV.SLIDE_TYPE,
    scale_factor=ENV.SCALE_FACTOR,
    tile_size=ENV.TILE_H_SIZE,
    tp_tiles_threshold=ENV.TP_TILES_THRESHOLD,
    debug_mode=False,
    task_name='steatosis_score',
    server_root='',
    pc_root='D:/FLINC_dataset/P62/',
    mac_root='',
    meta_folder_name='FLINC/meta',
    train_folder_name='train', 
    test_folder_name='train',
    pred_folder_name = 'prediction',
    stat_folder_name = 'statistic',
    model_folder_name = 'seg_models',
    tiles_folder_name = 'tiles',
    stain_type='P62',
    seg_batch_size=4,
    seg_num_worker=4,
    seg_num_epoch=20000
    )