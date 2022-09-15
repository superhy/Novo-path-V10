'''
@author: Yang Hu
'''

from support.env import ENV
from support.parames import parames_task


ENV_FLINC_P62_FIB = parames_task(
    project_name=ENV.PROJECT_NAME,
    slide_type=ENV.SLIDE_TYPE,
    scale_factor=ENV.SCALE_FACTOR,
    tile_size=ENV.TILE_H_SIZE,
    tp_tiles_threshold=ENV.TP_TILES_THRESHOLD,
    debug_mode=False,
    task_name='fibrosis_score',
    server_root='',
    pc_root='D:/FLINC_dataset/HE/',
    meta_folder_name='FLINC/meta',
    train_folder_name='train', 
    test_folder_name='train',
    pred_folder_name = 'prediction',
    stat_folder_name = 'statistic',
    model_folder_name = 'seg_models',
    tiles_folder_name = 'tiles',
    stain_type='P62',
    batch_size=4,
    num_worker=4,
    num_epoch=20000
    )
