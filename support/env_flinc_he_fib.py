'''
@author: Yang Hu
'''

from env import ENV
from parames import parames_task


ENV_FLINC_HE_FIB = parames_task(
    project_name=ENV.PROJECT_NAME,
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
    tiles_folder_name = 'tiles',
    stain_type='HE',
    seg_batch_size=4,
    seg_num_worker=4,
    seg_num_epoch=20000
    )
