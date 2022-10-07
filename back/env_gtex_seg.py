'''
@author: Yang Hu
'''
from support.parames import parames_task
from support.env import ENV

ENV_GTEX_SEG = parames_task(
    project_name=ENV.PROJECT_NAME,
    scale_factor=ENV.SCALE_FACTOR,
    tile_size=ENV.TILE_H_SIZE,
    tp_tiles_threshold=ENV.TP_TILES_THRESHOLD,
    debug_mode=False,
    task_name='MoNuSeg',
    server_root='/well/rittscher/projects/GTEx-Liver/GTEx_seg',
    pc_root='D:/LIVER_NASH_dataset/GTEx_seg',
    mac_root='',
    meta_folder_name='', 
    train_folder_name='slides',
    test_folder_name='slides',
    pred_folder_name = 'prediction',
    stat_folder_name = 'statistic',
    model_folder_name = 'seg_models',
    tiles_folder_name = 'tiles',
    seg_batch_size=4,
    seg_num_worker=4,
    seg_num_epoch=10000
    )