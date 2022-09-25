'''
@author: Yang Hu
'''
from env import ENV
from parames import parames_task


ENV_MONUSEG = parames_task(
    project_name=ENV.PROJECT_NAME,
    slide_type=ENV.SLIDE_TYPE,
    scale_factor=ENV.SCALE_FACTOR,
    tile_size=ENV.TILE_H_SIZE,
    tp_tiles_threshold=ENV.TP_TILES_THRESHOLD,
    debug_mode=False,
    task_name='MoNuSeg',
    server_root='/well/rittscher/projects/GTEx-Liver/MoNuSeg',
    pc_root='D:/LIVER_NASH_dataset/MoNuSeg',
    mac_root='',
    meta_folder_name='',
    train_folder_name='train', 
    test_folder_name='train',
    pred_folder_name = 'prediction',
    stat_folder_name = 'statistic',
    model_folder_name = 'seg_models',
    tiles_folder_name = 'tiles',
    seg_batch_size=4,
    seg_num_epoch=4,
    seg_num_worker=20000
    )