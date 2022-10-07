'''
@author: Yang Hu
'''

from support.env import ENV
from support.parames import parames_task


ENV_FLINC_HE_FIB = parames_task(
    project_name=ENV.PROJECT_NAME,
    scale_factor=ENV.SCALE_FACTOR,
    tile_size=ENV.TILE_H_SIZE,
    tp_tiles_threshold=ENV.TP_TILES_THRESHOLD,
    pil_image_file_format=ENV.PIL_IMAGE_FILE_FORMAT,
    debug_mode=False,
    task_name='fibrosis_score',
    server_root='',
    pc_root='D:/FLINC_dataset/',
    mac_root='',
    meta_folder_name='FLINC/meta',
    test_part_prop=0.3,
    fold_suffix='-0',
    loss_package=('ce'),
    num_att_epoch=80,
    slidemat_batch_size=8,
    slidemat_dataloader_worker=4,
    num_last_eval_epochs=5,
    reset_optim=True,
    num_round=5,
    tile_batch_size=64,
    tile_dataloader_worker=8,
    num_init_s_epoch=10,
    num_inround_s_epoch=5,
    num_inround_t_epoch=4,
    num_inround_rev_t_epoch=2,
    attpool_stop_loss=0.60,
    attpool_stop_maintains=3,
    overall_stop_loss=0.50,
    pos_refersh_pluse=1,
    neg_refersh_pluse=1,
    att_k=100,
    sup_k=50,
    reverse_n=20,
    reverse_gradient_alpha=1e-4,
    his_record_rounds=[0, 1, 2, 3, 4],
    lr_slide=1e-4,
    lr_tile=1e-4,
    seg_train_folder_name='train', 
    seg_test_folder_name='train',
    seg_pred_folder_name = 'prediction',
    seg_stat_folder_name = 'statistic',
    tiles_folder_name = 'tiles',
    stain_type='HE',
    seg_batch_size=4,
    seg_num_worker=4,
    seg_num_epoch=20000
    )
