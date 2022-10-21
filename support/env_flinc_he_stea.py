'''
@author: Yang Hu
'''

from support.env import ENV
from support.parames import parames_task


ENV_FLINC_HE_STEA = parames_task(
    project_name=ENV.PROJECT_NAME,
    scale_factor=ENV.SCALE_FACTOR,
    tile_size=ENV.TILE_H_SIZE,
    tp_tiles_threshold=ENV.TP_TILES_THRESHOLD,
    pil_image_file_format=ENV.PIL_IMAGE_FILE_FORMAT,
    debug_mode=False,
    task_name='steatosis_score',
    server_root='/data/slides/yang_he_stea',
    pc_root='D:/FLINC_dataset/slides/yang_he_stea',
    mac_root='',
    meta_folder_name='FLINC/meta',
    test_part_prop=0.3,
    fold_suffix='-0',
    loss_package=('wce', [0.6, 0.4]),
    num_att_epoch=100,
    slidemat_batch_size=8,
    slidemat_dataloader_worker=4,
    num_last_eval_epochs=5,
    reset_optim=True,
    num_round=5,
    tile_batch_size=64,
    tile_dataloader_worker=8,
    num_init_s_epoch=10,
    num_inround_s_epoch=5,
    num_inround_t_epoch=3,
    num_inround_rev_t_epoch=2,
    attpool_stop_loss=0.45,
    attpool_stop_maintains=3,
    overall_stop_loss=0.30,
    pos_refersh_pluse=1,
    neg_refersh_pluse=1,
    top_range_rate=0.05,
    sup_range_rate=[0.05, 0.8],
    neg_range_rate=0.2,
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

ENV_FLINC_HE_STEA_C2 = parames_task(
    project_name=ENV.PROJECT_NAME,
    scale_factor=ENV.SCALE_FACTOR,
    tile_size=ENV.TILE_H_SIZE,
    tp_tiles_threshold=ENV.TP_TILES_THRESHOLD,
    pil_image_file_format=ENV.PIL_IMAGE_FILE_FORMAT,
    debug_mode=False,
    task_name='steatosis_score_c2',
    server_root='/data/slides/yang_he_stea',
    pc_root='D:/FLINC_dataset/slides/yang_he_stea',
    mac_root='',
    meta_folder_name='FLINC/meta',
    test_part_prop=0.3,
    fold_suffix='-0',
    loss_package=('wce', [0.6, 0.4]),
    num_att_epoch=300,
    slidemat_batch_size=8,
    slidemat_dataloader_worker=4,
    num_last_eval_epochs=5,
    reset_optim=False,
    num_round=10,
    tile_batch_size=128,
    tile_dataloader_worker=8,
    num_init_s_epoch=5,
    num_inround_s_epoch=20,
    num_inround_t_epoch=1,
    num_inround_rev_t_epoch=1,
    attpool_stop_loss=0.50,
    attpool_stop_maintains=1,
    overall_stop_loss=0.05,
    pos_refersh_pluse=1,
    neg_refersh_pluse=1,
    top_range_rate=0.05,
    sup_range_rate=[0.05, 0.75],
    neg_range_rate=0.25,
    att_k=10,
    sup_k=5,
    reverse_n=5,
    reverse_gradient_alpha=1e-4,
    his_record_rounds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
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
