'''
Created on 25 Oct 2022

@author: Yang Hu
'''
from support.env import ENV
from support.parames import parames_task


ENV_FLINC_CD45_U = parames_task(
    project_name=ENV.PROJECT_NAME,
    scale_factor=ENV.SCALE_FACTOR,
    tile_size=ENV.TILE_H_SIZE,
    tp_tiles_threshold=ENV.TP_TILES_THRESHOLD,
    pil_image_file_format=ENV.PIL_IMAGE_FILE_FORMAT,
    debug_mode=True, # for unsupervised task, only use train set to generate the feature space
    task_name='unsupervised',
    server_root='/data/slides/yang_cd45_u',
    pc_root='D:/FLINC_dataset/slides/yang_cd45_u',
    mac_root='',
    meta_folder_name='FLINC/meta',
    test_part_prop=0.0, # for unsupervised task, no separation of train/test set, all train set (test set as well) 
    fold_suffix='-0',
    loss_package=('wce', [0.6, 0.4]),
    num_att_epoch=100,
    slidemat_batch_size=4,
    slidemat_dataloader_worker=4,
    num_last_eval_epochs=5,
    reset_optim=False,
    num_round=5,
    tile_batch_size=16,
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
    sup_range_rate=[0.05, 0.5],
    neg_range_rate=0.5,
    att_k=5,
    sup_k=2,
    reverse_n=1,
    reverse_gradient_alpha=1e-4,
    his_record_rounds=[0, 1, 2, 3, 4],
    lr_slide=1e-4,
    lr_tile=1e-5,
    sspt_num_epoch=500,
    sspt_record_pulse=50,
    num_slide_samples=200,
    vit_shape=32,
    seg_train_folder_name='train', 
    seg_test_folder_name='train',
    seg_pred_folder_name = 'prediction',
    seg_stat_folder_name = 'statistic',
    tiles_folder_name = 'tiles',
    stain_type='CD45',
    seg_batch_size=4,
    seg_num_worker=4,
    seg_num_epoch=20000
    )