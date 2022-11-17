'''
@author: Yang Hu
'''
import os
import platform


class parames_basic():
    
    def __init__(self, project_name,
                 scale_factor=32,
                 tile_size=256,
                 tp_tiles_threshold=70,
                 pil_image_file_format='.png',
                 debug_mode=False):
        """
        Args:
            project_name:, 
            project_dir: use project_name construct the project dir path,
            slide_type: dx or tx, default dx,
            scale_factor: scale ratio when visualization,
            tile_h_size: patch size to separate the whole slide image,
            tile_w_size,
            transforms_resize,
            tp_tiles_threshold,
            pil_image_file_format,
            debug_mode
        """
        
        self.OS_NAME = platform.system()
        self.PROJECT_NAME = project_name
        
        ''' some default dirs '''
        if self.OS_NAME == 'Windows':
            self.PROJECT_DIR = os.path.join('D:/eclipse-workspace', self.PROJECT_NAME)
        elif self.OS_NAME == 'Darwin':
            self.PROJECT_DIR = os.path.join('/Users/superhy/Documents/workspace/', self.PROJECT_NAME)
        else:
            self.PROJECT_DIR = os.path.join('/home/cqj236/workspace', self.PROJECT_NAME)
            
        if self.OS_NAME == 'Windows':
            self.TRANSFER_DIR = 'D:/FLINC_dataset/transfer'
        else:
            self.TRANSFER_DIR = '/data/transfer'
            
#         self.SLIDE_TYPE = slide_type
        self.SCALE_FACTOR = scale_factor
        self.TILE_H_SIZE = tile_size
        self.TILE_W_SIZE = self.TILE_H_SIZE
        self.TRANSFORMS_RESIZE = self.TILE_H_SIZE
        self.TP_TILES_THRESHOLD = tp_tiles_threshold
        self.PIL_IMAGE_FILE_FORMAT = pil_image_file_format
        self.DEBUG_MODE = debug_mode

class parames_task(parames_basic):
    
    def __init__(self,
                 project_name,
                 scale_factor,
                 tile_size,
                 tp_tiles_threshold,
                 pil_image_file_format,
                 debug_mode,
                 task_name,
                 server_root,
                 pc_root,
                 mac_root,
                 meta_folder_name,
                 test_part_prop,
                 fold_suffix,
                 loss_package,
                 num_att_epoch,
                 slidemat_batch_size,
                 slidemat_dataloader_worker,
                 num_last_eval_epochs,
                 reset_optim,
                 num_round,
                 tile_batch_size,
                 tile_dataloader_worker,
                 num_init_s_epoch,
                 num_inround_s_epoch,
                 num_inround_t_epoch,
                 num_inround_rev_t_epoch,
                 attpool_stop_loss,
                 attpool_stop_maintains,
                 overall_stop_loss,
                 pos_refersh_pluse,
                 neg_refersh_pluse,
                 top_range_rate,
                 sup_range_rate,
                 neg_range_rate,
                 att_k,
                 sup_k,
                 reverse_n,
                 reverse_gradient_alpha,
                 his_record_rounds,
                 lr_slide,
                 lr_tile,
                 sspt_num_epoch,
                 sspt_record_pulse,
                 num_slide_samples,
                 vit_shape,
                 seg_train_folder_name,
                 seg_test_folder_name,
                 seg_pred_folder_name,
                 seg_stat_folder_name,
                 tiles_folder_name,
                 stain_type='HE',
                 seg_batch_size=2,
                 seg_num_worker=2,
                 seg_num_epoch=20):
        """
        Args:
            seg_train_folder_name:
            seg_test_folder_name:
        """
        
        super(parames_task, self).__init__(project_name, 
                                           scale_factor, 
                                           tile_size, 
                                           tp_tiles_threshold,
                                           pil_image_file_format,
                                           debug_mode)
        
        self.TASK_NAME = task_name
        self.STAIN_TYPE = stain_type
        
        ''' --- for all case --- '''
        self.OS_NAME = platform.system()
        self.SERVER_ROOT = server_root
        self.PC_ROOT = pc_root
        self.MAC_ROOT = mac_root
        if self.OS_NAME == 'Windows':
            self.DATA_DIR = self.PC_ROOT
        elif self.OS_NAME == 'Darwin':
            self.DATA_DIR = self.MAC_ROOT
        else:
            self.DATA_DIR = self.SERVER_ROOT
        
        self.META_FOLDER = os.path.join(self.PROJECT_DIR, 'data/{}'.format(meta_folder_name))
        # the slide folder should copy the slide from transfer folder specifically for training, isolated from NOVO's transfer folder 
        self.SLIDE_FOLDER = os.path.join(self.DATA_DIR, 'tissues')
        self.MODEL_FOLDER = os.path.join(self.DATA_DIR, 'models')
        
        ''' --- slide process & general file storage params --- '''
        # for unsupervised task, no separation of train/test set, all train set (test set as well)
        self.TEST_PART_PROP = test_part_prop if self.TASK_NAME != 'unsupervised' else 0.0  
        self.EXPERIMENTS_DIR = 'example'
        self.TASK_REPO_DIR = os.path.join(self.DATA_DIR, self.EXPERIMENTS_DIR + '/{}'.format(self.TASK_NAME))
        self.FOLD_SUFFIX = fold_suffix
        self.TILESIZE_DIR = str(self.TILE_H_SIZE) + self.FOLD_SUFFIX # make different folders for multi-fold train/test
        self.TASK_TILE_PKL_TRAIN_DIR = os.path.join(self.DATA_DIR, '{}/{}/{}/train_pkl'.format(self.EXPERIMENTS_DIR,
                                                                                                self.TASK_NAME, 
                                                                                                self.TILESIZE_DIR))
        self.TASK_TILE_PKL_TEST_DIR = os.path.join(self.DATA_DIR, '{}/{}/{}/test_pkl'.format(self.EXPERIMENTS_DIR, 
                                                                                              self.TASK_NAME, 
                                                                                              self.TILESIZE_DIR))
        self.LOG_REPO_DIR = os.path.join(self.PROJECT_DIR, 'data/{}_{}/logs'.format(self.STAIN_TYPE, self.TASK_NAME))
        self.RECORDS_REPO_DIR = os.path.join(self.PROJECT_DIR, 'data/{}_{}/records'.format(self.STAIN_TYPE, self.TASK_NAME))
        self.HEATMAP_STORE_DIR = os.path.join(self.DATA_DIR, 'visualisation/heatmap')
        self.PLOT_STORE_DIR = os.path.join(self.DATA_DIR, 'visualisation/plot')
        self.STATISTIC_STORE_DIR = os.path.join(self.DATA_DIR, 'visualisation/statistic')
        
        ''' --- classification params --- '''
        self.LOSS_PACKAGE = loss_package
        # att mil method
        self.NUM_ATT_EPOCH = num_att_epoch
        self.MINI_BATCH_SLIDEMAT = slidemat_batch_size
        self.SLIDEMAT_DATALOADER_WORKER = slidemat_dataloader_worker
        self.ATTPOOL_RECORD_EPOCHS = [self.NUM_ATT_EPOCH - 1]
        self.NUM_LAST_EVAL_EPOCHS = num_last_eval_epochs
        self.SLIDE_ENCODES_DIR = 'encodes'
        self.TASK_SLIDE_MATRIX_TRAIN_DIR = os.path.join(self.PROJECT_DIR, 'data/{}_{}/{}/{}/train_encode'.format(self.STAIN_TYPE,
                                                                                                                 self.TASK_NAME,
                                                                                                                 self.SLIDE_ENCODES_DIR,
                                                                                                                 self.TILESIZE_DIR))
        self.TASK_SLIDE_MATRIX_TEST_DIR = os.path.join(self.PROJECT_DIR, 'data/{}_{}/{}/{}/test_encode'.format(self.STAIN_TYPE,
                                                                                                               self.TASK_NAME,
                                                                                                               self.SLIDE_ENCODES_DIR,
                                                                                                               self.TILESIZE_DIR))
        # lcsb mil method
        self.RESET_OPTIMIZER = reset_optim
        self.NUM_ROUND = num_round
        self.MINI_BATCH_TILE = tile_batch_size
        self.TILE_DATALOADER_WORKER = tile_dataloader_worker
        self.NUM_INIT_S_EPOCH = num_init_s_epoch
        self.NUM_INROUND_S_EPOCH = num_inround_s_epoch
        self.NUM_INROUND_REV_T_EPOCH = num_inround_rev_t_epoch
        self.NUM_INROUND_T_EPOCH = num_inround_t_epoch
        self.ATTPOOL_STOP_LOSS = attpool_stop_loss
        self.ATTPOOL_STOP_MAINTAINS = attpool_stop_maintains
        self.OVERALL_STOP_LOSS = overall_stop_loss
        self.POS_REFRESH_PULSE = pos_refersh_pluse
        self.NEG_REFRESH_PULSE = neg_refersh_pluse
        self.TOP_RANGE_RATE = top_range_rate # the ratio of all tiles in one slides, for [:TOP_RANGE_RATE]
        self.SUP_RANGE_RATE = sup_range_rate # the ratio of all tiles in one slides, for [SUP_RANGE_RATE[0]: SUP_RANGE_RATE[1]]
        self.NEG_RANGE_RATE = neg_range_rate # the ratio of all tiles in one slides, for [NEG_RANGE_RATE:]
        self.ATT_K = att_k
        self.SUP_K = sup_k
        self.REVERSE_N = reverse_n
        self.REVERSE_GRADIENT_ALPHA = reverse_gradient_alpha
        self.HIS_RECORD_ROUNDS = his_record_rounds
        # learning rate
        self.LR_SLIDE = lr_slide
        self.LR_TILE = lr_tile
        # suit for window test env
        if self.OS_NAME == 'Windows' or self.OS_NAME == 'Darwin':
            self.MINI_BATCH_SLIDEMAT = int(self.MINI_BATCH_SLIDEMAT / 2)
            self.MINI_BATCH_TILE = int(self.MINI_BATCH_TILE / 4)
            self.SLIDEMAT_DATALOADER_WORKER = int(self.SLIDEMAT_DATALOADER_WORKER / 2)
            self.TILE_DATALOADER_WORKER = int(self.TILE_DATALOADER_WORKER / 2)
            
        ''' --- self-supervised encoder pre-train parames --- '''
        self.NUM_ENC_SSPT_EPOCH = sspt_num_epoch # the number of total training epochs for self-supervised pre-train (sspt)
        self.SSPT_RECORD_PULSE = sspt_record_pulse # after every ? epochs, record the training log once
        self.NUM_SLIDE_SAMPLES = num_slide_samples # in each epoch, how many tiles are sampled for sspt from one slide
        self.VIT_SHAPE = vit_shape # the shape of ViT patch map, h / w for (h * w), h == w
        
        ''' --- segmentation params --- '''
        self.SEG_TRAIN_FOLDER_PATH = os.path.join(self.DATA_DIR, seg_train_folder_name)
        self.SEG_TEST_FOLDER_PATH = os.path.join(self.DATA_DIR, seg_test_folder_name)
        self.SEG_PREDICTION_FOLDER_PATH = os.path.join(self.SEG_TEST_FOLDER_PATH, seg_pred_folder_name)
        self.SEG_STATISTIC_FOLDER_PATH = os.path.join(self.SEG_TEST_FOLDER_PATH, seg_stat_folder_name)
        
        self.SEG_TILES_FOLDER_PATH = os.path.join(self.PC_ROOT, tiles_folder_name) if self.OS_NAME == 'Windows' else os.path.join(self.SERVER_ROOT, tiles_folder_name)
        
        self.SEG_MINI_BATCH = seg_batch_size
        self.SEG_NUM_WORKER = seg_num_worker
        self.SEG_NUM_EPOCH = seg_num_epoch
    
    def refresh_fold_suffix(self, new_fold_suffix):
        '''
        refresh the fold_suffix for validation or training on batch
        and change some necessary parameters
        '''
        self.FOLD_SUFFIX = new_fold_suffix
        self.TILESIZE_DIR = str(self.TILE_H_SIZE) + self.FOLD_SUFFIX
        
        self.TASK_TILE_PKL_TRAIN_DIR = os.path.join(self.DATA_DIR, '{}/{}/{}/train_pkl'.format(self.EXPERIMENTS_DIR,
                                                                                               self.TASK_NAME, 
                                                                                               self.TILESIZE_DIR))
        self.TASK_TILE_PKL_TEST_DIR = os.path.join(self.DATA_DIR, '{}/{}/{}/test_pkl'.format(self.EXPERIMENTS_DIR, 
                                                                                             self.TASK_NAME, 
                                                                                             self.TILESIZE_DIR))
        self.TASK_SLIDE_MATRIX_TRAIN_DIR = os.path.join(self.PROJECT_DIR, 'data/{}_{}/{}/{}/train_encode'.format(self.STAIN_TYPE,
                                                                                                                 self.TASK_NAME,
                                                                                                                 self.SLIDE_ENCODES_DIR,
                                                                                                                 self.TILESIZE_DIR))
        self.TASK_SLIDE_MATRIX_TEST_DIR = os.path.join(self.PROJECT_DIR, 'data/{}_{}/{}/{}/test_encode'.format(self.STAIN_TYPE,
                                                                                                               self.TASK_NAME,
                                                                                                               self.SLIDE_ENCODES_DIR,
                                                                                                               self.TILESIZE_DIR))
        
            
        

if __name__ == '__main__':
    pass