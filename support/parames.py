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
                 debug_mode=False):
        """
        Args:
            project_name:, 
            project_dir: use project_name construct the project dir path,
            slide_type: dx or tx, default dx,
            apply_tumor_roi: default False,
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
        if self.OS_NAME == 'Windows':
            self.PROJECT_DIR = os.path.join('D:/workspace', self.PROJECT_NAME)
        elif self.OS_NAME == 'Darwin':
            self.PROJECT_DIR = os.path.join('/Users/superhy/Documents/workspace/', self.PROJECT_NAME)
        else:
            self.PROJECT_DIR = os.path.join('/well/rittscher/users/lec468/workspace', self.PROJECT_NAME)
#         self.SLIDE_TYPE = slide_type
        self.SCALE_FACTOR = scale_factor
        self.TILE_H_SIZE = tile_size
        self.TILE_W_SIZE = self.TILE_H_SIZE
        self.TRANSFORMS_RESIZE = self.TILE_H_SIZE
        self.TP_TILES_THRESHOLD = tp_tiles_threshold
        self.DEBUG_MODE = debug_mode

class parames_task(parames_basic):
    
    def __init__(self,
                 project_name,
                 scale_factor,
                 tile_size,
                 tp_tiles_threshold,
                 debug_mode,
                 task_name,
                 server_root,
                 pc_root,
                 mac_root,
                 meta_folder_name,
                 slide_folder_name,
                 test_part_prop,
                 fold_suffix,
                 loss_package,
                 num_att_epoch,
                 mini_batch_slidemat,
                 slidemat_dataloader_worker,
                 num_last_eval_epochs,
                 reset_optim,
                 num_round,
                 mini_batch_tile,
                 lr_slide,
                 lr_tile,
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
        self.SLIDE_FOLDER = os.path.join(self.DATA_DIR, 'slide/{}'.format(slide_folder_name))
        self.MODEL_FOLDER = os.path.join(self.DATA_DIR, 'models')
        
        ''' --- slide process & general file storage params --- '''
        self.TEST_PART_PROP = test_part_prop
        self.EXPERIMENTS_DIR = 'example'
        self.TASK_REPO_DIR = os.path.join(self.DATA_DIR, self.EXPERIMENTS_DIR + '/{}'.format(self.TASK_NAME))
        self.FOLD_SUFFIX = fold_suffix
        self.TILESIZE_DIR = str(self.TILE_H_SIZE) + self.FOLD_SUFFIX # make different folders for multi-fold train/test
        self.TASK_TILE_PKL_TRAIN_DIR = os.path.join(self.DATA_DIR, '{}/{}/{}/train_tile'.format(self.EXPERIMENTS_DIR,
                                                                                                self.TASK_NAME, 
                                                                                                self.TILESIZE_DIR))
        self.TASK_TILE_PKL_TEST_DIR = os.path.join(self.DATA_DIR, '{}/{}/{}/test_tile'.format(self.EXPERIMENTS_DIR, 
                                                                                              self.TASK_NAME, 
                                                                                              self.TILESIZE_DIR))
        self.LOG_REPO_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/logs'.format(self.TASK_NAME))
        self.RECORDS_REPO_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/records'.format(self.TASK_NAME))
        self.HEATMAP_STORE_DIR = os.path.join(self.DATA_DIR, 'visualization/heatmap')
        self.PLOT_STORE_DIR = os.path.join(self.DATA_DIR, 'visualization/plot')
        self.STATISTIC_STORE_DIR = os.path.join(self.DATA_DIR, 'visualization/statistic')
        
        ''' --- classification params --- '''
        self.LOSS_PACKAGE = loss_package
        # att mil method
        self.NUM_ATT_EPOCH = num_att_epoch
        self.MINI_BATCH_SLIDEMAT = mini_batch_slidemat
        self.SLIDEMAT_DATALOADER_WORKER = slidemat_dataloader_worker
        self.ATTPOOL_RECORD_EPOCHS = [self.NUM_ATT_EPOCH - 1]
        self.NUM_LAST_EVAL_EPOCHS = num_last_eval_epochs
        self.SLIDE_ENCODES_DIR = 'encodes'
        self.TASK_PROJECT_TRAIN_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/{}/{}/train_tile'.format(self.TASK_NAME,
                                                                                                       self.SLIDE_ENCODES_DIR,
                                                                                                       self.TILESIZE_DIR))
        self.TASK_SLIDE_MATRIX_TRAIN_DIR = self.TASK_PROJECT_TRAIN_DIR.replace('train_tile', 'train_encode')
        self.TASK_PROJECT_TEST_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/{}/{}/test_tile'.format(self.TASK_NAME,
                                                                                                     self.SLIDE_ENCODES_DIR,
                                                                                                     self.TILESIZE_DIR))
        self.TASK_SLIDE_MATRIX_TEST_DIR = self.TASK_PROJECT_TEST_DIR.replace('test_tile', 'test_encode')
        # lcsb mil method
        self.RESET_OPTIMIZER = reset_optim
        self.NUM_ROUND = num_round
        self.MINI_BATCH_TILE = mini_batch_tile
        # learning rate
        self.LR_SLIDE = lr_slide
        self.LR_TILE = lr_tile
        # TODO:
        
        ''' --- segmentation params --- '''
        self.SEG_TRAIN_FOLDER_PATH = os.path.join(self.DATA_DIR, seg_train_folder_name)
        self.SEG_TEST_FOLDER_PATH = os.path.join(self.DATA_DIR, seg_test_folder_name)
        self.SEG_PREDICTION_FOLDER_PATH = os.path.join(self.SEG_TEST_FOLDER_PATH, seg_pred_folder_name)
        self.SEG_STATISTIC_FOLDER_PATH = os.path.join(self.SEG_TEST_FOLDER_PATH, seg_stat_folder_name)
        
        self.SEG_TILES_FOLDER_PATH = os.path.join(self.PC_ROOT, tiles_folder_name) if self.OS_NAME == 'Windows' else os.path.join(self.SERVER_ROOT, tiles_folder_name)
        
        self.SEG_MINI_BATCH = seg_batch_size
        self.SEG_NUM_WORKER = seg_num_worker
        self.SEG_NUM_EPOCH = seg_num_epoch
        
            
        

if __name__ == '__main__':
    pass