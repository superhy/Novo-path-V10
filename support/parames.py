'''
@author: Yang Hu
'''
import os
import platform



class parames_basic():
    
    def __init__(self, project_name,
                 slide_type='dx',
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
        self.SLIDE_TYPE = slide_type
        self.SCALE_FACTOR = scale_factor
        self.TILE_H_SIZE = tile_size
        self.TILE_W_SIZE = self.TILE_H_SIZE
        self.TRANSFORMS_RESIZE = self.TILE_H_SIZE
        self.TP_TILES_THRESHOLD = tp_tiles_threshold
        self.DEBUG_MODE = debug_mode

class parames_task(parames_basic):
    
    def __init__(self,
                 project_name,
                 slide_type,
                 scale_factor,
                 tile_size,
                 tp_tiles_threshold,
                 debug_mode,
                 task_name,
                 server_root,
                 pc_root,
                 mac_root,
                 meta_folder_name,
                 seg_train_folder_name,
                 seg_test_folder_name,
                 seg_pred_folder_name,
                 seg_stat_folder_name,
                 model_folder_name,
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
                                           slide_type, 
                                           scale_factor, 
                                           tile_size, 
                                           tp_tiles_threshold, 
                                           debug_mode)
        
        self.TASK_NAME = task_name
        self.STAIN_TYPE = stain_type
        
        self.OS_NAME = platform.system()
        self.SERVER_ROOT = server_root
        self.PC_ROOT = pc_root
        self.MAC_ROOT = mac_root
        self.META_FOLDER = os.path.join(self.PROJECT_DIR, 'data/{}'.format(meta_folder_name))
        
        # segmentation params
        if self.OS_NAME == 'Windows':
            self.SEG_TRAIN_FOLDER_PATH = os.path.join(self.PC_ROOT, seg_train_folder_name)
            self.SEG_TEST_FOLDER_PATH = os.path.join(self.PC_ROOT, seg_test_folder_name)
        elif self.OS_NAME == 'Darwin':
            self.SEG_TRAIN_FOLDER_PATH = os.path.join(self.MAC_ROOT, seg_train_folder_name)
            self.SEG_TEST_FOLDER_PATH = os.path.join(self.MAC_ROOT, seg_test_folder_name)
        else:
            self.SEG_TRAIN_FOLDER_PATH = os.path.join(self.SERVER_ROOT, seg_train_folder_name)
            self.SEG_TEST_FOLDER_PATH = os.path.join(self.SERVER_ROOT, seg_test_folder_name)
        self.SEG_PREDICTION_FOLDER_PATH = os.path.join(self.SEG_TEST_FOLDER_PATH, seg_pred_folder_name)
        self.SEG_STATISTIC_FOLDER_PATH = os.path.join(self.SEG_TEST_FOLDER_PATH, seg_stat_folder_name)
        
        self.SEG_TILES_FOLDER_PATH = os.path.join(self.PC_ROOT, tiles_folder_name) if self.OS_NAME == 'Windows' else os.path.join(self.SERVER_ROOT, tiles_folder_name)
        
        self.SEG_MINI_BATCH = seg_batch_size
        self.SEG_NUM_WORKER = seg_num_worker
        self.SEG_NUM_EPOCH = seg_num_epoch

        # classification params

        # for all case
        self.MODEL_FOLDER_PATH = os.path.join(self.PC_ROOT, model_folder_name) if self.OS_NAME == 'Windows' else os.path.join(self.SERVER_ROOT, model_folder_name)
        
            
        

if __name__ == '__main__':
    pass