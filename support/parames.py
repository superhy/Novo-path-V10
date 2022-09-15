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
                 meta_folder_name,
                 train_folder_name,
                 test_folder_name,
                 pred_folder_name,
                 stat_folder_name,
                 model_folder_name,
                 tiles_folder_name,
                 stain_type='HE',
                 batch_size=2,
                 num_worker=2,
                 num_epoch=20):
        """
        Args:
            train_folder_name:
            test_folder_name:
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
        self.META_FOLDER = os.path.join(self.PROJECT_DIR, 'data/{}'.format(meta_folder_name))
        
        if self.OS_NAME == 'Windows':
            self.TRAIN_FOLDER_PATH = os.path.join(self.PC_ROOT, train_folder_name)
            self.TEST_FOLDER_PATH = os.path.join(self.PC_ROOT, test_folder_name)
        else:
            self.TRAIN_FOLDER_PATH = os.path.join(self.SERVER_ROOT, train_folder_name)
            self.TEST_FOLDER_PATH = os.path.join(self.SERVER_ROOT, test_folder_name)
        self.PREDICTION_FOLDER_PATH = os.path.join(self.TEST_FOLDER_PATH, pred_folder_name)
        self.STATISTIC_FOLDER_PATH = os.path.join(self.TEST_FOLDER_PATH, stat_folder_name)
        
        self.MODEL_FOLDER_PATH = os.path.join(self.PC_ROOT, model_folder_name) if self.OS_NAME == 'Windows' else os.path.join(self.SERVER_ROOT, model_folder_name)
        self.TILES_FOLDER_PATH = os.path.join(self.PC_ROOT, tiles_folder_name) if self.OS_NAME == 'Windows' else os.path.join(self.SERVER_ROOT, tiles_folder_name)
        
        self.MINI_BATCH = batch_size
        self.NUM_WORKER = num_worker
        self.NUM_EPOCH = num_epoch
        
            
        

if __name__ == '__main__':
    pass