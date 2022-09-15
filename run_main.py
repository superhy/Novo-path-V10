'''
@author: Yang Hu
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import sys

from models import functions_monu_seg, functions_gtex_seg
from wsi.process import _run_gtexseg_slide_tiles_split
from support import env_monuseg, env_gtex_seg
from support import tools



class Logger(object):

    def __init__(self, filename='running_log-default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    log_name = 'running_log-{}-{}.log'.format(env_monuseg.ENV_MONUSEG.TASK_NAME, str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(log_name)
    
    ''' something training '''
#     functions_ukaih_seg._run_seg_train_unet(env_ukaih_fat.ENV_UKAIH_FAT)
#     functions_monu_seg._run_seg_train_unet(env_monuseg.ENV_MONUSEG)
    
    ''' something testing/prediction'''
#     functions_monu_seg._run_segmentation_slides_unet(env_monuseg.ENV_MONUSEG,
#                                                      os.path.join(env_monuseg.ENV_MONUSEG.MODEL_FOLDER_PATH, 'checkpoint_UNet-MoNuSeg-800-2022-05-15.pth'))
    
#     _run_gtexseg_slide_tiles_split(env_gtex_seg.ENV_GTEX_SEG)
    
    functions_gtex_seg._run_segmentation_slides_unet(env_gtex_seg.ENV_GTEX_SEG,
                                                     os.path.join(env_monuseg.ENV_MONUSEG.MODEL_FOLDER_PATH, 'checkpoint_UNet-MoNuSeg-1000-2022-05-15.pth'))
    
    