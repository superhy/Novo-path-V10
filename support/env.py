'''
@author: Yang Hu
'''
import torch

from support.parames import parames_basic


devices = torch.device('cuda')
devices_cpu = torch.device('cpu')

ENV = parames_basic(
        project_name='Novo-path-V10',
        scale_factor=1,
        tile_size=128,
        tp_tiles_threshold=20,
        pil_image_file_format='.jpeg',
        debug_mode=False
    )

if __name__ == '__main__':
    pass