'''
@author: Yang Hu
'''
import torch

from support.parames import parames_basic


devices = torch.device('cuda')
devices_cpu = torch.device('cpu')

ENV = parames_basic(
        project_name='Novo-path-V10',
        scale_factor=64,
        tile_size=256,
        tp_tiles_threshold=50,
        pil_image_file_format='.png',
        debug_mode=False
    )

if __name__ == '__main__':
    pass