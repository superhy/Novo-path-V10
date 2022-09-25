'''
@author: Yang Hu
'''
import torch

from parames import parames_basic


devices = torch.device('cuda')
devices_cpu = torch.device('cpu')

ENV = parames_basic(
        project_name='Novo-path-V10',
        slide_type='dx',
        scale_factor=8,
        tile_size=512,
        tp_tiles_threshold=2,
        debug_mode=False
    )

if __name__ == '__main__':
    pass