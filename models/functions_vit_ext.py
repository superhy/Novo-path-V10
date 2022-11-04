'''
@author: superhy
'''

import torch
import numpy as np

from models import functions
from models.datasets import Simple_Tile_Dataset


def access_att_maps_vit(tiles, trained_vit, batch_size, nb_workers, layer_id=-1):
    '''
    access and extract the original signal of attention maps from trained ViT model
    for a list of tiles
    
    Args:
        tiles: a list of tiles, can be from one slide or combined from several slides
        trained_vit: the ViT model
        layer_id: which layer you would like to extract the features
        
    Return:
        tiles_attns_nd: 
            with one extra patch due to the CLS token
    '''
    trained_vit.eval()
    # deploy a recorder for the backbone of trained vit
    trained_vit.deploy_recorder()
    
    # prepare the tile list dataloader
    transform = functions.get_transform()
    vis_tiles_set = Simple_Tile_Dataset(tiles_list=tiles, transform=transform)
    vis_tiles_dataloader = functions.get_data_loader(dataset=vis_tiles_set,
                                                     batch_size=batch_size,
                                                     num_workers=nb_workers,
                                                     sf=False, p_mem=True)
    
    tiles_attns_nd = None
    with torch.no_grad():
        for X in vis_tiles_dataloader:
            X = X.cuda()
            _, attns = trained_vit.backbone(X)
            attns_nd = attns.detach().cpu().numpy()
            if tiles_attns_nd is None:
                tiles_attns_nd = attns_nd
            else:
                tiles_attns_nd = np.concatenate((tiles_attns_nd, attns_nd), axis=0)
    # discard a recorder for the backbone of trained vit        
    trained_vit.discard_wrapper()
    
    return tiles_attns_nd


def access_full_embeds_vit(tiles, trained_vit, batch_size, nb_workers, layer_id=-1):
    '''
    access and extract the original signal of embeddings (for all heads) from trained ViT model
    for a list of tiles
    
    Args:
        tiles: a list of tiles, can be from one slide or combined from several slides
        trained_vit: the ViT model
        layer_id: which layer you would like to extract the features
        
    Return:
        tiles_embeds_nd: 
    '''
    
    trained_vit.eval()
    # deploy a extractor for the backbone of trained vit
    trained_vit.deploy_extractor()
    
    # prepare the tile list dataloader
    transform = functions.get_transform()
    vis_tiles_set = Simple_Tile_Dataset(tiles_list=tiles, transform=transform)
    vis_tiles_dataloader = functions.get_data_loader(dataset=vis_tiles_set,
                                                     batch_size=batch_size,
                                                     num_workers=nb_workers,
                                                     sf=False, p_mem=True)
    
    tiles_embeds_nd = None
    with torch.no_grad():
        for X in vis_tiles_dataloader:
            X = X.cuda()
            _, embeds = trained_vit.backbone(X)
            embeds_nd = embeds.detach().cpu().numpy()
            if tiles_embeds_nd is None:
                tiles_embeds_nd = embeds_nd
            else:
                tiles_embeds_nd = np.concatenate((tiles_embeds_nd, embeds_nd), axis=0)
    # discard a extractor for the backbone of trained vit        
    trained_vit.discard_wrapper()
    
    return tiles_embeds_nd

if __name__ == '__main__':
    pass