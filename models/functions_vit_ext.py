'''
@author: superhy
'''

import torch
import numpy as np
from einops.einops import reduce, rearrange

from models import functions
from models.datasets import Simple_Tile_Dataset


def access_encodes_vit(tiles, trained_vit, batch_size, nb_workers):
    '''
    access and extract the encode with trained ViT model
    for a list of tiles
    
    Return:
        tiles_encodes_nd: (t, k) - (tiles_number * encode_dim)
    '''
    trained_vit.eval()
    
    # prepare the tile list dataloader
    transform = functions.get_transform()
    vis_tiles_set = Simple_Tile_Dataset(tiles_list=tiles, transform=transform)
    vis_tiles_dataloader = functions.get_data_loader(dataset=vis_tiles_set,
                                                     batch_size=batch_size,
                                                     num_workers=nb_workers,
                                                     sf=False, p_mem=True)
    ''' 
    prepare a slide-tile location index dict
    {'slide_id-h_id-w_id': (encode_idx, tile, slide_id)}
    '''
    tile_loc_dict = {}
    for i, tile in enumerate(tiles):
        slide_tile_key = '{}-h{}-w{}'.format(tile.query_slideid(),
                                             tile.h_id, tile.w_id)
        tile_loc_dict[slide_tile_key] = (i, tile, tile.query_slideid())
        
    tiles_en_nd = None
    with torch.no_grad():
        for X in vis_tiles_dataloader:
            X = X.cuda()
            e = trained_vit.backbone(X)
            e_nd = e.detach().cpu().numpy()
            # t, k
            if tiles_en_nd is None:
                tiles_en_nd = e_nd
            else:
                tiles_en_nd = np.concatenate((tiles_en_nd, e_nd), axis=0)
    
    return tiles_en_nd, tile_loc_dict

def avg_neigb_encodes(tiles_en_nd, tile_loc_dict, key_encode_tuple):
    '''
    calculate the average tile encodes combination with neighbor location (up, down, left, right)
    
    Args:
        tiles_en_nd:
        tile_loc_dict:
        key_encode_tuple: (encode, tile, slide_id) for the key tile's encode, in middle
    '''
    encode, tile, slide_id = key_encode_tuple
    
    # [t-l, t, t-r, l, r, d-l, d, d-r]
    neigb_keys = ['{}-h{}-w{}'.format(slide_id, tile.h_id - 1, tile.w_id - 1),
                  '{}-h{}-w{}'.format(slide_id, tile.h_id - 1, tile.w_id),
                  '{}-h{}-w{}'.format(slide_id, tile.h_id - 1, tile.w_id + 1),
                  '{}-h{}-w{}'.format(slide_id, tile.h_id, tile.w_id + 1),
                  '{}-h{}-w{}'.format(slide_id, tile.h_id + 1, tile.w_id),
                  '{}-h{}-w{}'.format(slide_id, tile.h_id + 1, tile.w_id - 1),
                  '{}-h{}-w{}'.format(slide_id, tile.h_id + 1, tile.w_id),
                  '{}-h{}-w{}'.format(slide_id, tile.h_id + 1, tile.w_id + 1)]
    
    combine_encodes = [encode]
    for n_k in neigb_keys:
        encode_idx = tile_loc_dict[n_k][0]
        combine_encodes.append(tiles_en_nd[encode_idx])
    combine_encodes_nd = np.array(combine_encodes)
    combine_encodes_nd = np.average(combine_encodes_nd, axis=1)
    
    return combine_encodes_nd
    
    
def avg_dilated_neigb_encodes():
    '''
    '''
    
def access_att_maps_vit(tiles, trained_vit, batch_size, nb_workers, layer_id=-1):
    '''
    access and extract the original signal of attention maps from trained ViT model
    for a list of tiles
    
    Args:
        tiles: a list of tiles, can be from one slide or combined from several slides
        trained_vit: the ViT model
        
    Return:
        tiles_attns_nd:
            Deprecated: >>>>> shape: (t, l, h, q, k) - (tiles_number * layers * heads * (patch_number + 1) * (patch_number + 1))
            shape: (t, h, q, k) - (tiles_number * heads * (patch_number + 1) * (patch_number + 1))
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
            # t, l, h, q, k -> t h q k
            attns_nd = attns_nd[:, layer_id]
            if tiles_attns_nd is None:
                tiles_attns_nd = attns_nd
            else:
                tiles_attns_nd = np.concatenate((tiles_attns_nd, attns_nd), axis=0)
            # print(np.shape(tiles_attns_nd))
    # discard a recorder for the backbone of trained vit        
    trained_vit.discard_wrapper()
    return tiles_attns_nd

def ext_att_maps_pick_layer(tiles_attns_nd, comb_heads='mean'):
    '''
    extract the tiles attention map as numpy ndarray, from specific layer
    
    original features without normalization
    
    Args:
        tiles_attns_nd: the original att maps just was extracted from function <access_att_maps_vit>
        layer_id: which layer you would like to extract the features
        comb_heads: the style of combine the maps for multiple heads
            'max', 'mean', others: do not combine them.
    
    Return:
        l_attns_nd:
            shape: (t h q k) or (t q k), with/without heads combination
    '''
    # l_attns_nd = tiles_attns_nd[:, layer_id]
    l_attns_nd = tiles_attns_nd
    if comb_heads == 'max':
        l_attns_nd = reduce(l_attns_nd, 't h q k -> t q k', reduction='max')
    elif comb_heads == 'mean':
        l_attns_nd = reduce(l_attns_nd, 't h q k -> t q k', reduction='mean')
    else:
        pass
    
    return l_attns_nd
    
def ext_cls_patch_att_maps(l_attns_nd):
    '''
    extract the cls token -> all patches map from layer attention maps
    for specific layer
    
    original features without normalization
    
    Args:
        l_attns_nd: the layer att map just was extracted from function <ext_att_map_pick_layer>
        
    Return:
        cls_atts_nd:
            shape: (t, h, k - 1) - tiles_number * heads * patch_number, if the heads haven't been combined
                (t, k - 1) - tiles_number * patch_number, if the heads were combined, 
    '''
    # detect the order of layer attention map, (t h q k) or (t q k)
    if l_attns_nd.shape == 4:
        cls_atts_nd = l_attns_nd[:, :, 0, 1:]
    else:
        cls_atts_nd = l_attns_nd[:, 0, 1:]
        
    return cls_atts_nd

def ext_patches_adjmat(l_attns_nd):
    '''
    extract the adjacency matrix to describe the relevant between patches, from layer attention maps
    for specific layer
    
    original features without normalization
    
    Args:
    
    Return:
        
    '''
    

def access_full_embeds_vit(tiles, trained_vit, batch_size, nb_workers):
    '''
    access and extract the original signal of embeddings (for all heads) from trained ViT model
    for a list of tiles
    
    Args:
        tiles: a list of tiles, can be from one slide or combined from several slides
        trained_vit: the ViT model
        layer_id: which layer you would like to extract the features
        
    Return:
        tiles_embeds_nd:
            shape: (t, q, d) - (tiles_number * (patch_number + 1) * dim)
            with one extra patch due to the CLS token
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