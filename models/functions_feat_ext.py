'''
@author: superhy
'''

import math
import warnings

import einops
import torch
from tqdm import tqdm

from models import functions
from models.datasets import Simple_Tile_Dataset
import numpy as np
from support.tools import normalization, normalization_sk


def access_encodes_imgs(tiles, trained_encoder, batch_size, nb_workers):
    '''
    access and extract the encode with trained ViT model or ResNet model
    for a list of tiles
    
    Return:
        tiles_encodes_nd: (t, k) - (tiles_number * encode_dim)
    '''
    trained_encoder.eval()
    
    # prepare the tile list dataloader
    transform = functions.get_zoom_transform()
    tiles_set = Simple_Tile_Dataset(tiles_list=tiles, transform=transform)
    tiles_dataloader = functions.get_data_loader(dataset=tiles_set,
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
        for X in tqdm(tiles_dataloader, desc='Generating tiles\' embedding', leave=True):
            X = X.cuda()
            e = trained_encoder.backbone(X)
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
        if n_k in tile_loc_dict.keys():
            encode_idx = tile_loc_dict[n_k][0]
            combine_encodes.append(tiles_en_nd[encode_idx])
    combine_encodes_nd = np.array(combine_encodes)
    combine_encodes_nd = np.average(combine_encodes_nd, axis=0)
    
    return combine_encodes_nd
    
    
def avg_dilated_neigb_encodes():
    '''
    TODO:
    '''
    
    
''' 
functions family for combined encoding the tile's feature and its regional context's semantic 
'''
    
def make_neighb_coords(radius, tile):
    '''
    calculate the coordinates on the image corresponding to each point on the square
    '''
    h, w = tile.h_id, tile.w_id
    
    left = max(0, w - radius)
    right = w + radius
    top = max(0, h - radius)
    bottom = h + radius

    # calculate the coordinates on the image corresponding to each point on the square
    coordinates = [(i, j) for i in range(top, bottom) for j in range(left, right)]
    
    return coordinates, left, right, top, bottom, h, w

def gen_ctx_grid_tensor(radius, tiles_en_nd, tile_loc_dict, key_encode_tuple, print_info):
    '''
    '''
    encode, tile, slide_id = key_encode_tuple
    
    dim = encode.shape[-1]
    # create a empty region_ctx_nd
    C = radius * 2 + 1
    region_ctx_nd = np.zeros((C, C, dim))

    # calculate the coordinates on the image corresponding to each point on the square
    coordinates, left, right, top, bottom, h, w = make_neighb_coords(radius, tile)
    
    # fill the region context region_ctx_nd
    for i, coord in enumerate(coordinates):
        q_h, q_w = coord
        grid_h, grid_w = q_h - top, q_w - left, 
        if q_h == h and q_w == w:
            region_ctx_nd[grid_h, grid_w] = encode # encode of key tile
            continue
        
        ctx_keys = '{}-h{}-w{}'.format(slide_id, q_h, q_w)
        if ctx_keys in tile_loc_dict.keys():
            ctx_en_idx = tile_loc_dict[ctx_keys][0]
            ctx_en = tiles_en_nd[ctx_en_idx]
            region_ctx_nd[grid_h, grid_w] = ctx_en
    region_ctx_nd = np.transpose(region_ctx_nd, (2, 0, 1))
    
    if print_info:
        print('load region ctx for tile', slide_id, h, w, 'region_ctx_nd:', region_ctx_nd.shape)
            
    return region_ctx_nd, coordinates
    
def encode_region_ctx_prior(region_ctx_nd, tile_en_nd, vit_region, comb_layer, 
                            radius, ctx_type, print_info):
    '''
    Args:
        ctx_type: 
            'reg' -> only region context
            'ass' - > only associations to key tile
            'reg_ass' -> both region context & associations to key tile
    '''
    vit_region.eval()
    comb_layer.eval()
    
    ctx_tensor = torch.from_numpy(region_ctx_nd).to(torch.float)
    ctx_tensor = ctx_tensor.cuda()
    ctx_tensor = torch.unsqueeze(ctx_tensor, 0) # (h, w) -> (1, h, w)
    if vit_region.with_wrapper is False:
        vit_region.deploy_recorder()
    en_ctx, attn_ctx = vit_region.backbone(ctx_tensor)
    en_ctx = torch.squeeze(en_ctx, 0) # back to (1, d) -> (d)
    if ctx_type == 'ass':
        attn_ctx = extra_reg_assoc_key_tile(attn_ctx, radius)
        en_ctx = attn_ctx
    elif ctx_type == 'reg_ass':
        attn_ctx = extra_reg_assoc_key_tile(attn_ctx, radius)
        en_ctx = torch.cat((en_ctx, attn_ctx), -1)
    
    en_t = torch.from_numpy(tile_en_nd)
    en_t = en_t.cuda()
    if print_info:
        print('combine region prior and tile:', en_ctx.shape, en_t.shape, '...')

    ctx_prior_e = comb_layer(en_ctx, en_t)
    t_ctx_e_nd = ctx_prior_e.detach().cpu().numpy()
    return t_ctx_e_nd

def extra_reg_assoc_key_tile(attn_ctx, radius):
    '''
    TODO (for future): set a dimensionality reduction via PCA to reduce the ass_featrue dim
    '''
    # attn_ctx: (batch, layers, heads, patch, patch)
    attn_ctx = attn_ctx.cpu().detach().numpy()
    
    heads_attn_map = attn_ctx[0, -1, :, 1:, 1:] # b l h p1 p2 -> h p1 p2
    attn_map = einops.reduce(heads_attn_map, 'h p1 p2 -> p1 p2', 'mean')
    key_i = (radius * 2 + 1) * radius + radius # centre
    att_vec_1 = einops.rearrange(attn_map[key_i, :], '... -> ...')
    att_vec_2 = einops.rearrange(attn_map[:, key_i], '... -> ...')
    att_vec = (att_vec_1 + att_vec_2) / 2.0
    
    return torch.from_numpy(att_vec).cuda()
    
def comput_region_ctx_comb_encodes(reg_radius, tiles_en_nd, tile_loc_dict, key_encode_tuple,
                                   vit_region, comb_layer, ctx_type, print_info=False):
    '''
    Args:
        reg_radius:
        tiles_en_nd:
        tile_loc_dict:
        key_encode_tuple: (encode, tile, slide_id) for the key tile's encode, in middle
        vit_region:
        comb_layer:
        ctx_type:
    '''
    encode, _, _ = key_encode_tuple
    region_ctx_nd, coordinates = gen_ctx_grid_tensor(reg_radius, tiles_en_nd, tile_loc_dict, 
                                                     key_encode_tuple, print_info)
    tile_region_ctx_encode_nd = encode_region_ctx_prior(region_ctx_nd, encode, vit_region, comb_layer,
                                                        reg_radius, ctx_type, print_info)
    
    return tile_region_ctx_encode_nd

    
''' 
------------- access functions for pick up some visualization information ------------- 
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
    transform = functions.get_zoom_transform()
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
    fuse the attention values on various heads
    
    original features without normalization
    
    Args:
        tiles_attns_nd: the original att maps just was extracted from function <access_att_maps_vit>
        layer_id: which layer you would like to extract the features
        comb_heads: the style of combine the maps for multiple heads
            'max', 'mean', others: do not combine them.
    
    Return:
        l_attns_nd:
            shape: (t q k) or (t h q k), with/without heads combination
    '''
    # l_attns_nd = tiles_attns_nd[:, layer_id]
    l_attns_nd = tiles_attns_nd
    # fuse the attention values on various heads
    if comb_heads == 'max':
        l_attns_nd = einops.reduce(l_attns_nd, 't h q k -> t q k', reduction='max')
    elif comb_heads == 'mean':
        l_attns_nd = einops.reduce(l_attns_nd, 't h q k -> t q k', reduction='mean')
    else:
        pass
    
    return l_attns_nd
    
def ext_cls_patch_att_maps(l_attns_nd):
    '''
    extract the cls token -> all patches map from layer attention maps
    for specific layer
    
    original features without normalization
    
    Args:
        l_attns_nd: the layer att map just was extracted from function <ext_att_maps_pick_layer>
        
    Return:
        cls_atts_nd:
            shape: (t, h, k - 1) - tiles_number * heads * patch_number, if the heads haven't been combined
                (t, k - 1) - tiles_number * patch_number, if the heads were combined, 
    '''
    nb_row = int(math.sqrt(l_attns_nd.shape[-1] - 1) ) # row = column = sqrt(k - 1)
    # detect the order of layer attention map, (t h q k) or (t q k)
    if len(l_attns_nd.shape) == 4:
        cls_atts_nd = l_attns_nd[:, :, 0, 1:]
        cls_atts_nd = einops.rearrange(cls_atts_nd, 't h (r c) -> t h r c', r=nb_row)
    else:
        cls_atts_nd = l_attns_nd[:, 0, 1:]
        cls_atts_nd = einops.rearrange(cls_atts_nd, 't (r c) -> t r c', r=nb_row)
        
    return cls_atts_nd

''' ----------- visualization accessing for regional context (around key tiles) ----------- '''

def reg_ass_key_tile(radius, key_encode_tuple, tiles_en_nd, tile_loc_dict, vit_region,
                     centre_ass=False):
    '''
    access the regional context association based on the attention 
    '''
    region_ctx_nd, coordinates = gen_ctx_grid_tensor(radius, tiles_en_nd, tile_loc_dict, key_encode_tuple, print_info=True)
    vit_region.eval()
    
    ctx_tensor = torch.from_numpy(region_ctx_nd).to(torch.float)
    ctx_tensor = ctx_tensor.cuda()
    ctx_tensor = torch.unsqueeze(ctx_tensor, 0) # (h, w) -> (1, h, w)
    if vit_region.with_wrapper is False:
        vit_region.deploy_recorder()
    en_ctx, attn_ctx = vit_region.backbone(ctx_tensor)
    if centre_ass is True:
        ''' !!! did [1:, 1:] at below function '''
        ass_vec = extra_reg_assoc_key_tile(attn_ctx, radius)
        ass_vec = ass_vec.cpu().detach().numpy()
        ass_mat = vector_to_matrix(ass_vec, radius)
    else:
        attn_ctx = attn_ctx.cpu().detach().numpy()
        ''' !!! did [1:, 1:] at here, but without any normalization '''
        heads_attn_map = attn_ctx[0, -1, :, 1:, 1:] # b l h p1 p2 -> h p1 p2
        ass_mat = einops.reduce(heads_attn_map, 'h p1 p2 -> p1 p2', 'mean')
    
    return ass_mat

def vector_to_matrix(vector, r):
    attention_matrix = np.zeros((2*r + 1, 2*r + 1))
    center = r

    idx = 0
    for i in range(2*r + 1):
        for j in range(2*r + 1):
            if i == center and j == center:
                continue
            attention_matrix[i][j] = vector[idx]
            idx += 1

    return attention_matrix


''' --------------------- graph extraction functions ---------------------- '''

def ext_patches_adjmats(l_attns_nd):
    '''
    extract the adjacency matrix to describe the associations between patches, from layer attention maps
    for specific layer
    
    original features without normalization
    
    Args:
        l_attns_nd: the layer att map just was extracted from function <ext_att_maps_pick_layer>
        
    Return:
        adj_mats_nd:
            shape: (t, h, [k - 1 x k - 1]) - tiles_number * heads * patch_number * patch_number, without fusion of heads
                (t, [k - 1 x k - 1]) - tiles_number * patch_number * patch_number, with fusion of heads
    '''
    # detect the order of layer attention map, (t h q k) or (t q k)
    if len(l_attns_nd.shape) == 4:
        # the original attention outcomes are just the adjacency map with (q x k), q = k = number of all patches
        adj_mats_nd = l_attns_nd[:, :, 1:, 1:]
    else:
        adj_mats_nd = l_attns_nd[:, 1:, 1:]
        
    return adj_mats_nd

def norm_exted_maps(maps_nd, in_pattern):
    '''
    normalize the extracted maps, in 4 input patterns
    
    Args:
        maps_nd: the input tensor, usually we expect that its last dimension is flatten for all attention values we care about
            if not, we need to flatten it first
        in_pattern: can only be: 1. 't h v' (tiles, heads, values), 2. 't h q k' (tiles, heads, patches, patches),
                                 3. 't v' (tiles, values), 4. 't q k' (tiles, patches, patches).
    '''
    if in_pattern not in ['t h v', 't h q k', 't v', 't q k', 'q k']:
        warnings.warn('!!! Sorry, the input pattern statement is wrong, so cannot conduct normalization and return the ORG tensor.')
        return maps_nd
    
    if in_pattern == 't h v':
        (t, h, v) = maps_nd.shape
        norm_nd = []
        for i in range(t):
            norm_nd.append([normalization(maps_nd[i, j, :]) for j in range(h)])
        norm_nd = np.array(norm_nd)
    elif in_pattern == 't h q k':
        (t, h, q, k) = maps_nd.shape
        maps_nd = einops.rearrange(maps_nd, 't h q k -> t h (q k)')
        norm_nd = []
        for i in range(t):
            norm_nd.append([normalization(maps_nd[i, j, :]) for j in range(h)])
        norm_nd = np.array(norm_nd)
        norm_nd = einops.rearrange(norm_nd, 't h (a b) -> t h a b', a=q)
    elif in_pattern == 't v':
        (t, v) = maps_nd.shape
        norm_nd = np.array([normalization(maps_nd[i, :]) for i in range(t)])
    elif in_pattern == 'q k':
        (q, k) = maps_nd.shape
        maps_nd = einops.rearrange(maps_nd, 'q k -> (q k)')
        norm_nd = normalization(maps_nd)
        norm_nd = einops.rearrange(norm_nd, '(a b) -> a b', a=q)
    else:
        (t, q, k) = maps_nd.shape
        maps_nd = einops.rearrange(maps_nd, 't q k -> t (q k)')
        norm_nd = np.array([normalization(maps_nd[i, :]) for i in range(t)])
        norm_nd = einops.rearrange(norm_nd, 't (a b) -> t a b', a=q)
        
    return norm_nd

def norm_sk_exted_maps(maps_nd, in_pattern, amplify=1, mode='max'):
    '''
    >>> deprecated for the moment <<<
    normalize the extracted maps, only in 2 input patterns
    with sklearn method, has more mode: 'l1' and 'l2' normalization
    
    Args:
        maps_nd: the input tensor, usually we expect that its last dimension is flatten for all attention values we care about
            if not, we need to flatten it first
        in_pattern: can only be: 1. 't v' (tiles, values), 2. 't q k' (tiles, patches, patches).
    '''
    if in_pattern not in ['t v', 't q k']:
        warnings.warn('!!! Sorry, the input pattern statement is wrong, so cannot conduct normalization and return the ORG tensor.')
        return maps_nd
    
    if in_pattern == 't v':
        (t, v) = maps_nd.shape
        norm_nd = normalization_sk(maps_nd * amplify, mode)
    else:
        (t, q, k) = maps_nd.shape
        maps_nd = einops.rearrange(maps_nd, 't q k -> t (q k)')
        norm_nd = normalization_sk(maps_nd*amplify, mode)
        norm_nd = einops.rearrange(norm_nd, 't (a b) -> t a b', a=q)
        
    return norm_nd
    
def symm_adjmats(adjmats_nd, rm_selfloop=True):
    '''
    symmetrize adjacency matrix, make mat[i, j] == mat[j, i]
    
    Args:
        adjmats_nd:
        rm_selfloop: indicate to if remove the self-loops, make mat[i, i] == 0
    '''
    if len(adjmats_nd.shape) == 4:
        (t, h, q, k) = adjmats_nd.shape
        symmats_nd = np.zeros((t, h, q, k), dtype='float16')
        for i in range(q):
            for j in range(k):
                symmats_nd[:, :, i, j] = (adjmats_nd[:, :, i, j] + adjmats_nd[:, :, j, i]) / 2.0
                if rm_selfloop and i == j:
                    symmats_nd[:, :, i, j] = .0
    elif len(adjmats_nd.shape) == 3:
        (t, q, k) = adjmats_nd.shape
        symmats_nd = np.zeros((t, q, k), dtype='float16')
        # print(symmats_nd)
        for i in range(q):
            for j in range(k):
                symmats_nd[:, i, j] = (adjmats_nd[:, i, j] + adjmats_nd[:, j, i]) / 2.0
                if rm_selfloop and i == j:
                    symmats_nd[:, i, j] = .0
    else:
        # for only one tile's ass or attn -> (q, k)
        (q, k) = adjmats_nd.shape
        symmats_nd = np.zeros((q, k), dtype='float16')
        for i in range(q):
            for j in range(k):
                symmats_nd[i, j] = (adjmats_nd[i, j] + adjmats_nd[j, i]) / 2.0
                if rm_selfloop and i == j:
                    symmats_nd[i, j] = .0
           
    return symmats_nd     

def gen_edge_adjmats(adjmats_nd, one_hot=True, b_edge_threshold=0.5):
    '''
    generate edges from adjacency matrix, by edge threshold
    
    Args:
        adjmats_nd:
        one_hot: if outcome the one-hot (0-1) matrices
        b_edge_threshold: the threshold to keep the edge
    '''
    adjmats_nd[adjmats_nd < b_edge_threshold] = .0
    if one_hot is True:
        adjmats_nd[adjmats_nd >= b_edge_threshold] = 1
        adjmats_nd.astype('int32')
        
    return adjmats_nd

def node_pos_t_adjmat(t_adjmat):
    '''
    for single tile
    calculate the nodes' positions in x-y axis
    '''
    dtype = t_adjmat.dtype.name
    # record the position according to original grid map
    (q, k) = t_adjmat.shape
    s = int(math.sqrt(q)) # size of the grid map
    n_id, id_pos_dict = 0, {}
    for i in range(s):
        for j in range(s):
            id_pos_dict[n_id] = (j, s-1-i)
            n_id += 1
            
    return id_pos_dict

def filter_node_pos_t_adjmat(t_adjmat):
    '''
    for single tile
    from adjacency matrix, remove the nodes without connection with other nodes
    also record the positions of left nodes
    
    Args:
        t_adjmat: adjacency matrix of one tile, shape: (q, k), q == k
            the input adj matrix is after symmetrize or one-hot process
    '''
    dtype = t_adjmat.dtype.name
    # record the position according to original grid map
    (q, k) = t_adjmat.shape
    s = int(math.sqrt(q)) # size of the grid map
    n_id, id_pos_dict = 0, {}
    for i in range(s):
        for j in range(s):
            id_pos_dict[n_id] = (j, s-1-i)
            n_id += 1
    
    new_n_id, new_org_nid_dict = 0, {}
    for i in range(q):
        if np.sum(t_adjmat[i]) > 0:
            new_org_nid_dict[new_n_id] = i
            new_n_id += 1
    f_t_adjmat, f_id_pos_dict = np.zeros((new_n_id, new_n_id), dtype=dtype), {}
    for i in range(new_n_id):
        f_id_pos_dict[i] = id_pos_dict[new_org_nid_dict[i]]
        for j in range(new_n_id):
            f_t_adjmat[i, j] = t_adjmat[new_org_nid_dict[i], new_org_nid_dict[j]]
            
    return f_t_adjmat, f_id_pos_dict
    
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
    transform = functions.get_zoom_transform()
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