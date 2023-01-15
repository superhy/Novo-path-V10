'''
Created on 13 Jan 2023

@author: Yang Hu
'''
from interpre.prep_clst_vis import load_clst_res_label_tile_slide
from interpre.prep_tools import safe_random_sample, store_nd_dict_pkl
from models.functions_vit_ext import access_att_maps_vit, \
    ext_att_maps_pick_layer, ext_patches_adjmats, norm_exted_maps, symm_adjmats, \
    gen_edge_adjmats


def extra_adjmats(tiles_attns_nd, symm, one_hot, edge_th):
    '''
    extra the graph adjacency matrix
    '''
    l_attns_nd = ext_att_maps_pick_layer(tiles_attns_nd) # t q+1 k+1
    adj_mats_nd = ext_patches_adjmats(l_attns_nd) # t q k
    adj_mats_nd = norm_exted_maps(adj_mats_nd, 't q k')
    if symm is True:
        adj_mats_nd = symm_adjmats(adj_mats_nd, rm_selfloop=True)
    if one_hot or edge_th > .0:
        adj_mats_nd = gen_edge_adjmats(adj_mats_nd, one_hot, edge_th)
    
    # tiles, q, k
    return adj_mats_nd 

def vit_graph_adjmat_tiles(ENV_task, tiles, trained_vit, layer_id=-1,
                           with_org=True, with_one_hot=False, edge_th=0.5):
    '''
    for a list of tiles,
    
    Return:
        org_adjmats_nd: original adjacency matrix, mat[x, y] may != mat[y, x]
        symm_heat_adjmats_nd: generate symmetrized adjacency matrix, mat[x, y] == mat[x, y]
        symm_onehot_adjmats_nd: symmetrized and one_hot adjacency matrix, mat[x, y] == mat[x, y] and mat[i, j] == 1
    '''
    tiles_attns_nd = access_att_maps_vit(tiles, trained_vit, 
                                         ENV_task.MINI_BATCH_TILE, 
                                         ENV_task.TILE_DATALOADER_WORKER,
                                         layer_id)
    if with_org:
        org_adjmats_nd = extra_adjmats(tiles_attns_nd, symm=False, one_hot=False, edge_th=.0)
        print('>>> generate original adjacency matrix, mat[x, y] may != mat[y, x]')
    else:
        org_adjmats_nd = None
        
    symm_heat_adjmats_nd = extra_adjmats(tiles_attns_nd, symm=True, one_hot=False, edge_th=edge_th)
    print('>>> generate symmetrized adjacency matrix, mat[x, y] == mat[x, y]')
    
    if with_one_hot:
        symm_onehot_adjmats_nd = extra_adjmats(tiles_attns_nd, symm=True, one_hot=True, edge_th=edge_th)
        print('>>> generate symmetrized and one_hot adjacency matrix, mat[x, y] == mat[x, y] and mat[i, j] == 1')
    else:
        symm_onehot_adjmats_nd = None
        
    return org_adjmats_nd, symm_heat_adjmats_nd, symm_onehot_adjmats_nd
    
    
def make_vit_graph_adjmat_clusters(ENV_task, clustering_pkl_name, 
                                   trained_vit, cluster_id=0, nb_sample=20,
                                   with_org=False, with_one_hot=True, edge_th=0.5, store_adj=False):
    '''
    generate adjacency matrix for example tiles in specific cluster
    
    Return:
        
    '''
    model_store_dir = ENV_task.MODEL_FOLDER
    graph_store_dir = ENV_task.GRAPH_STORE_DIR
    
    clst_tile_slideid_dict = load_clst_res_label_tile_slide(model_store_dir, clustering_pkl_name)
    # get (tile, slideid) tuples from specific cluster (sp_c)
    sp_c_tile_slideid_tuples = clst_tile_slideid_dict[cluster_id]
    picked_tile_slideids = safe_random_sample(sp_c_tile_slideid_tuples, nb_sample)
    
    tiles, rec_slideids = [], []
    for tile, slide_id in picked_tile_slideids:
        tiles.append(tile)
        rec_slideids.append(slide_id)
        
    org_adj_nd, symm_adj_nd, onehot_adj_nd = vit_graph_adjmat_tiles(ENV_task, tiles, trained_vit, layer_id=-1,
                                                                    with_org=with_org, with_one_hot=with_one_hot,
                                                                    edge_th=edge_th)
    adj_mats_dict = {'org': org_adj_nd, 'symm': symm_adj_nd, 'onehot': onehot_adj_nd}
    print('adj mats for org: {}, symm: {}, onehot: {}'.format('no' if org_adj_nd is None else 'yes',
                                                              'no' if symm_adj_nd is None else 'yes',
                                                              'no' if onehot_adj_nd is None else 'yes') )
    if store_adj:
        clst_adjmats_pkl_name = clustering_pkl_name.replace('clst-res', 'c-%d-adjmats'%(cluster_id) )
        store_nd_dict_pkl(graph_store_dir, adj_mats_dict, clst_adjmats_pkl_name)
        print('Store example tiles adjmats of cluster-{} package as: {}'.format(cluster_id, clst_adjmats_pkl_name))
        
    return adj_mats_dict, rec_slideids
    

if __name__ == '__main__':
    pass







