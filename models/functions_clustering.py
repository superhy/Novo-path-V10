'''
@author: Yang Hu
'''

import bisect
import os
import pickle
import warnings

from sklearn.cluster._affinity_propagation import AffinityPropagation
from sklearn.cluster._dbscan import DBSCAN
from sklearn.cluster._kmeans import KMeans, MiniBatchKMeans
from sklearn.cluster._mean_shift import MeanShift, estimate_bandwidth
from sklearn.cluster._spectral import SpectralClustering
import torch

from models import datasets, functions_attpool, functions_lcsb, functions, \
    networks
from models.functions_feat_ext import access_encodes_imgs, avg_neigb_encodes, \
    comput_region_ctx_comb_encodes, make_neighb_coords
from models.networks import ViT_D6_H8, reload_net, ViT_Region_4_6, CombLayers, \
    check_reuse_net, GatedAttentionPool, AttentionPool
import numpy as np
from support.env_flinc_cd45 import ENV_FLINC_CD45_REG_PT
from support.tools import Time
from wsi.process import recovery_tiles_list_from_pkl
from wsi.tiles_tools import indicate_slide_tile_loc


def store_clustering_pkl(model_store_dir, clustering_model_res, cluster_store_name):
    '''
    Args:
        model_store_dir
        clustering_model_res: clustering model or results
        cluster_store_name:
    '''
    if not os.path.exists(model_store_dir):
        os.makedirs(model_store_dir)
    with open(os.path.join(model_store_dir, cluster_store_name), 'wb') as f_pkl:
        pickle.dump(clustering_model_res, f_pkl)

        
def load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name):
    pkl_filepath = os.path.join(model_store_dir, clustering_pkl_name)
    with open(pkl_filepath, 'rb') as f_pkl:
        clustering_pkg = pickle.load(f_pkl)
        
    return clustering_pkg       


def load_tiles_en_rich_tuples(ENV_task, encoder, load_tile_list=None):
    '''
    load all tiles from .pkl folder and generate the tiles rich encode tuple list for clustering
    
    Return:
        tiles_richencode_tuples: [(encode, tile object, slide_id) ...]
    '''
    _env_process_slide_tile_pkl_dir = ENV_task.TASK_TILE_PKL_TRAIN_DIR if ENV_task.DEBUG_MODE else ENV_task.TASK_TILE_PKL_TEST_DIR
    slides_tiles_pkl_dir = _env_process_slide_tile_pkl_dir
    
    loading_time = Time()
    
    if load_tile_list == None:
        tile_list = []
        for slide_tiles_filename in os.listdir(slides_tiles_pkl_dir):
            slide_tiles_list = recovery_tiles_list_from_pkl(os.path.join(slides_tiles_pkl_dir, slide_tiles_filename))
            tile_list.extend(slide_tiles_list)
    else:
        tile_list = load_tile_list
        
    ''' >>>> the encoder here only support ViT <for the moment> '''
    tiles_en_nd, _ = access_encodes_imgs(tile_list, encoder, ENV_task.MINI_BATCH_TILE, ENV_task.TILE_DATALOADER_WORKER)
    
    tiles_richencode_tuples = []
    for i, tile in enumerate(tile_list):
        encode = tiles_en_nd[i]
        slide_id = tile.query_slideid()
        tiles_richencode_tuples.append((encode, tile, slide_id))
    print('%d tiles\' encodes have been loaded, take %s sec' % (len(tiles_richencode_tuples), str(loading_time.elapsed())))
    
    return tiles_richencode_tuples


def load_tiles_neb_en_rich_tuples(ENV_task, encoder, load_tile_list=None):
    '''
    load all tiles from .pkl folder and generate the tiles rich encode tuple list for clustering
    for each tile -> combine neighbor tiles for the key tile to generate the combination encode
    
    Return:
        tiles_richencode_tuples: [(encode, tile object, slide_id) ...]
    '''
    _env_process_slide_tile_pkl_dir = ENV_task.TASK_TILE_PKL_TRAIN_DIR if ENV_task.DEBUG_MODE else ENV_task.TASK_TILE_PKL_TEST_DIR
    slides_tiles_pkl_dir = _env_process_slide_tile_pkl_dir
    
    loading_time = Time()
    
    if load_tile_list == None:
        tile_list = []
        for slide_tiles_filename in os.listdir(slides_tiles_pkl_dir):
            slide_tiles_list = recovery_tiles_list_from_pkl(os.path.join(slides_tiles_pkl_dir, slide_tiles_filename))
            tile_list.extend(slide_tiles_list)
    else:
        tile_list = load_tile_list
        
    ''' >>>> the encoder here only support ViT <for the moment> '''
    tiles_en_nd, tile_loc_dict = access_encodes_imgs(tile_list, encoder,
                                                    ENV_task.MINI_BATCH_TILE, ENV_task.TILE_DATALOADER_WORKER)
    
    tiles_richencode_tuples = []
    for i, tile in enumerate(tile_list):
        encode = tiles_en_nd[i]
        slide_id = tile.query_slideid()
        key_encode_tuple = (encode, tile, slide_id)
        tiles_richencode_tuples.append((avg_neigb_encodes(tiles_en_nd, tile_loc_dict, key_encode_tuple), tile, slide_id))
    print('%d tiles\' neighbor combined encodes have been loaded, take %s sec' % (len(tiles_richencode_tuples),
                                                                                  str(loading_time.elapsed())))
    
    return tiles_richencode_tuples


def load_tiles_dilneb_en_rich_tuples(ENV_task, encoder, load_tile_list=None):
    '''
    load all tiles from .pkl folder and generate the tiles rich encode tuple list for clustering
    for each tile -> combine dilated + neighbor tiles for the key tile to generate the combination encode
    
    Return:
        tiles_richencode_tuples: [(encode, tile object, slide_id) ...]
        
    TODO: give the function
    '''
    tiles_richencode_tuples = []
    return tiles_richencode_tuples


def load_tiles_slidectx_en_rich_tuples(ENV_task, encoder, load_tile_list=None):
    '''
    What? from name, re-design...
    
    TODO: give the function
    '''
    tiles_richencode_tuples = []
    return tiles_richencode_tuples


def load_tiles_regionctx_en_rich_tuples(ENV_task, encoder, reg_encoder,
                                        comb_layer, ctx_type, load_tile_list=None):
    '''
    load all tiles from .pkl folder and generate the tiles rich encode tuple list for clustering
    for each tile -> combine the encode of tile and its regional context encode for the 
    key tile to generate the combination encode
    
    Args:
        encoder: the encoder for tile image (3, 256, 256)
        reg_encoder: the encoder for regional context (256, 31, 31) with grid of 31x31 each vector dim 256
        comb_layer: one layer for combine the encodes of tile and regional context
    
    Return:
        tiles_richencode_tuples: [(encode, tile object, slide_id) ...]
    '''
    _env_process_slide_tile_pkl_dir = ENV_task.TASK_TILE_PKL_TRAIN_DIR if ENV_task.DEBUG_MODE else ENV_task.TASK_TILE_PKL_TEST_DIR
    slides_tiles_pkl_dir = _env_process_slide_tile_pkl_dir
    
    loading_time = Time()
    
    if load_tile_list == None:
        tile_list = []
        for slide_tiles_filename in os.listdir(slides_tiles_pkl_dir):
            slide_tiles_list = recovery_tiles_list_from_pkl(os.path.join(slides_tiles_pkl_dir, slide_tiles_filename))
            tile_list.extend(slide_tiles_list)
    else:
        tile_list = load_tile_list
        
    ''' >>>> the encoder here only support ViT <for the moment> '''
    tiles_en_nd, tile_loc_dict = access_encodes_imgs(tile_list, encoder,
                                                     ENV_task.MINI_BATCH_TILE, ENV_task.TILE_DATALOADER_WORKER)    
    
    '''
    be noted that the encode for tiles now is combined 
    the regional context with their own encodes
    '''
    tiles_richencode_tuples = []
    for i, tile, in enumerate(tile_list):
        tile_encode = tiles_en_nd[i]
        slide_id = tile.query_slideid()
        key_encode_tuple = (tile_encode, tile, slide_id)
        print_info = True if i == 0 else False
        tiles_richencode_tuples.append((comput_region_ctx_comb_encodes(ENV_task.REG_RADIUS, tiles_en_nd,
                                                                       tile_loc_dict, key_encode_tuple,
                                                                       reg_encoder, comb_layer, ctx_type, print_info),
                                        tile, slide_id))
    print('%d tiles\' regional context combined encodes have been loaded, take %s sec' % (len(tiles_richencode_tuples),
                                                                                          str(loading_time.elapsed())))
      
    return tiles_richencode_tuples


def load_tiles_graph_en_rich_tuples(ENV_task, encoder, load_tile_list=None):
    '''
    '''
    tiles_richencode_tuples = []
    return tiles_richencode_tuples


def save_load_tiles_rich_tuples(ENV_task, tiles_richencode_tuples, tiles_tuples_pkl_name):
    '''
    save the load tiles rich encode tuples
    '''
    _encode_store_dir = ENV_task.TASK_SLIDE_MATRIX_TRAIN_DIR if ENV_task.DEBUG_MODE else ENV_task.TASK_SLIDE_MATRIX_TEST_DIR
    if not os.path.exists(_encode_store_dir):
        os.makedirs(_encode_store_dir)
    
    pkl_filepath = os.path.join(_encode_store_dir, tiles_tuples_pkl_name)
    # force store, remove the old one
    if os.path.exists(pkl_filepath):
        os.remove(pkl_filepath)
        print('remove the old .pkl file, ', end='')
        
    with open(pkl_filepath, 'wb') as f_pkl:
        pickle.dump(tiles_richencode_tuples, f_pkl)
    print('store %d tiles rich tuples at: %s' % (len(tiles_richencode_tuples), pkl_filepath))


def get_preload_tiles_rich_tuples(ENV_task, tiles_tuples_pkl_name):
    '''
    load the pre_load tiles rich encode tuples
    '''
    _encode_store_dir = ENV_task.TASK_SLIDE_MATRIX_TRAIN_DIR if ENV_task.DEBUG_MODE else ENV_task.TASK_SLIDE_MATRIX_TEST_DIR
    pkl_filepath = os.path.join(_encode_store_dir, tiles_tuples_pkl_name)
    with open(pkl_filepath, 'rb') as f_pkl:
        tiles_richencode_tuples = pickle.load(f_pkl)
    print('load the exist tiles tuples from: %s' % pkl_filepath)
        
    return tiles_richencode_tuples



''' ------------ select top attention tiles for un-supervised analysis ------------ '''
  
def select_top_att_tiles(ENV_task, tile_encoder, 
                         agt_model_filenames, label_dict,
                         K_ratio=0.3, att_thd=0.25, fill_void=False):
    '''
    select the top attention tiles by the attention pool aggregator
    using for some other tile-based analysis, like clustering. Indeed, most used for un-supervised analysis
    
    Args:
        ENV_task:
        attpool_net: aggregator, unloaded, !!! need to load here !!!
        tile_encoder: encoder, loaded, must contains the backbone
        agt_model_filenames: multi-fold aggregator trained model files, for loading here
        label_dict: <SlideMatrix_Dataset> initialize need it
        K_ratio: for calculating K for each slide, with a fixed ratio
        att_thd: at least, the attention score should higher than this
        fill_void: if fill the missed void surrounded by hot spots
        
    Return:
        att_all_tiles_list:
        slide_k_tiles_atts_dict:
    '''
    
    def average_vectors(list_of_vectors):
        array = np.array(list_of_vectors)
        average_vector = np.mean(array, axis=0)
        return average_vector
    
    batch_size_ontiles = ENV_task.MINI_BATCH_TILE
    tile_loader_num_workers = ENV_task.TILE_DATALOADER_WORKER
    batch_size_onslides = ENV_task.MINI_BATCH_SLIDEMAT
    slidemat_loader_num_workers = ENV_task.SLIDEMAT_DATALOADER_WORKER
    
    tiles_all_list, _, slides_tileidxs_dict = datasets.load_richtileslist_fromfile(ENV_task)
    
    slide_matrix_file_sets = functions_attpool.check_load_slide_matrix_files(ENV_task, 
                                                                             batch_size_ontiles, 
                                                                             tile_loader_num_workers, 
                                                                             encoder_net=tile_encoder.backbone, 
                                                                             force_refresh=False)
    # embedding_dim = np.load(slide_matrix_file_sets[0][2]).shape[-1]
    embedding_dim = 512 # TODO: change back
    if agt_model_filenames[0].find('GatedAttPool') != -1:
        attpool_net = GatedAttentionPool(embedding_dim=embedding_dim, output_dim=2)
    else:
        attpool_net = AttentionPool(embedding_dim=embedding_dim, output_dim=2)
    
    slidemat_set = datasets.SlideMatrix_Dataset(slide_matrix_file_sets, label_dict, label_free=True)
    slidemat_loader = functions.get_data_loader(slidemat_set, batch_size_onslides, slidemat_loader_num_workers, 
                                                sf=False, p_mem=False)
    # load multi-fold attention scores
    slide_attscores_dict_list = []
    for agt_model_name in agt_model_filenames:
        attpool_net, _ = networks.reload_net(attpool_net, os.path.join(ENV_task.MODEL_FOLDER, agt_model_name) )
        attpool_net = attpool_net.cuda()
        slide_attscores_dict = functions_attpool.query_slides_attscore(slidemat_loader, attpool_net,
                                                                       cutoff_padding=True, norm=True)
        slide_attscores_dict_list.append(slide_attscores_dict)
        # release the CUDA memory
        attpool_net = attpool_net.cpu()
        torch.cuda.empty_cache()
    
    att_all_tiles_list = []
    slide_k_tiles_atts_dict = {}
    for slide_id in slides_tileidxs_dict.keys():
        slide_tileidxs_list = slides_tileidxs_dict[slide_id]
        # calculate the attention score (average) from multi-fold
        list_of_attscores = []
        for slide_attscores_dict in slide_attscores_dict_list:
            slide_attscores = slide_attscores_dict[slide_id]
            list_of_attscores.append(slide_attscores)
        avg_attscores = average_vectors(list_of_attscores)
        # print(avg_attscores, len(avg_attscores))
        
        nb_tiles = len(slide_tileidxs_list)
        # print(nb_tiles)
        K = int(nb_tiles * K_ratio)
        k_slide_tiles_list, k_attscores = functions_lcsb.filter_singlesldie_top_thd_attKtiles(tiles_all_list, slide_tileidxs_list,
                                                                                              avg_attscores, K, att_thd)
        
        if fill_void:
            # fill the missed voids surrounded by hot spots
            slide_tiles_list = []
            for t_id in slide_tileidxs_list:
                slide_tiles_list.append(tiles_all_list[t_id])
            tile_key_loc_dict = indicate_slide_tile_loc(slide_tiles_list)
            k_slide_tiles_list, k_attscores = fill_surrounding_void(ENV_task, k_slide_tiles_list, k_attscores, 
                                                                    slide_id, tile_key_loc_dict)
        
        att_all_tiles_list.extend(k_slide_tiles_list)
        # print(len(k_slide_tiles_list))
        slide_k_tiles_atts_dict[slide_id] = (k_slide_tiles_list, k_attscores)
        
    return att_all_tiles_list, slide_k_tiles_atts_dict

def check_surrounding(state_map, h, w, fill=3):
    '''
    check the localised context if have hot spots enough
    
    Args:
        state_map: the activate state map on whole picture (could be heat_np for instance)
        h, w: position of point need to be check
        fill: threshold to check if this point should be included as well
    '''
     
    # print(state_map)
    (H, W) = state_map.shape
    moves = [[-1, -1], [-1, 0], [-1, +1],
             [ 0, -1], [ 0, 0], [ 0, +1],
             [+1, -1], [+1, 0], [+1, +1]]
    # print(np.max(state_map))
    
    nb_surd = 0
    sum_surd = 0.0
    for m in moves:
        p_h, p_w = h + m[0], w + m[1]
        if p_h < 0 or p_w < 0 or p_h >= H or p_w >= W:
            continue
        if state_map[p_h, p_w] > 0.0:
            # print('shot surd for: ', (h, w))
            nb_surd += 1
            sum_surd += state_map[p_h, p_w]
    if state_map[h, w] > 0.0:
        stat = False # already activate, no need to change
    else:
        stat = nb_surd >= fill
    avg_surd = sum_surd / nb_surd if nb_surd > 0 else 0.0
    return stat, avg_surd

def fill_surrounding_void(ENV_task, k_slide_tiles_list, k_attscores, 
                          slide_id, tile_key_loc_dict):
    '''
    if a spot is surrounded by attention patches, we are going to fill it as the attention one
    '''
    slide_np, _ = k_slide_tiles_list[0].get_np_scaled_slide()
    H = round(slide_np.shape[0] * ENV_task.SCALE_FACTOR / ENV_task.TILE_H_SIZE)
    W = round(slide_np.shape[1] * ENV_task.SCALE_FACTOR / ENV_task.TILE_W_SIZE)
    
    # place the original hot spots on slides
    heat_np = np.zeros((H, W), dtype=np.float64)
    for i, att_score in enumerate(k_attscores):
        h = k_slide_tiles_list[i].h_id - 1 
        w = k_slide_tiles_list[i].w_id - 1
        if h >= H or w >= W or h < 0 or w < 0:
            warnings.warn('Out of range coordinates.')
            continue
        heat_np[h, w] = att_score
    print('originally attention tiles: ', len(k_attscores) )
    
    fill_k_slide_tiles_list, fill_k_attscores = k_slide_tiles_list, k_attscores.tolist()
    # fill the voids which surrounded by hot-spots
    nb_fill = 0
    for i_h in range(H):
        for i_w in range(W):
            stat, avg_surd = check_surrounding(heat_np, i_h, i_w)
            if stat:
                tile_key = '{}-h{}-w{}'.format(slide_id, i_h, i_w)
                fill_k_slide_tiles_list.append(tile_key_loc_dict[tile_key])
                fill_k_attscores.append(avg_surd)
    print('fill surrounding tiles: ', nb_fill)
    
    return fill_k_slide_tiles_list, fill_k_attscores
    

class Instance_Clustering():
    '''
    Clustering for tile instances from all (multiple) slides
    '''

    def __init__(self, ENV_task, encoder, cluster_name, embed_type='encode',
                 tiles_r_tuples_pkl_name=None, attention_tiles_list=None, exist_clustering=None,
                 reg_encoder=None, comb_layer=None, ctx_type='reg_ass'):
        '''
        Args:
            ENV_task: task environment (hyper-parameters)
            encoder: used for load the embedding of tile-features (in default will be a ViT model)
            cluster_name: the name of selected clustering algorithm
                'Kmeans', 'Spectral', 'MeanShift', 'AffinityPropa', 'DBSCAN'
            embed_type: the type of the embedding of tile features
                'encode' -> the typical encoding from encoder (available for both ResNet and ViT)
                'semantic' ->  the semantic map extracted from ViT encoder
                'graph' -> the topological graph embeded from ViT encoder
            tiles_r_tuples_pkl_name: if have preload tiles rich encode tuples
            
            [exist_clustering: for the moment, it's None]
        '''
        
        self.ENV_task = ENV_task
        ''' prepare some parames '''
        _env_task_name = self.ENV_task.TASK_NAME
        
        self.model_store_dir = self.ENV_task.MODEL_FOLDER
        self.cluster_name = cluster_name
        self.embed_type = embed_type
        self.alg_name = '{}-{}_{}'.format(self.cluster_name, self.embed_type, _env_task_name)
        self.encoder = encoder
        self.encoder = self.encoder.cuda()
        self.n_clusters = ENV_task.NUM_CLUSTERS
        self.attention_tiles_list = attention_tiles_list
        
        self.reg_encoder, self.comb_layer = None, None
        if self.embed_type in ['region_ctx']:
            if reg_encoder is None or comb_layer is None:
                warnings.warn('need reg_encoder and comb_layer both!')
                self.embed_type = 'encode'
            else:
                self.reg_encoder, self.comb_layer = reg_encoder, comb_layer
                self.reg_encoder = self.reg_encoder.cuda()
                self.comb_layer = self.comb_layer.cuda()
        self.ctx_type = ctx_type
        
        print('![Initial Stage] clustering mode')
        print('Initialising the embedding of overall tile features...')
        if tiles_r_tuples_pkl_name is not None:
            self.tiles_richencode_tuples = get_preload_tiles_rich_tuples(self.ENV_task, tiles_r_tuples_pkl_name)
        else:
            # generate encode rich tuples
            self.tiles_richencode_tuples = self.gen_tiles_richencode_tuples()
            # store the encode rich tuples pkl
            tiles_tuples_pkl_name = '{}-{}_{}.pkl'.format(self.encoder.name, self.embed_type, Time().date)
            save_load_tiles_rich_tuples(self.ENV_task, self.tiles_richencode_tuples, tiles_tuples_pkl_name)
            
        self.clustering_model = exist_clustering  # if the clustering model here is None, means the clustering has not been trained
        
    def gen_tiles_richencode_tuples(self):
        if self.embed_type == 'encode':
            tiles_richencode_tuples = load_tiles_en_rich_tuples(self.ENV_task, self.encoder,
                                                                load_tile_list=self.attention_tiles_list)
        elif self.embed_type == 'neb_encode':
            tiles_richencode_tuples = load_tiles_neb_en_rich_tuples(self.ENV_task, self.encoder,
                                                                    load_tile_list=self.attention_tiles_list)
        elif self.embed_type == 'region_ctx':
            tiles_richencode_tuples = load_tiles_regionctx_en_rich_tuples(self.ENV_task, self.encoder,
                                                                          self.reg_encoder, self.comb_layer, self.ctx_type,
                                                                          load_tile_list=self.attention_tiles_list)
        elif self.embed_type == 'graph':
            tiles_richencode_tuples = load_tiles_graph_en_rich_tuples(self.ENV_task, self.encoder,
                                                                      load_tile_list=self.attention_tiles_list)
        else:
            # default use the 'encode' mode
            tiles_richencode_tuples = load_tiles_en_rich_tuples(self.ENV_task, self.encoder)
        return tiles_richencode_tuples
    
    def fit_predict(self):
        '''
        fit the clustering algorithm and get the predict results
        
        Return:
            clustering_res_pkg: list of results tuple: [(cluster_res, encode, tile, slide_id), ()...]
            clustering.cluster_centers_: clustering centres
        '''
        encodes, tiles, slide_ids = [], [], []
        for info_tuple in self.tiles_richencode_tuples:
            encodes.append(info_tuple[0])
            tiles.append(info_tuple[1])
            slide_ids.append(info_tuple[2])
        encodes_X = np.array(encodes)
        print('data tuples loaded!')
        
        print('Using clustering: {}'.format(self.cluster_name))
        if self.cluster_name == 'Kmeans':
            clustering = self.load_K_means()
        elif self.cluster_name == 'Spectral':
            clustering = self.load_SpectralClustering()
        elif self.cluster_name == 'MeanShift':
            clustering = self.load_MeanShift(encodes_X)
        elif self.cluster_name == 'AffinityPropa':
            clustering = self.load_AffinityPropagation()
        elif self.cluster_name == 'DBSCAN':
            clustering = self.load_DBSCAN()
        else:
            clustering = self.load_K_means()
        
        clustering_time = Time()
        print('In fitting...', end='')
        cluster_res = clustering.fit_predict(encodes_X)
        # record the trained clustering model for this
        self.clustering_model = clustering
        
        # combine the results package
        clustering_res_pkg = []
        for i, res in enumerate(cluster_res):
            clustering_res_pkg.append((res, encodes[i], tiles[i], slide_ids[i]))
        print('execute %s clustering on %d tiles with time: %s sec' % (self.cluster_name,
                                                                       len(self.tiles_richencode_tuples),
                                                                       str(clustering_time.elapsed())))
        
        clustering_res_name = 'clst-res_{}{}.pkl'.format(self.alg_name, Time().date)
        store_clustering_pkl(self.model_store_dir, clustering_res_pkg, clustering_res_name)
        print('store the clustering results at: {} / {}'.format(self.model_store_dir, clustering_res_name))
        clustering_model_name = 'clst-model_{}{}.pkl'.format(self.alg_name, Time().date)
        store_clustering_pkl(self.model_store_dir, self.clustering_model, clustering_model_name)
        print('store the clustering model at: {} / {}'.format(self.model_store_dir, clustering_model_name))
        
        centers = []
        if self.cluster_name == 'DBSCAN':
            for res_tuple in clustering_res_pkg:
                if res_tuple[0] not in centers:
                    centers.append(res_tuple[0])
        else:
            centers = clustering.cluster_centers_
        
        return clustering_res_pkg, centers
    
    def minibatch_fit(self, batch_size):
        '''
        '''
        batch_inds = []
        for i in range(int(len(self.tiles_richencode_tuples) / batch_size)):
            if (i + 1) * batch_size >= len(self.tiles_richencode_tuples):
                batch_inds.append((i * batch_size), len(self.tiles_richencode_tuples))
            else:
                batch_inds.append((i * batch_size, (i + 1) * batch_size))
                
        clustering = self.load_minibatch_K_means()
        self.cluster_name = 'MinibatchKmeans'
        
        clustering_time = Time()
        print('In fitting...', end='')
        
        for ind in batch_inds:
            encodes = []
            for info_tuple in self.tiles_richencode_tuples[ind[0]: ind[1]]:
                encodes.append(info_tuple[0])
            encodes_X = np.array(encodes)
            print('partial data tuples loaded, and fit!')
            clustering = clustering.partial_fit(encodes_X)
        self.clustering_model = clustering
        print('execute %s clustering on %d tiles with time: %s sec' % (self.cluster_name,
                                                                       len(self.tiles_richencode_tuples),
                                                                       str(clustering_time.elapsed())))
    
    def predict(self, tiles_outside_pred_tuples):
        '''
        predict cluster for new data with the trained clustering model
        Return:
            prediction_res_pkg: similar with clustering_res_pkg, list of results tuple: [(cluster_res, encode, tile, slide_id), ()...]
        '''
        if self.clustering_model is None:
            warnings.warn('no exist clustering model for prediction, please check!')
            return
        
        encodes, tiles, slide_ids = [], [], []
        for info_tuple in tiles_outside_pred_tuples:
            encodes.append(info_tuple[0])
            tiles.append(info_tuple[1])
            slide_ids.append(info_tuple[2])
        encodes_X = np.array(encodes)
        print('outside data tuples loaded!')
        
        prediction_time = Time()
        print('In fitting...', end='')
        cluster_res = self.clustering_model.predict(encodes_X)
        
        # combine the results package
        prediction_res_pkg = []
        for i, res in enumerate(cluster_res):
            prediction_res_pkg.append((res, encodes[i], tiles[i], slide_ids[i]))
        print('predict for %d tiles by %s, with time: %s sec' % (len(self.tiles_richencode_tuples),
                                                                 self.cluster_name,
                                                                 str(prediction_time.elapsed())))
        
        prediction_res_name = 'pred-res_{}{}.pkl'.format(self.alg_name, Time().date)
        store_clustering_pkl(self.model_store_dir, prediction_res_pkg, prediction_res_name)
        print('store the prediction results at: {} / {}'.format(self.model_store_dir, prediction_res_name))
        
        return prediction_res_pkg
        
    def load_K_means(self):
        '''
        load the model of k-means clustering algorithm
            with number of clusters
        
        Return:
            clustering: empty clustering model without fit
        '''
        # n_clusters = 6
        
        clustering = KMeans(n_clusters=self.n_clusters)
        return clustering
    
    def load_minibatch_K_means(self, batch_size):
        '''
        '''
        # n_clusters = 6
        
        clustering = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=batch_size)
        return clustering
    
    def load_SpectralClustering(self):
        '''
        load the model of spectral clustering algorithm
            cite: Normalized cuts and image segmentation, 2000 Jianbo Shi, Jitendra Malik
            with number of clusters
            
        Return:
            clustering: empty clustering model without fit
        '''
        # n_clusters = 6
        assign_labels = 'discretize'
        
        clustering = SpectralClustering(n_clusters=self.n_clusters, assign_labels=assign_labels)
        return clustering
        
    def load_MeanShift(self, X):
        '''
        load the model of meanshift clustering algorithm
            Dorin Comaniciu and Peter Meer, â€œMean Shift: A robust approach toward feature space analysis.
                IEEE Transactions on Pattern Analysis and Machine Intelligence. 2002. pp. 603-619.
            without number of clusters
        
        Args:
            X: we need the input encodes_X here
        
        Return:
            clustering: empty clustering model without fit
        '''
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=5000, random_state=2000)
        # bandwidth = 1.0
        print('with bandwidth: ', bandwidth)
        bin_seeding = True
        
        clustering = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
        return clustering
    
    def load_AffinityPropagation(self):
        '''
        load the model of affinity propagation clustering algorithm
            without number of clusters
        
        Return:
            clustering: empty clustering model without fit
        '''
        damping = 0.5
        max_iter = 200
        random_state = 0
        linkage = 'ward'
        
        clustering = AffinityPropagation(damping=damping, max_iter=max_iter,
                                         linkage=linkage, random_state=random_state)
        return clustering
    
    def load_DBSCAN(self):
        '''
        load the model of DBSCAN clustering algorithm
            Ester, M., H. P. Kriegel, J. Sander, and X. Xu, â€œA Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noiseâ€�.
                In: Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996
            without number of clusters
            
        Return:
            clustering: empty clustering model without fit
        '''
        eps = 1.0
        min_samples = 10
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        return clustering
    
    
''' ------ further split the clustering results into more refined clusters ------ '''

def make_tileid_label_dict(clustering_res_pkg):
    '''
    '''
    tileid_label_dict = {}
    for clst_res_tuple in clustering_res_pkg:
        label, _, tile, slide_id = clst_res_tuple
        tile_id = '{}-h{}-w{}'.format(slide_id, str(tile.h_id), str(tile.w_id))
        tileid_label_dict[tile_id] = label
    print('load clustering results package from .pkl')
        
    return tileid_label_dict

def check_neig_clst_labels(slide_id, tile, tileid_label_dict, radius):
    '''
    '''
    coordinates, _, _, _, _, _, _ = make_neighb_coords(radius, tile)
    
    neig_labels = []
    for q_h, q_w in coordinates:
        if q_h < 0 or q_w < 0:
            continue
        neig_tile_id = '{}-h{}-w{}'.format(slide_id, str(q_h), str(q_w))
        if neig_tile_id in tileid_label_dict.keys():
            neig_labels.append(tileid_label_dict[neig_tile_id])
            
    return neig_labels, coordinates
    
def refine_sp_cluster_homoneig(clustering_res_pkg, tgt_lbl, iso_thd, radius):
    '''
    split a specific cluster into several more small clusters
    check surrounding +/- 2, 5*5-1 24 neighbours
    
    Args:
        clustering_res_pkg:
        tgt_lbl: the cluster label (from 0) need to be refined
    '''
    
    tileid_label_dict = make_tileid_label_dict(clustering_res_pkg)
    slide_tgt_tiles_2_dict = {}
    for clst_res_tuple in clustering_res_pkg:
        res_lbl, _, tile, slide_id = clst_res_tuple
        if slide_id not in slide_tgt_tiles_2_dict.keys():
            slide_tgt_tiles_2_dict[slide_id] = []
        
        if res_lbl == tgt_lbl:
            neig_labels, coords = check_neig_clst_labels(slide_id, tile, tileid_label_dict, radius)
            nb_tgt_lbl = neig_labels.count(tgt_lbl)
            pct_tgt_lbl = nb_tgt_lbl * 1.0 / len(coords)
            slide_tgt_tiles_2_dict[slide_id].append((nb_tgt_lbl, pct_tgt_lbl, 0 if pct_tgt_lbl < iso_thd else 1, tile))
            print('find tile (h{}-w{}) in slide: {}, with: '.format(slide_id, str(tile.h_id), str(tile.w_id) ), 
                  (nb_tgt_lbl, pct_tgt_lbl, 'iso' if pct_tgt_lbl < iso_thd else 'gath') )
            
    return slide_tgt_tiles_2_dict

def assign_label(value, boundaries=[0.05, 0.1, 0.2, 0.5, 1.0]):
    label = bisect.bisect(boundaries, min(value, 1 - 1e-5))
    return label

def refine_sp_cluster_levels(clustering_res_pkg, tgt_lbl, radius,
                             boundaries=[0.05, 0.1, 0.2, 0.5, 1.0]):
    '''
    split a specific cluster into several more small clusters
    check surrounding +/- 2, 5*5-1 24 neighbours
    
    Args:
        clustering_res_pkg:
        tgt_lbl: the cluster label (from 0) need to be refined
    '''
    
    tileid_label_dict = make_tileid_label_dict(clustering_res_pkg)
    slide_tgt_tiles_n_dict = {}
    for clst_res_tuple in clustering_res_pkg:
        res_lbl, _, tile, slide_id = clst_res_tuple
        if slide_id not in slide_tgt_tiles_n_dict.keys():
            slide_tgt_tiles_n_dict[slide_id] = []
        
        if res_lbl == tgt_lbl:
            neig_labels, coords = check_neig_clst_labels(slide_id, tile, tileid_label_dict, radius)
            nb_tgt_lbl = neig_labels.count(tgt_lbl)
            pct_tgt_lbl = nb_tgt_lbl * 1.0 / len(coords)
            level_lbl = assign_label(pct_tgt_lbl, boundaries)
            slide_tgt_tiles_n_dict[slide_id].append((nb_tgt_lbl, pct_tgt_lbl, level_lbl, tile) )
            print('find tile in slide: {}, with level '.format(slide_id), (nb_tgt_lbl, pct_tgt_lbl, level_lbl))
            
    return slide_tgt_tiles_n_dict, boundaries

    
''' ------------------ use kmeans ------------------- '''

def _run_kmeans_encode_vit_6_8(ENV_task, vit_pt_name, tiles_r_tuples_pkl_name=None):
    vit_encoder = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                            patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    vit_encoder, _ = reload_net(vit_encoder, os.path.join(ENV_task.MODEL_FOLDER, vit_pt_name))
    clustering = Instance_Clustering(ENV_task=ENV_task, encoder=vit_encoder,
                                     cluster_name='Kmeans', embed_type='encode',
                                     tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name)
    
    clustering_res_pkg, cluster_centers = clustering.fit_predict()
    print('clustering number of centres:', len(cluster_centers))
    res_dict = {}
    for res_tuple in clustering_res_pkg:
        if res_tuple[0] not in res_dict.keys():
            res_dict[res_tuple[0]] = 0
        else:
            res_dict[res_tuple[0]] += 1
    print(res_dict)
    
def _run_kmeans_neb_encode_vit_6_8(ENV_task, vit_pt_name, tiles_r_tuples_pkl_name=None):
    vit_encoder = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                            patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    vit_encoder, _ = reload_net(vit_encoder, os.path.join(ENV_task.MODEL_FOLDER, vit_pt_name))
    clustering = Instance_Clustering(ENV_task=ENV_task, encoder=vit_encoder,
                                     cluster_name='Kmeans', embed_type='neb_encode',
                                     tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name)
    
    clustering_res_pkg, cluster_centers = clustering.fit_predict()
    print('clustering number of centres:', len(cluster_centers))
    res_dict = {}
    for res_tuple in clustering_res_pkg:
        if res_tuple[0] not in res_dict.keys():
            res_dict[res_tuple[0]] = 0
        else:
            res_dict[res_tuple[0]] += 1
    print(res_dict)
    
def _run_keamns_region_ctx_encode_vit_6_8(ENV_task, vit_pt_name,
                                          reg_vit_pt_name,
                                          tiles_r_tuples_pkl_name=None,
                                          ctx_type='reg_ass'):
    '''
    Args:
        ctx_type: 
            'reg' -> only region context
            'ass' - > only associations to key tile
            'reg_ass' -> both region context & associations to key tile
    '''
    reg_ctx_size = (ENV_task.REG_RADIUS * 2) + 1
    in_dim1_dict = {'reg': 256,
                    'ass': reg_ctx_size ** 2 - 1,
                    'reg_ass': 256 + (reg_ctx_size ** 2)}
    
    vit_encoder = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                            patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    reg_vit_encoder = ViT_Region_4_6(image_size=2 * ENV_task.REG_RADIUS + 1, patch_size=1,
                                     channels=ENV_task.TRANSFORMS_RESIZE)
    vit_encoder, _ = reload_net(vit_encoder, os.path.join(ENV_task.MODEL_FOLDER, vit_pt_name))
    
    # here, 144 = ((5 * 2 + 1) + 1) ^ 2, 5 is the radius, 
    # the first 1 is the core and the second 1 just make the image_size not odd 
    vit_reg_load = ViT_Region_4_6(image_size=144, patch_size=int(144/ENV_FLINC_CD45_REG_PT.VIT_SHAPE), channels=3)
    reg_vit_encoder, _ = check_reuse_net(reg_vit_encoder, vit_reg_load,
                                         os.path.join(ENV_task.MODEL_FOLDER, reg_vit_pt_name))
    del vit_reg_load
    
    comb_layer = CombLayers(in_dim1=in_dim1_dict[ctx_type], in_dim2=256, out_dim=256)

    clustering = Instance_Clustering(ENV_task=ENV_task, encoder=vit_encoder,
                                     cluster_name='Kmeans', embed_type='region_ctx',
                                     tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name,
                                     reg_encoder=reg_vit_encoder, comb_layer=comb_layer, ctx_type=ctx_type)
    
    clustering_res_pkg, cluster_centers = clustering.fit_predict()
    print('clustering number of centres:', len(cluster_centers))
    res_dict = {}
    for res_tuple in clustering_res_pkg:
        if res_tuple[0] not in res_dict.keys():
            res_dict[res_tuple[0]] = 0
        else:
            res_dict[res_tuple[0]] += 1
    print(res_dict)
    
def _run_kmeans_attKtiles_encode_resnet18(ENV_task):
    '''
    TODO:
    '''
    
    
''' ---- some other clustering algorithm ---- '''
 
def _run_meanshift_encode_vit_6_8(ENV_task, vit_pt_name, tiles_r_tuples_pkl_name=None):
    vit_encoder = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                            patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    vit_encoder, _ = reload_net(vit_encoder, os.path.join(ENV_task.MODEL_FOLDER, vit_pt_name))
    clustering = Instance_Clustering(ENV_task=ENV_task, encoder=vit_encoder,
                                     cluster_name='MeanShift', embed_type='encode',
                                     tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name)
    
    _, cluster_centers = clustering.fit_predict()
    print('clustering number of centres:', len(cluster_centers))

    
def _run_dbscan_encode_vit_6_8(ENV_task, vit_pt_name, tiles_r_tuples_pkl_name=None):
    vit_encoder = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                            patch_size=int(ENV_task.TILE_H_SIZE / ENV_task.VIT_SHAPE), output_dim=2)
    vit_encoder, _ = reload_net(vit_encoder, os.path.join(ENV_task.MODEL_FOLDER, vit_pt_name))
    clustering = Instance_Clustering(ENV_task=ENV_task, encoder=vit_encoder,
                                     cluster_name='DBSCAN', embed_type='encode',
                                     tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name)
    
    clustering_res_pkg, cluster_centers = clustering.fit_predict()
    print('clustering number of centres (-1 is noise):', len(cluster_centers) - 1, cluster_centers)
    res_dict = {}
    for res_tuple in clustering_res_pkg:
        if res_tuple[0] not in res_dict.keys():
            res_dict[res_tuple[0]] = 0
        else:
            res_dict[res_tuple[0]] += 1
    print(res_dict)
    

if __name__ == '__main__':
    pass
