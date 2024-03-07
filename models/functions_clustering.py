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
from sklearn.metrics.cluster._unsupervised import silhouette_score
import torch
from tqdm import tqdm

from models import datasets, functions_attpool, functions_lcsb, functions, \
    networks, functions_feat_ext, seg_datasets, functions_mil_tryk
from models.functions_feat_ext import access_encodes_imgs, avg_neigb_encodes, \
    comput_region_ctx_comb_encodes, make_neighb_coords, fill_surrounding_void, \
    select_top_att_tiles, select_top_active_tiles, filter_neg_att_tiles
from models.networks import ViT_D6_H8, reload_net, ViT_Region_4_6, CombLayers, \
    check_reuse_net, GatedAttentionPool, AttentionPool, BasicResNet18
import numpy as np
from support.env_flinc_cd45 import ENV_FLINC_CD45_REG_PT
from support.metadata import query_task_label_dict_fromcsv
from support.tools import Time, normalization
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


def load_tiles_en_rich_tuples(ENV_task, encoder, load_tile_list=None, get_ihc_dab=False):
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
    tiles_en_nd, _ = access_encodes_imgs(tile_list, encoder, ENV_task.MINI_BATCH_TILE, ENV_task.TILE_DATALOADER_WORKER,
                                         get_ihc_dab=get_ihc_dab)
    
    tiles_richencode_tuples = []
    for i, tile in enumerate(tile_list):
        encode = tiles_en_nd[i]
        slide_id = tile.query_slideid()
        tiles_richencode_tuples.append((encode, tile, slide_id))
    print('%d tiles\' encodes have been loaded, take %s sec' % (len(tiles_richencode_tuples), str(loading_time.elapsed())))
    
    return tiles_richencode_tuples


def load_tiles_neb_en_rich_tuples(ENV_task, encoder, load_tile_list=None, get_ihc_dab=False):
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
                                                    ENV_task.MINI_BATCH_TILE, ENV_task.TILE_DATALOADER_WORKER,
                                                    get_ihc_dab=get_ihc_dab)
    
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
                                        comb_layer, ctx_type, load_tile_list=None, get_ihc_dab=False):
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
                                                     ENV_task.MINI_BATCH_TILE, ENV_task.TILE_DATALOADER_WORKER,
                                                     get_ihc_dab=get_ihc_dab)    
    
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


class Instance_Clustering():
    '''
    Clustering for tile instances from all (multiple) slides
    '''

    def __init__(self, ENV_task, encoder, cluster_name, embed_type='encode', encoder_filename=None,
                 tiles_r_tuples_pkl_name=None, key_tiles_list=None, exist_clustering=None,
                 reg_encoder=None, comb_layer=None, ctx_type='reg_ass', manu_n_clusters=None, 
                 clst_in_ihc_dab=False):
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
        self.encoder = encoder
        if encoder_filename is not None:
            self.encoder, _ = reload_net(self.encoder, os.path.join(self.model_store_dir, encoder_filename))
            self.encoder = self.encoder.cuda()
        self.clst_in_ihc_dab = clst_in_ihc_dab
        ihc_dab_flag = '' if self.clst_in_ihc_dab == False else 'dab'
        self.alg_name = '{}-{}-{}-{}_{}'.format(self.cluster_name, 
                                             'none' if self.encoder is None else self.encoder.name,
                                             self.embed_type, ihc_dab_flag, _env_task_name)
        self.n_clusters = ENV_task.NUM_CLUSTERS if manu_n_clusters is None else manu_n_clusters
        self.key_tiles_list = key_tiles_list
        
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
            tiles_tuples_pkl_name = '{}-{}_{}.pkl'.format('none' if self.encoder is None else self.encoder.name,
                                                          self.embed_type, Time().date)
            if self.tiles_richencode_tuples is not None:
                save_load_tiles_rich_tuples(self.ENV_task, 
                                            self.tiles_richencode_tuples, tiles_tuples_pkl_name)
            
        self.clustering_model = exist_clustering  # if the clustering model here is None, means the clustering has not been trained
        
    def gen_tiles_richencode_tuples(self):
        if self.encoder is None:
            return None
        
        if self.embed_type == 'encode':
            tiles_richencode_tuples = load_tiles_en_rich_tuples(self.ENV_task, self.encoder,
                                                                load_tile_list=self.key_tiles_list,
                                                                get_ihc_dab=self.clst_in_ihc_dab)
        elif self.embed_type == 'neb_encode':
            tiles_richencode_tuples = load_tiles_neb_en_rich_tuples(self.ENV_task, self.encoder,
                                                                    load_tile_list=self.key_tiles_list,
                                                                    get_ihc_dab=self.clst_in_ihc_dab)
        elif self.embed_type == 'region_ctx':
            tiles_richencode_tuples = load_tiles_regionctx_en_rich_tuples(self.ENV_task, self.encoder,
                                                                          self.reg_encoder, self.comb_layer, self.ctx_type,
                                                                          load_tile_list=self.key_tiles_list,
                                                                          get_ihc_dab=self.clst_in_ihc_dab)
        elif self.embed_type == 'graph':
            ''' not available temporary '''
            tiles_richencode_tuples = load_tiles_graph_en_rich_tuples(self.ENV_task, self.encoder,
                                                                      load_tile_list=self.key_tiles_list)
        else:
            # default use the 'encode' mode
            tiles_richencode_tuples = load_tiles_en_rich_tuples(self.ENV_task, self.encoder, get_ihc_dab=self.clst_in_ihc_dab)
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
    
    def check_subclst_silhouette(self, clustering_res_pkg, check_clst, random_state=42):
        '''
        check the silhouette for current clustering results
        '''
        # get the cluster id which we need to check
        cluster_encodes = [encode for res, encode, tile, slide_id in clustering_res_pkg if res == check_clst]
    
        if len(cluster_encodes) < 2:
            # at least need 2 samples in cluster
            return 1
        
        cluster_encodes = np.array(cluster_encodes)
        
        # try a kmeans, test if the current cluster can be separated more 
        test_kmeans = KMeans(n_clusters=2, random_state=random_state)
        test_labels = test_kmeans.fit_predict(cluster_encodes)
        print('test (sub-)clustering...', end='')
    
        # calculate the inner-cluster silhouette score
        silhouette = silhouette_score(cluster_encodes, test_labels)
        print(f'checked cluster: {check_clst} -> silhouette: {silhouette}')
        return silhouette
    
    def hierarchical_clustering(self, init_clst_res_pkg, silhouette_thd=0.5, max_rounds=3):
        '''
        conduct hierarchical clustering, using recursive style
        
        Args:
            silhouette_thd:
            max_rounds:
        '''
        
        hierarchical_clst_result = []
        current_clst_pkg = init_clst_res_pkg
    
        for round in range(max_rounds):
            print(f'round: {round}')
            updated_clst_pkg = []
            split_occurred = False
    
            unique_clusters = set([res for res, _, _, _ in current_clst_pkg])
            for cluster_id in unique_clusters:
                cluster_subset = [item for item in current_clst_pkg if item[0] == cluster_id]
                cluster_encodes = [encode for res, encode, tile, slide_id in cluster_subset]
    
                cluster_silhouette = self.check_subclst_silhouette(current_clst_pkg, cluster_id)

                if cluster_silhouette > silhouette_thd: # test sub-clustering is nice, can be separate
                    split_occurred = True
                    # only apply Kmeans for hierarchical clustering
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(cluster_encodes))
                    sub_cluster_labels = kmeans.labels_

                    for i, label in enumerate(sub_cluster_labels):
                        new_res = f"{cluster_id}_{label}"
                        updated_clst_pkg.append((new_res, *cluster_subset[i][1:]))
                else:
                    # the clusters without any more split, put in the final clustering results
                    hierarchical_clst_result.extend(cluster_subset)
    
            current_clst_pkg = updated_clst_pkg
    
            if not split_occurred:  # no more split
                break
    
        # put the last round of results to final results
        hierarchical_clst_result.extend(current_clst_pkg)
        return hierarchical_clst_result
    
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
    
    
''' ------- feature assimilating functions (based on clustering results) ------- '''
    
class Feature_Assimilate():
    '''
    Assimilating the tiles with close distance 
    '''
    def __init__(self, ENV_task, clustering_res_pkg, sensitive_labels,
                 encoder, tile_en_filename=None, attK_clst=True, exc_clustered=True, 
                 assimilate_ratio=0.1, embed_type='encode', reg_encoder=None, 
                 comb_layer=None, ctx_type='reg_ass', assim_centre=True,
                 record_t_tuples_name=None, load_t_tuples_name=None,
                 clst_in_ihc_dab=False):
        '''
        TODO: annotations of this function
                
        Args:
            clustering_res_pkg = [(res, encodes[i], tiles[i], slide_ids[i])...]
            attK_clst: if the cluster results from attention K clustering?
            exc_clustered: when assimilating, if exclude the tiles which has been included in any other cluster?
        '''
        self._env_task = ENV_task
        self.for_train = True if self._env_task.DEBUG_MODE else False
        self.model_store_dir = self._env_task.MODEL_FOLDER
        
        self.assimilate_ratio = assimilate_ratio
        self.embed_type = embed_type
        self.encoder = encoder
        if tile_en_filename is not None:
            self.encoder, _ = reload_net(self.encoder, os.path.join(self.model_store_dir, tile_en_filename) )
        self.encoder = self.encoder.cuda()
        self.reg_encoder = reg_encoder
        self.comb_layer = comb_layer
        self.ctx_type = ctx_type
        self.clst_in_ihc_dab = clst_in_ihc_dab
        ihc_dab_flag = '' if self.clst_in_ihc_dab == False else 'dab'
        self.alg_name = 'ft_ass-{}-{}-{}_{}'.format(self.embed_type, self.encoder.name,
                                                    ihc_dab_flag, self._env_task.TASK_NAME)
        
        self.clustering_res = clustering_res_pkg
        self.sensitive_labels = sensitive_labels
        sensitive_res = []
        self.sensitive_tiles = []
        self.clst_sensi_tiles_dict = {}
        
        self.record_t_tuples_name = record_t_tuples_name if load_t_tuples_name is None else None
        self.load_t_tuples_name = load_t_tuples_name
        
        print('![Initial Stage] tiles assimilate_to_centre')
        # last_clst = 0 # count the last cluster id
        s_tile_keys_dict = {}
        '''
        s_tile_keys_dict: 
        {
            slide_id: tile_keys_list([...])
        }
        '''
        for clst_item in self.clustering_res:
            res, encode, tile, slide_id = clst_item
            # get the largest cluster no., use +1 indicates the assimilated tiles
            # if res > self.last_clst:
            #     last_clst = res
            
            # prepare the sensitive tiles (rich tuple) list
            if res in self.sensitive_labels:
                sensitive_res.append((res, encode, tile, slide_id))
                self.sensitive_tiles.append((tile, slide_id))
                print('> loaded tiles for all sensitive clusters.')
                if res not in self.clst_sensi_tiles_dict.keys():
                    self.clst_sensi_tiles_dict[res] = []
                    self.clst_sensi_tiles_dict[res].append((tile, slide_id))
                else:
                    self.clst_sensi_tiles_dict[res].append((tile, slide_id))
                print('> loaded tiles for each sensitive cluster, as a dictionary: {label <-> sensitive tiles for this cluster}.')
                
            tile_key = '{}-h{}-w{}'.format(slide_id, tile.h_id, tile.w_id)
            if attK_clst and exc_clustered:
                # prepare the s_tile_keys_dict which have been considered in clustering
                if slide_id not in s_tile_keys_dict.keys():
                    s_tile_keys_dict[slide_id] = []
                s_tile_keys_dict[slide_id].append(tile_key)
            else:
                if res in self.sensitive_labels:
                    if slide_id not in s_tile_keys_dict.keys():
                        s_tile_keys_dict[slide_id] = []
                    s_tile_keys_dict[slide_id].append(tile_key)
        print('> Prepared tile_keys list of tiles which have been considered for %d slides' % len(s_tile_keys_dict))
                    
        # self.ext_clst_id = last_clst + 1 # this is the clst_id of assimilated additional tiles
        if assim_centre is True:
            self.sensitive_centre = self.avg_encode_sensitive_tiles(sensitive_res)
            self.key_clusters_centres, self.key_clusters = None, []
        else:
            self.sensitive_centre = None
            self.key_clusters_centres, self.key_clusters = self.avg_encode_each_key_cluster(sensitive_res)
        self.remain_tiles_tuples = self.load_remain_tiles_encodes(s_tile_keys_dict)
        if self.record_t_tuples_name is not None:
            self.store_remain_tiles_encode_tuples(self.remain_tiles_tuples, self.record_t_tuples_name) 
        print('prepare: 1. tiles with sensitive labels as similarity source \n\
        2. tiles not in clustering as candidate tiles')
        
        # load {slide_id: {tile_loc_key: tile}}, key -> tile dictionary for each slide
        tiles_all_list, _, slides_tileidxs_dict = datasets.load_richtileslist_fromfile(self._env_task)
        self.slide_t_key_tiles_dict = {}
        for slide_id in slides_tileidxs_dict.keys():
            slide_tileidxs_list = slides_tileidxs_dict[slide_id]
            slide_tiles_list = []
            for t_id in slide_tileidxs_list:
                slide_tiles_list.append(tiles_all_list[t_id])
            tile_key_loc_dict = indicate_slide_tile_loc(slide_tiles_list)
            self.slide_t_key_tiles_dict[slide_id] = tile_key_loc_dict
        print('prepare: 3. key -> tile dict for %d slides' % len(self.slide_t_key_tiles_dict))
        
    def avg_encode_sensitive_tiles(self, sensitive_res_tuples):
        '''
        Computes the average encoding from a list of 1D vector encodings.
        
        Returns:
            numpy.ndarray: The average encoding, represented as a 1D numpy array.
        
        Raises:
            ValueError: If not all encodings have the same dimension.
        '''
        sensitive_encodes = []
        for s_res_tuple in sensitive_res_tuples:
            _, encode, _, _ = s_res_tuple
            sensitive_encodes.append(encode)
            
        try:
            encodes_array = np.array(sensitive_encodes)
        except Exception as e:
            raise ValueError('All encodes must have the same dimension!') from e
        print('have encodes of all sensitive tiles with shape', encodes_array.shape)
        mean_encode = np.mean(encodes_array, axis=0)
        return mean_encode
    
    def avg_encode_each_key_cluster(self, sensitive_res_tuples):
        '''
        calculate the encode centre for each key cluster
        '''
        key_clusters_centres, sensi_clst_lbls = [], []
        for label in self.sensitive_labels:
            # get all encodes from current label (cluster)
            encodes = [item[1] for item in sensitive_res_tuples if item[0] == label]
            sensi_clst_lbls.append(label)
            
            try:
                encodes_array = np.array(encodes)
            except Exception as e:
                raise ValueError('Combined encodes must have the same dimension!') from e
            print(f'have encodes of sensitive tiles in cluster {label} with shape', encodes_array.shape)
            # calculate the encode centre of current label
            centre = np.mean(encodes_array, axis=0)
            key_clusters_centres.append(centre)
        return key_clusters_centres, sensi_clst_lbls
                
        
    def gen_tiles_richencode_tuples(self, remain_tiles_list):
        if self.embed_type == 'encode':
            tiles_richencode_tuples = load_tiles_en_rich_tuples(self._env_task, self.encoder,
                                                                load_tile_list=remain_tiles_list,
                                                                get_ihc_dab=self.clst_in_ihc_dab)
        elif self.embed_type == 'neb_encode':
            tiles_richencode_tuples = load_tiles_neb_en_rich_tuples(self.ENV_task, self.encoder,
                                                                    load_tile_list=remain_tiles_list,
                                                                    get_ihc_dab=self.clst_in_ihc_dab)
        elif self.embed_type == 'region_ctx':
            tiles_richencode_tuples = load_tiles_regionctx_en_rich_tuples(self._env_task, self.encoder,
                                                                          self.reg_encoder, self.comb_layer, self.ctx_type,
                                                                          load_tile_list=remain_tiles_list,
                                                                          get_ihc_dab=self.clst_in_ihc_dab)
        elif self.embed_type == 'graph':
            ''' not available temporary '''
            tiles_richencode_tuples = load_tiles_graph_en_rich_tuples(self._env_task, self.encoder,
                                                                      load_tile_list=remain_tiles_list)
        else:
            # default use the 'encode' mode
            tiles_richencode_tuples = load_tiles_en_rich_tuples(self._env_task, self.encoder,
                                                                load_tile_list=remain_tiles_list,
                                                                get_ihc_dab=self.clst_in_ihc_dab)
        print('> loaded {} remain tiles and their encodes.'.format(str(len(tiles_richencode_tuples)) ) )
        return tiles_richencode_tuples
    
    def store_remain_tiles_encode_tuples(self, remain_tiles_tuples, tiles_encode_tuples_fname):
        '''
        generate the tiles encode tuples will take too much time
        store the tiles encode tuples for reloading next time
        '''
        store_clustering_pkl(self.model_store_dir, remain_tiles_tuples, tiles_encode_tuples_fname)
        print(f'store encode tuples for {len(remain_tiles_tuples)} \
        remain tiles at {self.model_store_dir} / {tiles_encode_tuples_fname}')
    
    def reload_remain_tiles_encode_tuples(self, tiles_encode_tuples_fname):
        '''
        reload the exist encode_tuples for remain tiles and save time
        need to check if the list of key clusters is match with remain tuples
        '''
        tiles_richencode_tuples = load_clustering_pkg_from_pkl(self.model_store_dir, tiles_encode_tuples_fname)
        print(f'reload encode tuples for {len(tiles_richencode_tuples)} \
        remain tiles from {self.model_store_dir} / {tiles_encode_tuples_fname}')
        return tiles_richencode_tuples
        
    def load_remain_tiles_encodes(self, clst_s_tile_keys_dict):
        '''
        load the tiles which are not participate in clustering 
        (at here, we use multi-threaded execution of tiles checking and loading)
        Return:
            remian_tiles_richencode_tuples, [(encode, tile, slide_id)...] of remain_tiles_list
        '''
        def process_tile(tile):
            s_id = tile.query_slideid()
            tile_key = '{}-h{}-w{}'.format(s_id, tile.h_id, tile.w_id)
            if s_id not in clst_s_tile_keys_dict.keys():
                return None
            if tile_key not in clst_s_tile_keys_dict[s_id]:
                return tile
            else:
                return None
        
        if self.load_t_tuples_name is not None:
            return self.reload_remain_tiles_encode_tuples(self.load_t_tuples_name)
        else:
            tiles_all_list, _, _ = datasets.load_richtileslist_fromfile(self._env_task, self.for_train)
            
            remain_tiles_list = []
            for tile in tqdm(tiles_all_list, desc="Finding remains tiles"):
                load_tile = process_tile(tile)
                if load_tile is not None:
                    remain_tiles_list.append(tile)
            # with ProcessPoolExecutor(max_workers=self._env_task.TILE_DATALOADER_WORKER) as executor:
            #     print('MULTI-PROCESS proceeding assigning...')
            #     load_tiles = list(tqdm(executor.map(process_tile, tiles_all_list), total=len(tiles_all_list), desc="Finding remains tiles"))
            #     remain_tiles_list = [tile for tile in load_tiles if tile is not None]
            print('need to load the embedding for %d not-yet-attention tiles...' % len(remain_tiles_list) )
                    
            return self.gen_tiles_richencode_tuples(remain_tiles_list)
    
    def assimilate_to_centre(self):
        '''
        Need:
            self.sensitive_centre
            self.remain_tiles_tuples
        
        Return:
            assim_tuples: [(tile, slide_id)...]
        '''
        if self.sensitive_centre is None:
            warnings.warn('Error: please check the sensitive_centre, remember: \
            if you set assim_centre as False, the sensitive_centre will be None. \
            Call <assimilate_to_each> please!')
            return []
        
        # compute distances
        distances = [(np.linalg.norm(np.array(encode) - self.sensitive_centre), encode, tile, slide_id) 
                     for encode, tile, slide_id in self.remain_tiles_tuples]
        # sort tuples by distance
        sorted_tuples = sorted(distances, key=lambda x: x[0])
        # determine the index for the top 10%
        assimilate_pct_index = int(len(sorted_tuples) * self.assimilate_ratio)
        # if the list is too small, ensure at least one element is selected
        assimilate_pct_index = max(assimilate_pct_index, 1)
        
        # print the distance threshold
        # print(assimilate_pct_index)
        distance_threshold = sorted_tuples[assimilate_pct_index - 1][0]
        print(f"--- distance threshold for top {self.assimilate_ratio}: {distance_threshold}")
        
        # Step 4: Return the top 10% tuples, removing the distance value from each tuple
        assim_tuples = [(tile, slide_id) for _, _, tile, slide_id in sorted_tuples[:assimilate_pct_index]]
        print('> assimilated %d tiles with close distance.' % len(assim_tuples))
        return assim_tuples
    
    def assimilate_to_each(self):
        '''
        Need:
            self.key_clusters_centres: List of encodes for key cluster centres
            self.remain_tiles_tuples
        
        Return:
            assim_tuples: [(tile, slide_id)...] closest to each key cluster centre
        '''
        if self.key_clusters_centres is None:
            warnings.warn('Error: please check the sensitive_centre, remember: \
            if you set assim_centre as True, the key_clusters_centres will be None. \
            Call <assimilate_to_centre> please!')
            return []
        
        assimilated = set()  # To keep track of tiles already assimilated, no repeat
        clst_assimilated_dict = {}
        
        self.key_clusters
    
        for i, centre in enumerate(self.key_clusters_centres):
            # Compute distances for remaining tiles to current centre
            distances = [(np.linalg.norm(np.array(encode) - centre), encode, tile, slide_id)
                         for encode, tile, slide_id in self.remain_tiles_tuples if (tile, slide_id) not in assimilated]
            
            # Sort tuples by distance
            sorted_tuples = sorted(distances, key=lambda x: x[0])
            
            # Determine the index for the top fixed ratio
            assimilate_pct_index = int(len(sorted_tuples) * self.assimilate_ratio)
            assimilate_pct_index = max(assimilate_pct_index, 1)  # Ensure at least one element is selected
            
            # Print the distance threshold
            distance_threshold = sorted_tuples[assimilate_pct_index - 1][0]
            print(f"--- distance threshold for top {self.assimilate_ratio} to centre {centre}: {distance_threshold}")
            
            # Update the set of assimilated tiles
            assim_tuples = [(tile, slide_id) for _, _, tile, slide_id in sorted_tuples[:assimilate_pct_index] ]
            assimilated.update(assim_tuples)
            # record the assimilated tuples for each cluster
            clst_assimilated = set(assim_tuples)
            clst_assimilated_dict[self.key_clusters[i]] = list(clst_assimilated)
            
        
        print('> assimilated %d tiles with close distance.' % len(assimilated))
        # assim_tuples = list(assimilated)
        return list(assimilated), clst_assimilated_dict
    
    def fill_void_4_assim_sensi_tiles(self, assim_tile_tuples, fills=[3]):
        '''
        fill the void for all key-clusters' hot points and assimilated patches
        '''
        hot_tiles_tuples = self.sensitive_tiles + assim_tile_tuples
        # categorise tiles for each slide
        slide_tile_dict = {}
        for i, tile_tuple in enumerate(hot_tiles_tuples):
            tile, slide_id = tile_tuple
            tile.change_to_pc_root(self._env_task)
            # print(tile.original_slide_filepath)
            if slide_id not in slide_tile_dict.keys():
                slide_tile_dict[slide_id] = []
                slide_tile_dict[slide_id].append(tile)
            else:
                slide_tile_dict[slide_id].append(tile)
              
        print('We have %d slides in list which contain hot-spot tiles' % (len(slide_tile_dict) ) )
        filled_tuples = []  
        for slide_id in slide_tile_dict.keys():
            if slide_id not in self.slide_t_key_tiles_dict.keys():
                print(f'Warning!!! This slide: {slide_id} is not in the slide list of local folder!')
                continue
            slide_tiles_list = slide_tile_dict[slide_id]
            t_key_tile_dict = self.slide_t_key_tiles_dict[slide_id]
            filled_tiles_list, _ = fill_surrounding_void(self._env_task, slide_tiles_list, [1.0]*len(slide_tiles_list), 
                                                         slide_id, tile_key_loc_dict=t_key_tile_dict, 
                                                         fills=fills, inc_org_tiles_list=False)
            slide_filled_tuples = [(t, slide_id) for t in filled_tiles_list]
            filled_tuples.extend(slide_filled_tuples)
        print('> filled void %d tiles' % len(filled_tuples))
        return filled_tuples
    
    def fill_void_4_assim_sensi_tiles_1by1_clst(self, clst_assimilated_dict, fills=[3]):
        '''
        fill the void for each cluster's hot points and assimilated patches, one by one
        '''
        clst_filled_tps_dict = {}
        for clst_lbl in clst_assimilated_dict.keys():
            clst_sensi_t_tps = self.clst_sensi_tiles_dict[clst_lbl]
            clst_assim_tile_t_tps = clst_assimilated_dict[clst_lbl]
            clst_hot_t_tps = clst_sensi_t_tps + clst_assim_tile_t_tps
            
            # categorise tiles for each slide
            slide_tile_dict = {}
            for i, tile_tuple in enumerate(clst_hot_t_tps):
                tile, slide_id = tile_tuple
                tile.change_to_pc_root(self._env_task)
                # print(tile.original_slide_filepath)
                if slide_id not in slide_tile_dict.keys():
                    slide_tile_dict[slide_id] = []
                    slide_tile_dict[slide_id].append(tile)
                else:
                    slide_tile_dict[slide_id].append(tile)
            
            print('We have %d slides in list which contain hot-spot tiles' % (len(slide_tile_dict) ) )
            c_filled_tuples = []
            for slide_id in slide_tile_dict.keys():
                if slide_id not in self.slide_t_key_tiles_dict.keys():
                    print(f'Warning!!! This slide: {slide_id} is not in the slide list of local folder!')
                    continue
                slide_tiles_list = slide_tile_dict[slide_id]
                t_key_tile_dict = self.slide_t_key_tiles_dict[slide_id]
                filled_tiles_list, _ = fill_surrounding_void(self._env_task, slide_tiles_list, [1.0]*len(slide_tiles_list), 
                                                             slide_id, tile_key_loc_dict=t_key_tile_dict, 
                                                             fills=fills, inc_org_tiles_list=False)
                slide_filled_tuples = [(t, slide_id) for t in filled_tiles_list]
                c_filled_tuples.extend(slide_filled_tuples)
                
            clst_filled_tps_dict[clst_lbl] = c_filled_tuples
            print(f'> filled void {len(c_filled_tuples)} tiles for cluster: {clst_lbl}')
        return clst_filled_tps_dict
    
    def store(self, assim_tuples, filled_tuples=[]):
        '''
        store all assimilated(not include original sensitive tiles from clusters) and filled results
        '''
        assim_tuples.extend(filled_tuples)
        assimilate_pkl_name = 'assimilate_{}{}.pkl'.format(self.alg_name, Time().date)
        store_clustering_pkl(self.model_store_dir, assim_tuples, assimilate_pkl_name)
        print('store the assimilate_to_centre(s) tiles at: {} / {}'.format(self.model_store_dir, assimilate_pkl_name))
        
    def store_each_clst(self, clst_assimilated_dict, clst_filled_tps_dict):
        '''
        store assimilated(not include original sensitive tiles from clusters) and filled results for each cluster one by one
        '''
        for clst_lbl in clst_assimilated_dict.keys():
            if clst_lbl in clst_filled_tps_dict.keys():
                clst_assimilated_dict[clst_lbl].extend(clst_filled_tps_dict[clst_lbl])
        c_assimilate_pkl_name = 'assimilate_1by1_{}{}.pkl'.format(self.alg_name, Time().date)
        store_clustering_pkl(self.model_store_dir, clst_assimilated_dict, c_assimilate_pkl_name)
        print('store the assimilate_to_centre tiles for each cluster(1by1) at: {} / {}'.format(self.model_store_dir, c_assimilate_pkl_name))
        
  
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
    
def _run_kmeans_attKtiles_encode_resnet18(ENV_task, ENV_annotation, agt_model_filenames,
                                          K_ratio, att_thd, fills, 
                                          filter_out_slide_keys=[],
                                          manu_n_clusters=5, tiles_r_tuples_pkl_name=None,
                                          selection_ihc_dab=False, clustering_ihc_dab=False):
    '''
    clustering the tiles with high attention values 
    by classification trained on H&E reports annotations
    
    Args:
        K_ratio: top % samples with attention value
        att_thd: the minimum acceptable attention value
    '''
    tile_encoder = networks.BasicResNet18(output_dim=2)
    tile_encoder = tile_encoder.cuda()
    label_dict = query_task_label_dict_fromcsv(ENV_annotation)
    
    att_all_tiles_list, _ = select_top_att_tiles(ENV_task, ENV_annotation, 
                                                 tile_encoder, agt_model_filenames, label_dict,
                                                 filter_out_slide_keys=filter_out_slide_keys,
                                                 K_ratio=K_ratio, att_thd=att_thd, fills=fills,
                                                 get_ihc_dab=selection_ihc_dab)
    
    # if we set up manu_n_clusters=3 here, only 3 clusters
    clustering = Instance_Clustering(ENV_task=ENV_task, encoder=tile_encoder,
                                     cluster_name='Kmeans', embed_type='encode',
                                     tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name,
                                     key_tiles_list=att_all_tiles_list,
                                     manu_n_clusters=manu_n_clusters,
                                     clst_in_ihc_dab=clustering_ihc_dab)
    
    clustering_res_pkg, cluster_centers = clustering.fit_predict()
    print('clustering number of centres:', len(cluster_centers))
    res_dict = {}
    for res_tuple in clustering_res_pkg:
        if res_tuple[0] not in res_dict.keys():
            res_dict[res_tuple[0]] = 0
        else:
            res_dict[res_tuple[0]] += 1
    print(res_dict)
    
    return clustering_res_pkg

def _run_kmeans_act_K_tiles_encode_resnet18(ENV_task, ENV_annotation, tile_net_filenames,
                                            K_ratio, act_thd, fills, manu_n_clusters=5,
                                            tiles_r_tuples_pkl_name=None):
    '''
    clustering the tiles with high activation values 
    by classification trained on H&E reports annotations
    
    Args:
        K_ratio: top % samples with activation score
        act_thd: the minimum acceptable activation value
    '''
    tile_encoder = networks.BasicResNet18(output_dim=2)
    
    act_all_tiles_list, _ = select_top_active_tiles(ENV_task, tile_encoder, tile_net_filenames,
                                                    K_ratio=K_ratio, act_thd=act_thd, fills=fills)
    
    # if we set up manu_n_clusters=4 here, only 4 clusters
    clustering = Instance_Clustering(ENV_task=ENV_task, encoder=tile_encoder,
                                     cluster_name='Kmeans', embed_type='encode',
                                     encoder_filename=tile_net_filenames[0],
                                     tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name,
                                     key_tiles_list=act_all_tiles_list,
                                     manu_n_clusters=manu_n_clusters)
    
    clustering_res_pkg, cluster_centers = clustering.fit_predict()
    print('clustering number of centres:', len(cluster_centers))
    res_dict = {}
    for res_tuple in clustering_res_pkg:
        if res_tuple[0] not in res_dict.keys():
            res_dict[res_tuple[0]] = 0
        else:
            res_dict[res_tuple[0]] += 1
    print(res_dict)
    
    return clustering_res_pkg

def _run_kmeans_filter_act_K_tiles_encode_resnet18(ENV_task,
                                                   tile_net_filenames, neg_t_filenames_list,
                                                   K_ratio, act_thd, fills, neg_parames,
                                                   manu_n_clusters=5, tiles_r_tuples_pkl_name=None):
    '''
    '''
    tile_encoder = networks.BasicResNet18(output_dim=2)
    
    act_all_tiles_list, slide_k_tiles_acts_dict = select_top_active_tiles(ENV_task, tile_encoder,
                                                                          tile_net_filenames,
                                                                          K_ratio=K_ratio, act_thd=act_thd, fills=fills)
    # filtering the negative activation from original maps
    for i, neg_model_filenames in enumerate(neg_t_filenames_list):
        print(f'> filter the negative activation for {neg_model_filenames[0]}...')
        k_rat, a_thd = neg_parames[i]
        _, slide_neg_tiles_acts_dict = select_top_active_tiles(ENV_task, tile_encoder, 
                                                               neg_model_filenames,
                                                               k_rat, a_thd, fills=fills)
        act_all_tiles_list, slide_k_tiles_acts_dict = filter_neg_att_tiles(slide_k_tiles_acts_dict, 
                                                                           slide_neg_tiles_acts_dict)
    
    # if we set up manu_n_clusters=4 here, only 4 clusters
    clustering = Instance_Clustering(ENV_task=ENV_task, encoder=tile_encoder,
                                     cluster_name='Kmeans', embed_type='encode',
                                     encoder_filename=tile_net_filenames[0],
                                     tiles_r_tuples_pkl_name=tiles_r_tuples_pkl_name,
                                     key_tiles_list=act_all_tiles_list,
                                     manu_n_clusters=manu_n_clusters)
    
    clustering_res_pkg, cluster_centers = clustering.fit_predict()
    print('clustering number of centres:', len(cluster_centers))
    res_dict = {}
    for res_tuple in clustering_res_pkg:
        if res_tuple[0] not in res_dict.keys():
            res_dict[res_tuple[0]] = 0
        else:
            res_dict[res_tuple[0]] += 1
    print(res_dict)
    
    return clustering_res_pkg
    

def _run_hierarchical_kmeans_encode_same(ENV_task, init_clst_pkl_name, 
                                         silhouette_thd=0.5, max_rounds=3,
                                         clst_in_ihc_dab=False):
    '''
    TODO:
    '''
    init_clst_res_pkg = load_clustering_pkg_from_pkl(ENV_task.MODEL_FOLDER, init_clst_pkl_name)
    # this clustering object is not runnable, only used for hierarchical clustering
    sec_clustering = Instance_Clustering(ENV_task=ENV_task, encoder=None,
                                         cluster_name='Kmeans', embed_type='encode',
                                         clst_in_ihc_dab=clst_in_ihc_dab)
    
    hierarchical_res_pkg = sec_clustering.hierarchical_clustering(init_clst_res_pkg, 
                                                                  silhouette_thd, max_rounds)
    hierarchical_name = init_clst_pkl_name.replace('clst-res', f'hiera-res-r{max_rounds}')
    
    store_clustering_pkl(ENV_task.MODEL_FOLDER, hierarchical_res_pkg, hierarchical_name)
    print(f'load and storage clustering results as {hierarchical_name}: ')
    res_dict = {}
    for res_tuple in hierarchical_res_pkg:
        if res_tuple[0] not in res_dict.keys():
            res_dict[res_tuple[0]] = 0
        else:
            res_dict[res_tuple[0]] += 1
    print(res_dict)
    
    return hierarchical_res_pkg
    

def _run_tiles_assimilate_centre_clst_en_resnet18(ENV_task, clustering_pkl_name, sensitive_labels, 
                                                 tile_net_filename=None, exc_clustered=True,
                                                 assim_ratio=0.01, fills=[3],
                                                 record_t_tuples_name=None, load_t_tuples_name=None,
                                                 clst_in_ihc_dab=False):
    '''
    '''
    tile_encoder = networks.BasicResNet18(output_dim=2)
    clustering_res_pkg = load_clustering_pkg_from_pkl(ENV_task.MODEL_FOLDER, clustering_pkl_name)
    assimilating = Feature_Assimilate(ENV_task, clustering_res_pkg, sensitive_labels, 
                                      encoder=tile_encoder, tile_en_filename=tile_net_filename,
                                      attK_clst=True, exc_clustered=exc_clustered,
                                      assimilate_ratio=assim_ratio, embed_type='encode', assim_centre=True,
                                      record_t_tuples_name=record_t_tuples_name, load_t_tuples_name=load_t_tuples_name,
                                      clst_in_ihc_dab=clst_in_ihc_dab)
    
    assim_tuples = assimilating.assimilate_to_centre()
    filled_tuples = assimilating.fill_void_4_assim_sensi_tiles(assim_tuples, fills) if fills is not None else []
    assimilating.store(assim_tuples, filled_tuples)
    
def _run_tiles_assimilate_each_clst_en_resnet18(ENV_task, clustering_pkl_name, sensitive_labels, 
                                               tile_net_filename=None, exc_clustered=True,
                                               assim_ratio=0.01, fills=[3],
                                               record_t_tuples_name=None, load_t_tuples_name=None,
                                               clst_in_ihc_dab=False):
    '''
    '''
    tile_encoder = networks.BasicResNet18(output_dim=2)
    clustering_res_pkg = load_clustering_pkg_from_pkl(ENV_task.MODEL_FOLDER, clustering_pkl_name)
    assimilating = Feature_Assimilate(ENV_task, clustering_res_pkg, sensitive_labels, 
                                      encoder=tile_encoder, tile_en_filename=tile_net_filename,
                                      attK_clst=True, exc_clustered=exc_clustered,
                                      assimilate_ratio=assim_ratio, embed_type='encode', assim_centre=False,
                                      record_t_tuples_name=record_t_tuples_name, load_t_tuples_name=load_t_tuples_name,
                                      clst_in_ihc_dab=clst_in_ihc_dab)
    assim_tuples, _ = assimilating.assimilate_to_each()
    filled_tuples = assimilating.fill_void_4_assim_sensi_tiles(assim_tuples, fills) if fills is not None else []
    assimilating.store(assim_tuples, filled_tuples)
    
def _run_tiles_assimilate_each_clst_1by1_en_resnet18(ENV_task, clustering_pkl_name, sensitive_labels, 
                                                     tile_net_filename=None, exc_clustered=True,
                                                     assim_ratio=0.01, fills=[3],
                                                     record_t_tuples_name=None, load_t_tuples_name=None,
                                                     clst_in_ihc_dab=False):
    '''
    '''
    tile_encoder = networks.BasicResNet18(output_dim=2)
    clustering_res_pkg = load_clustering_pkg_from_pkl(ENV_task.MODEL_FOLDER, clustering_pkl_name)
    assimilating = Feature_Assimilate(ENV_task, clustering_res_pkg, sensitive_labels, 
                                      encoder=tile_encoder, tile_en_filename=tile_net_filename,
                                      attK_clst=True, exc_clustered=exc_clustered,
                                      assimilate_ratio=assim_ratio, embed_type='encode', assim_centre=False,
                                      record_t_tuples_name=record_t_tuples_name, load_t_tuples_name=load_t_tuples_name,
                                      clst_in_ihc_dab=clst_in_ihc_dab)
    assim_tuples, clst_assimilated_dict = assimilating.assimilate_to_each()
    # record the together results
    filled_tuples = assimilating.fill_void_4_assim_sensi_tiles(assim_tuples, fills) if fills is not None else []
    assimilating.store(assim_tuples, filled_tuples)
    # record the one by one results (as dictionary)
    clst_filled_tps_dict = assimilating.fill_void_4_assim_sensi_tiles_1by1_clst(clst_assimilated_dict)
    assimilating.store_each_clst(clst_assimilated_dict, clst_filled_tps_dict)
    
    
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
