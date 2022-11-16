'''
@author: Yang Hu
'''

import os

from sklearn.cluster._affinity_propagation import AffinityPropagation
from sklearn.cluster._dbscan import DBSCAN
from sklearn.cluster._kmeans import KMeans
from sklearn.cluster._mean_shift import MeanShift, estimate_bandwidth
from sklearn.cluster._spectral import SpectralClustering
import torch
from torch.nn.functional import softmax

from models.functions_vit_ext import access_encodes_vit
import numpy as np
from support.tools import Time
from wsi.process import recovery_tiles_list_from_pkl


def load_tiles_encode_rich_tuples(ENV_task, encoder):
    '''
    load all tiles from .pkl folder and generate the tiles rich encode tuple list for clustering
    
    Return:
        tiles_richencode_tuples: [(encode, tile object, slide_id) ...]
    '''
    _env_process_slide_tile_pkl_test_dir = ENV_task.TASK_TILE_PKL_TRAIN_DIR if ENV_task.DEBUG_MODE else ENV_task.TASK_TILE_PKL_TEST_DIR
    slides_tiles_pkl_dir = _env_process_slide_tile_pkl_test_dir
    
    tile_list = []
    for slide_tiles_filename in os.listdir(slides_tiles_pkl_dir):
        slide_tiles_list = recovery_tiles_list_from_pkl(os.path.join(slides_tiles_pkl_dir, slide_tiles_filename))
        tile_list.extend(slide_tiles_list)
        
    ''' >>>> the encoder here only support ViT <for the moment> '''
    tiles_en_nd = access_encodes_vit(tile_list, encoder, ENV_task.MINI_BATCH_TILE, ENV_task.TILE_DATALOADER_WORKER)
    
    tiles_richencode_tuples = []
    for i, tile in enumerate(tile_list):
        encode = tiles_en_nd[i]
        slide_id = tile.query_slideid()
        tiles_richencode_tuples.append((encode, tile, slide_id) )
    
    return tiles_richencode_tuples

def load_tiles_semantic_rich_tuples():
    '''
    '''
    tiles_richencode_tuples = []
    return tiles_richencode_tuples

def load_tiles_graph_embed_rich_tuples():
    '''
    '''
    tiles_richencode_tuples = []
    return tiles_richencode_tuples

class Instance_Clustering():
    '''
    Clustering for tile instances from all (multiple) slides
    '''
    def __init__(self, ENV_task, encoder, cluster_name, embed_type='encode'):
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
        '''
        
        self.ENV_task = ENV_task
        ''' prepare some parames '''
        _env_task_name = self.ENV_task.TASK_NAME
        
        self.model_store_dir = self.ENV_task.MODEL_FOLDER
        self.alg_name = 'cluster-{}_{}'.format(cluster_name, _env_task_name)
        self.encoder = encoder
        self.cluster_name = cluster_name
        
        print('![Initial Stage] clustering mode')
        print('Initialising the embedding of overall tile features...')
        
        self.tiles_richencode_tuples = []
        
        self.clustering_model = None # if the clustering model here is None, means the clustering has not been trained
        
    
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
        
        return clustering_res_pkg, clustering.cluster_centers_
    
    def predict(self, tiles_outside_pred_tuples):
        '''
        predict cluster for new data with the trained clustering model
        '''
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
        
        return prediction_res_pkg
        
        
    def load_K_means(self):
        '''
        load the model of k-means clustering algorithm
            with number of clusters
        
        Return:
            clustering: empty clustering model without fit
        '''
        n_clusters = 10
        
        clustering = KMeans(n_clusters=n_clusters)
        return clustering
    
    def load_SpectralClustering(self):
        '''
        load the model of spectral clustering algorithm
            cite: Normalized cuts and image segmentation, 2000 Jianbo Shi, Jitendra Malik
            with number of clusters
            
        Return:
            clustering: empty clustering model without fit
        '''
        n_clusters = 10
        assign_labels = 'discretize'
        
        clustering = SpectralClustering(n_clusters=n_clusters, assign_labels=assign_labels)
        return clustering
        
    def load_MeanShift(self, X):
        '''
        load the model of meanshift clustering algorithm
            Dorin Comaniciu and Peter Meer, “Mean Shift: A robust approach toward feature space analysis”.
                IEEE Transactions on Pattern Analysis and Machine Intelligence. 2002. pp. 603-619.
            without number of clusters
        
        Args:
            X: we need the input encodes_X here
        
        Return:
            clustering: empty clustering model without fit
        '''
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
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
                                         linkage = linkage,random_state=random_state)
        return clustering
    
    def load_DBSCAN(self):
        '''
        load the model of DBSCAN clustering algorithm
            Ester, M., H. P. Kriegel, J. Sander, and X. Xu, “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise”.
                In: Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996
            without number of clusters
            
        Return:
            clustering: empty clustering model without fit
        '''
        eps = 0.3
        min_samples = 10
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        return clustering
    

if __name__ == '__main__':
    pass