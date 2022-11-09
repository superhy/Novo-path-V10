'''
@author: Yang Hu
'''

'''
clustering algorithm functions
'''

from sklearn.cluster._affinity_propagation import AffinityPropagation
from sklearn.cluster._kmeans import KMeans
from sklearn.cluster._mean_shift import MeanShift
from sklearn.cluster._spectral import SpectralClustering
import torch
from torch.nn.functional import softmax

import numpy as np
from support.tools import Time


def clustering_tiles_K_means(attK_input_richencodes_tuples, n_clusters=10):
    '''
    running a k-means clustering on tiles' encodes
    
    Args:
        attK_input_richencodes_tuples: EMT-high or/and low rich encodes tuple list
        
    Return:
        cluster_res_info_pkg: the cluster results list of tuple
            (cluster result (int), encode (vector), tile (tile object), slide_id (string))
    '''
    
    cluster_alg_name = '{}@Kmeans'.format(n_clusters)
    
    clustering_time = Time()
    
    encodes, tiles, slide_ids = [], [], []
    for info_tuple in attK_input_richencodes_tuples:
        encodes.append(info_tuple[0])
        tiles.append(info_tuple[1])
        slide_ids.append(info_tuple[2])
    
    encodes_X = np.array(encodes)
    
    clustering = KMeans(n_clusters=n_clusters)
    cluster_res = clustering.fit_predict(encodes_X)
    
    # combine the results package
    cluster_res_info_pkg = []
    for i, res in enumerate(cluster_res):
        cluster_res_info_pkg.append((res, encodes[i], tiles[i], slide_ids[i]))
        
    print('execute KMeans clustering on %d tiles with time: %s sec' % (len(attK_input_richencodes_tuples), str(clustering_time.elapsed())))
    
    return cluster_res_info_pkg, cluster_alg_name, clustering.cluster_centers_

def clustering_tiles_MeanShift(attK_input_richencodes_tuples):
    '''
    running mean-shift clustering on tiles' encodes
    
    Args:
        attK_input_richencodes_tuples: EMT-high or/and low rich encodes tuple list
        
    Return:
        cluster_res_info_pkg: the cluster results list of tuple
            (cluster result (int), encode (vector), tile (tile object), slide_id (string))
    '''
    cluster_alg_name = 'n@MeanShift'
    
    clustering_time = Time()
    
    encodes, tiles, slide_ids = [], [], []
    for info_tuple in attK_input_richencodes_tuples:
        encodes.append(info_tuple[0])
        tiles.append(info_tuple[1])
        slide_ids.append(info_tuple[2])
    
    encodes_X = np.array(encodes)
    
    clustering = MeanShift()
    cluster_res = clustering.fit_predict(encodes_X)
    
    # combine the results package
    cluster_res_info_pkg = []
    for i, res in enumerate(cluster_res):
        cluster_res_info_pkg.append((res, encodes[i], tiles[i], slide_ids[i]))
        
    print('execute MeanShift clustering on %d tiles with time: %s sec' % (len(attK_input_richencodes_tuples), str(clustering_time.elapsed())))
    
    return cluster_res_info_pkg, cluster_alg_name, clustering.cluster_centers_

def clustering_tiles_AffinityPropagation(attK_bi_richencodes_tuples):
    '''
    running a graph-based clustering method named "AffinityPropagation" on tiles' encodes
    this cluster do not need to set the cluster number
    
    Args:
        attK_bi_richencodes_tuples: EMT-high or low (only available for one side) rich encodes tuple list
        
    Return:
        cluster_res_info_pkg: the cluster results list of tuple
            (cluster result (int), encode (vector), tile (tile object), slide_id (string))
    '''
    
    cluster_alg_name = 'n@AffinityPropa'
    
    clustering_time = Time()
    
    encodes, tiles, slide_ids = [], [], []
    for info_tuple in attK_bi_richencodes_tuples:
        encodes.append(info_tuple[0])
        tiles.append(info_tuple[1])
        slide_ids.append(info_tuple[2])
    
    encodes_X = np.array(encodes)
    
    clustering = AffinityPropagation(damping=0.5, max_iter=200, random_state=0)
    cluster_res = clustering.fit_predict(encodes_X)
    
    # combine the results package
    cluster_res_info_pkg = []
    for i, res in enumerate(cluster_res):
        cluster_res_info_pkg.append((res, encodes[i], tiles[i], slide_ids[i]))
        
    print('execute AffinityPropagation clustering on %d tiles with time: %s sec' % (len(attK_bi_richencodes_tuples), str(clustering_time.elapsed())))
    
    return cluster_res_info_pkg, cluster_alg_name, clustering.cluster_centers_


def clustering_tiles_SpectralClustering(attK_bi_richencodes_tuples, n_clusters=10):
    '''
    running a SpectralClustering clustering on tiles' encodes
    this one should be set as the default
    
    Args:
        attK_bi_richencodes_tuples: EMT-high or low (only available for one side) rich encodes tuple list
        
    Return:
        cluster_res_info_pkg: the cluster results list of tuple
            (cluster result (int), encode (vector), tile (tile object), slide_id (string))
    '''
    
    cluster_alg_name = '{}@SpectralCluster'.format(n_clusters)
    
    clustering_time = Time()
    
    encodes, tiles, slide_ids = [], [], []
    for info_tuple in attK_bi_richencodes_tuples:
        encodes.append(info_tuple[0])
        tiles.append(info_tuple[1])
        slide_ids.append(info_tuple[2])
    
    encodes_X = np.array(encodes)
    
    clustering = SpectralClustering(n_clusters=n_clusters)
    cluster_res = clustering.fit_predict(encodes_X)
    
    # combine the results package
    cluster_res_info_pkg = []
    for i, res in enumerate(cluster_res):
        cluster_res_info_pkg.append((res, encodes[i], tiles[i], slide_ids[i]))
        
    print('execute SpectralClustering clustering on %d tiles with time: %s sec' % (len(attK_bi_richencodes_tuples), str(clustering_time.elapsed())))
    
    return cluster_res_info_pkg, cluster_alg_name, clustering.cluster_centers_

if __name__ == '__main__':
    pass