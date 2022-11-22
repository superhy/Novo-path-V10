'''
@author: Yang Hu
'''
import os
import pickle
import random


def store_map_nd_dict_pkl(_env_heatmap_store_dir,
                          slide_heatmap_dict, heatmap_pkl_name):
    if not os.path.exists(_env_heatmap_store_dir):
        os.makedirs(_env_heatmap_store_dir)
    with open(os.path.join(_env_heatmap_store_dir, heatmap_pkl_name), 'wb') as f_pkl:
        pickle.dump(slide_heatmap_dict, f_pkl)

def load_map_pkg_from_pkl(_env_heatmap_store_dir,
                           package_pkl_name):
    pkl_filepath = os.path.join(_env_heatmap_store_dir, package_pkl_name)
    with open(pkl_filepath, 'rb') as f_pkl:
        slide_heatmap_dict = pickle.load(f_pkl)
        
    return slide_heatmap_dict

def safe_random_sample(pickpool, K):
    
    if len(pickpool) > K:
        return random.sample(pickpool, K)
    else:
        return pickpool


if __name__ == '__main__':
    pass