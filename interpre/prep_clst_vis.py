'''
Created on 17 Nov 2022

@author: Yang Hu
'''
from models.functions_clustering import load_clustering_pkg_from_pkl


# fixed discrete color value mapping (with 20 colors) for cv2 color palette
def col_pal_cv2_20(i_nd):
    return 5.0 + (255.0 / 20) * i_nd
# same with above with 10 colors
def col_pal_cv2_10(i_nd):
    return 10.0 + (255.0 / 10) * i_nd
# 5 colors
def col_pal_cv2_5(i_nd):
    return 20.0 + (255.0 / 5) * i_nd



def load_clst_res_encode_label(model_store_dir, clustering_pkl_name, nb_points_clst=None):
    '''
    Return:
        {clst_label: [encodes]}, [(clst_label, encode)]
        
    can sampling some points for each clst_label, not all
    '''
    clustering_res_pkg = load_clustering_pkg_from_pkl(model_store_dir, clustering_pkl_name)
    
def load_clst_res_slide_tile_label(model_store_dir, clustering_pkl_name):
    '''
    Return:
        {slide_id: [(tile, clst_label)]}
    '''

def make_clsuters_space_maps():
    '''
    '''

def make_spatial_clusters_on_slides():
    '''
    '''

if __name__ == '__main__':
    pass