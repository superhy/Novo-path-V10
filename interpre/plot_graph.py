'''
@author: Yang Hu
'''

import os

from interpre.draw_maps import draw_original_image
import matplotlib.pyplot as plt
from models.functions_graph import load_adj_pkg_from_pkl, \
    nx_graph_from_npadj, nx_neb_graph_from_symadj
import networkx as nx


def plot_tile_nx_graph(ENV_task, tile_nx_G, positions, tile_graph_name='test.png'):
    '''
    '''
    graph_store_dir = ENV_task.GRAPH_STORE_DIR
    
    # fig = plt.figure(figsize=(8, 8))
    # ax_1 = fig.add_subplot(1, 1, 1)
    
    nx.draw(tile_nx_G,
            pos=positions,
            # ax=ax_1,
            node_color = 'b', 
            edge_color = 'r',
            with_labels=False,
            node_size=30)
    print(positions, 'nodes: %d' % len(tile_nx_G.nodes()), 'edges: %d' % len(tile_nx_G.edges()) )
    # print(tile_nx_G.edges())
#     plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(graph_store_dir, tile_graph_name) )
    # clear the plt, otherwise, the figures will be repeated coverage
    plt.clf(), plt.cla(), plt.close()
    
def plot_tiles_onehot_nx_graphs(ENV_task, adj_mats_dict, clst_id):
    '''
    '''
    onehot_adj_list, pos_list = adj_mats_dict['onehot'], adj_mats_dict['pos']
    tiles, rec_slideids = adj_mats_dict['tiles'], adj_mats_dict['slideids']
    
    clst_tilegraph_folder = 'c-{}-tiles-graph'.format(clst_id)
    clst_tilegraph_dir = os.path.join(ENV_task.GRAPH_STORE_DIR, clst_tilegraph_folder)
    if not os.path.exists(clst_tilegraph_dir):
        os.makedirs(clst_tilegraph_dir)
        print('create file dir {}'.format(clst_tilegraph_dir) )
    
    for i, tile in enumerate(tiles):
        tile_onehot_adj_nd = onehot_adj_list[i]
        positions = pos_list[i]
        t_nx_G = nx_graph_from_npadj(tile_onehot_adj_nd)
        
        tile_slide_id = rec_slideids[i]
        tiledemo_str = '{}-tile_{}'.format(tile_slide_id, 'h{}-w{}'.format(tile.h_id, tile.w_id) )
        tile_img = tile.get_np_tile()
        draw_original_image(clst_tilegraph_dir, tile_img, (tiledemo_str, '') )
        g_tile_subpath = os.path.join(clst_tilegraph_folder, '{}-tile_{}-g{}.png'.format(tile_slide_id, 'h{}-w{}'.format(tile.h_id, tile.w_id),
                                                                                         clst_id ))
        plot_tile_nx_graph(ENV_task, t_nx_G, positions,
                           tile_graph_name=g_tile_subpath)
        
def plot_tiles_vit_neb_nx_graphs(ENV_task, adj_mats_dict, clst_id):
    '''
    generate the graphs for tiles based on both self-attention in VIT and neighbors
    then plot the graphs
    '''
    symm_adj_list, pos_list = adj_mats_dict['symm'], adj_mats_dict['pos']
    tiles, rec_slideids = adj_mats_dict['tiles'], adj_mats_dict['slideids']
    
    clst_tilegraph_folder = 'c-{}-neb-graph'.format(clst_id)
    clst_tilegraph_dir = os.path.join(ENV_task.GRAPH_STORE_DIR, clst_tilegraph_folder)
    if not os.path.exists(clst_tilegraph_dir):
        os.makedirs(clst_tilegraph_dir)
        print('create file dir {}'.format(clst_tilegraph_dir) )
        
    for i, tile in enumerate(tiles):
        tile_symm_adj_nd = symm_adj_list[i]
        id_pos_dict = pos_list[i]
        neig_nxG, neig_id_pos_dict = nx_neb_graph_from_symadj(tile_symm_adj_nd, id_pos_dict)
        
        tile_slide_id = rec_slideids[i]
        tiledemo_str = '{}-tile_{}'.format(tile_slide_id, 'h{}-w{}'.format(tile.h_id, tile.w_id) )
        tile_img = tile.get_np_tile()
        draw_original_image(clst_tilegraph_dir, tile_img, (tiledemo_str, '') )
        g_tile_subpath = os.path.join(clst_tilegraph_folder, '{}-tile_{}-ng{}.png'.format(tile_slide_id, 'h{}-w{}'.format(tile.h_id, tile.w_id),
                                                                                         clst_id ))
        plot_tile_nx_graph(ENV_task, neig_nxG, neig_id_pos_dict,
                           tile_graph_name=g_tile_subpath)
        
''' --------------------------------------------------------------------------------------- '''
        
def _run_plot_tiles_onehot_nx_graphs(ENV_task, adjdict_pkl_name):
    adj_mats_dict = load_adj_pkg_from_pkl(ENV_task.GRAPH_STORE_DIR, adjdict_pkl_name)
    clst_id = adjdict_pkl_name.split('-')[1]
    plot_tiles_onehot_nx_graphs(ENV_task, adj_mats_dict, clst_id)
    
def _run_plot_tiles_neb_nx_graphs(ENV_task, adjdict_pkl_name):
    adj_mats_dict = load_adj_pkg_from_pkl(ENV_task.GRAPH_STORE_DIR, adjdict_pkl_name)
    clst_id = adjdict_pkl_name.split('-')[1]
    plot_tiles_vit_neb_nx_graphs(ENV_task, adj_mats_dict, clst_id)

if __name__ == '__main__':
    pass