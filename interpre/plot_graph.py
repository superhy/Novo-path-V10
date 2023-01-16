'''
@author: Yang Hu
'''

import os

import matplotlib.pyplot as plt
import networkx as nx


def plot_tile_nx_graph(ENV_task, tile_nx_G, positions, tile_graph_name='test.png'):
    '''
    '''
    graph_store_dir = ENV_task.GRAPH_STORE_DIR
    
    nx.draw(tile_nx_G,
            pos=positions,
            node_color = 'b', 
            edge_color = 'r',
            with_labels=True)
#     plt.show()
    plt.savefig(os.path.join(graph_store_dir, tile_graph_name) )

if __name__ == '__main__':
    pass