'''
@author: Yang Hu
'''

from plotly.graph_objs.layout import xaxis, yaxis
from scipy.interpolate._bsplines import make_interp_spline

from interpre.prep_tools import load_vis_pkg_from_pkl
import numpy as np
import pandas as pd
import plotly.graph_objects as go


colors_5 = ['rgb(255, 65, 54)', 'rgb(93, 164, 214)',
            'rgb(255, 144, 14)', 'rgb(44, 160, 101)',
            'rgb(198, 113, 113)']
colors_10 = ['rgb(220, 20, 60)', 'rgb(255, 174, 185)',
             'rgb(148, 0, 211)', 'rgb(131, 111, 255)',
             'rgb(135, 206, 250)', 'rgb(0, 205, 102)', 
             'rgb(255, 255, 0)', 'rgb(139, 131, 120)', 
             'rgb(255, 127, 0)', 'rgb(198, 113, 113)']

def plot_clst_scatter(clst_redu_en_dict):
    '''
    using plotly tools for this chart
    
    k: top_k attentional embeddings
    d: last_d attentional embeddings
    
    Args:
        nb_demo: cut the first N embeds used as the demos
    '''
    
    colors = colors_10
    
    labels, embed_nds, legend_names = [], [], []
    for i in range(len(clst_redu_en_dict.keys())):
        labels.append(i)
        embed_nds.append(clst_redu_en_dict[i])
        legend_names.append('cluster-%d' % (i + 1))
    
    fig = go.Figure()
    for embeds, color, node_name in zip(embed_nds, colors, legend_names):
        fig.add_trace(go.Scatter(
            x=embeds[:, 0], y=embeds[:, 1], mode='markers',
            name=node_name,
            marker=dict(
                color=color,
#                 size=scores * 3.0 + 8.0,
                )
            ))
        
    fig.update_layout(
        plot_bgcolor='white',
        margin=dict(l=10, b=10, r=10, t=10, pad=0),
        xaxis=dict(
            showticklabels=False
            ),
        yaxis=dict(
            showticklabels=False
            ),
        legend=dict(
            font=dict(color='black', size=24),
            bordercolor='black',
            borderwidth=1,
            yanchor="top",
            y=0.2,
            xanchor="right",
            x=0.2,
            ),
        width=1000, height=1000,
        )
        
    fig.write_html('fig5_demo_encodes.html', auto_open=True)
    

''' ---------------------------------------------------------------------------------- '''
    
def _run_plot_clst_scatter(ENV_task, clustering_pkl_name):
    clst_redu_en_dict = load_vis_pkg_from_pkl(ENV_task.STATISTIC_STORE_DIR, clustering_pkl_name)
    plot_clst_scatter(clst_redu_en_dict)
    

if __name__ == '__main__':
    pass