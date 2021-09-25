from functools import lru_cache
import numpy as np
import pandas as pd
from tqdm import tqdm
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from arnet.utils import read_header
#read_header = lru_cache(read_header) # might take lots of memory?


def plot_sorted_probability(df, by=None, subset=None, visible=None):
    """
    Args:
        df: has columns ['prefix', 'arpnum', label', 'prob', 't_end', 'AREA', ...]
        by: columns to sort by. Default to ['label', 'prob'].
        subset: columns to show in scatter plot.
        visible: columns that only show legend. Default: same as subset
    """
    by = by or ['label', 'prob']
    hover_data = ['t_end', 'AREA']
    visible = visible or subset
    axis_titles = dict(
        xaxis_title='Sorted index',
        yaxis_title='Predicted probability',
    )
    
    df = df.copy() # https://stackoverflow.com/questions/37846357/why-does-this-dataframe-modification-within-function-change-global-outside-funct
    df['hover_name'] = df['prefix'] + ' ' + df['arpnum'].astype(str)
    df = df.sort_values(by=by).reset_index(drop=True)
    fig = px.scatter(df,
                     x=df.index,
                     y=subset,
                     hover_name='hover_name',
                     hover_data=hover_data,
                     labels={
                         'index': axis_titles['xaxis_title'],
                         'value': axis_titles['yaxis_title'],
                         't_end': 'time',
                     },
                    )
    
    set
    for col, trace in zip(subset, fig.data):
        if col not in visible:
            trace.visible = 'legendonly'
            
    fig.update_layout(**axis_titles)
    
    return fig


def plot_reliability(df, cols, reduce='intra'):
    """
    Args:
        df: each row is a sample. Columns include labels and pred prob.
        cols: prob columns in df to use to draw reliability diagram.
        reduce: Reduce method. Default to 'intra'. Can be
            - 'intra': mean across cols of the observed rate
            - 'inter': observed rate of the sum across cols
            - one of the columns.
    """
    ## Implmentation 1: melt
    # df_hist = pd.melt(df,
    #                   id_vars=['label'],
    #                   value_vars=[f'prob{i}' for i in range(5)])
    # hist, bin_edges = np.histogram(df_hist['value'], bins=20)
    # pos_hist, _ = np.histogram(df_hist.loc[df_hist['label'], 'value'], bins=bin_edges)

    ## Implmentation 2: simpler
    bin_edges = np.linspace(0, 1, 11)
    total = np.array([np.histogram(df[c], bins=bin_edges)[0]
                      for c in cols])
    positive = np.array([np.histogram(df.loc[df['label'], c], bins=bin_edges)[0]
                         for c in cols])

    if reduce == 'intra':
        posterior = (positive / total).mean(axis=0)
        std = (positive / total).std(axis=0)
        hist = total.sum(axis=0)
    elif reduce == 'inter':
        posterior = positive.sum(axis=0) / total.sum(axis=0)
        # std doesn't make sense
        std = (positive / total).std(axis=0)
        hist = total.sum(axis=0)
    elif reduce in cols:
        i = cols.index(reduce)
        posterior = positive[i] / total[i]
        hist = total[i]
        # std to be verified
        std = np.sqrt(posterior * (1-posterior)/(hist + 3))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ## Implementation 1: bar plot with error bars
    # fig = px.bar(x=bin_centers,
    #              y=posterior.mean(axis=0),
    #              error_y=posterior.std(axis=0))

    ## Implementation 2: scatter
    colors = px.colors.qualitative.D3
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    reliability = px.line(x=bin_centers,
                          y=posterior,
                          #color=[colors[0]] * len(bin_centers), # not working
                          error_y=std).data[0]
    reliability.line.color = colors[0]
    fig.add_trace(go.Scatter(x=[0, 1],
                             y=[0, 1],
                             mode='lines',
                             line={
                                 'color': 'rgba(150, 150, 150, 0.5)',
                                 'dash': 'dash',
                             },
                             ))
    fig.add_trace(reliability,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=bin_centers,
                             y=hist,
                             mode='lines+markers',
                             line=dict(dash='dot', color=colors[1]),
                             showlegend=False),
                  secondary_y=True)

    ## Implementation 3: box plot (not working nor useful)
    # fig = px.box(x=bin_centers,
    #              y=posterior)

    fig.update_layout(
        showlegend=False, # with color specified
        xaxis_title='Predicted probability',
        xaxis_tick0=0,
        xaxis_dtick=0.2,
        yaxis_title='Observed rate',
        yaxis_scaleanchor='x',
        yaxis_showgrid=True,
        yaxis_color=colors[0],
        yaxis2_title='Number of samples',
        yaxis2_color=colors[1],
        yaxis2_showgrid=False,
        width=460,
        height=450,
    )
    return fig


def plot_time_series(arpnum, df, df_flares,
                     prob_columns=None):
    """
    Args:
        arpnum (int): HARPNUM
        df: has column 't_end', 'label', 'prob', ['prob2', ...].
        df_flares: has column 'start_time', 'intensity', 'flare_class'
        columns: columns in df to plot.
        
    Note: We can visualize (1) mean/all pred prob of cnn/lstm (2) features, each a different yaxis
    """
    columns = None or ['prob']
    color_map = dict(zip(['A', 'B', 'C', 'M', 'X'], px.colors.qualitative.Dark2))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    ## Solution 1: graphical objects
    #fig.add_trace(go.Scatter(x=df['t_end'],
    #                         y=df['label'],
    #                         mode='markers',
    #                         marker={
    #                             'symbol': 'square-open',
    #                             'size': 10,
    #                         },
    #                         name='label'))
    #fig.add_trace(go.Scatter(x=df['t_end'],
    #                         y=df['prob'],
    #                         mode='markers',
    #                         marker={'symbol': 'cross'},
    #                         name='prob'))
    
    ## Solution 2: px and df.melt
    # You can use px.scatter(df, x='t_end', y=['prob', 'label']) to draw multiple traces.
    # But to control details, you have to melt (unpivot) the wide table into a long table.
    #https://stackoverflow.com/a/65950912
    df_melt = df.melt(id_vars=[c for c in df.columns if c not in ['label'] + prob_columns],
                      # id_vars are duplicated into rows to accommodate all value_vars.
                      
                      value_vars=['label'] + prob_columns, # columns to unpivot
                      
                      #var_name='variable', # 'var_' as in value_vars
                      # The 'variable' series contains old column names.
                      
                      #value_name='value' # 'value_' as in value_vars # value of column
                      # The 'value' series contains corresponding column values of this id.
                     )
    fig1 = px.scatter(df_melt,
                      x='t_end',
                      y='value',
                      symbol='variable',
                      symbol_map={'label': 'square-open', 'prob': 'cross'},
                      color='variable',
                      color_discrete_map={'label': 'blue', 'prob': 'red'},
                     )
    for trace in fig1.data:
        fig.add_trace(trace, secondary_y=False)
    
    fig2 = px.scatter(df_flares,
                      x='start_time',
                      y='intensity',
                      color='flare_class',
                      #size=[0.001] * len(flares), # too long
                      symbol_sequence=['line-ns-open'] * len(df_flares),
                      color_discrete_map=color_map,
                     )
    for trace in fig2.data:
        fig.add_trace(trace, secondary_y=True)

    fig.update_layout(
        title='HARP '+str(arpnum),
        yaxis_title='Predicted probability',
        yaxis_range=[-0.05, 1.05],
        yaxis2_showgrid=False,
        yaxis2_title='Log scale flare intensity'
    )
    
    return fig


def get_harpnum_to_noaa(harpnums):
    """
    Usage:
    harpnum_to_noaa = get_harpnum_to_noaa(df_box['arpnum'].unique())
    np.save('harpnum_to_noaa.npy', harpnum_to_noaa)
    """
    harpnum_to_noaa = {}
    for harpnum in tqdm(harpnums):
        header = read_header('sharp', harpnum)
        noaa_ars = header['NOAA_ARS'].astype(str).unique()
        assert len(noaa_ars) == 1
        noaa_ars = [int(ar) for ar in noaa_ars[0].split(',')]
        harpnum_to_noaa[harpnum] = noaa_ars
    return harpnum_to_noaa