from functools import lru_cache
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from arnet.utils import read_header, get_log_intensity
#read_header = lru_cache(read_header) # might take lots of memory?


harpnum_to_noaa = np.load('harpnum_to_noaa.npy', allow_pickle=True).item()
goes_df = pd.read_csv('/home/zeyusun/SOLSTICE/goes/GOES_HMI.csv', index_col=0)
goes_df['start_time'] = goes_df['start_time'].apply(pd.to_datetime)
goes_df['intensity'] = goes_df['goes_class'].apply(get_log_intensity)
goes_df['flare_class'] = goes_df['goes_class'].str[0]


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

    #color = df['AREA'].values
    #color = ((color - color.min()) / (color.max() - color.min())) ** (1/3)
    #color = (color.argsort().argsort()) / len(color)
    #for trace in fig.data:
    #    if trace.name == by[1]:
    #        trace.marker.color=color
    for col, trace in zip(subset, fig.data):
        if col not in visible:
            trace.visible = 'legendonly'
            
    fig.update_layout(**axis_titles)
    
    return fig


def draw_reliability_error(data, reduce='intra', n_bins=10, figsize=(3.75, 3)):
    """
    Args:
        data (dict): Key is the name of the method. Value is a number of realizations,
            each having a list of y_true and a list of corresp y_prob.
        reduce (str or int): Reduction method of realizations within each key of data.
            - 'intra': Plot each realization, then taking the average and eval uncertainty.
            - 'inter': Aggregate all realization, then plot RD.
            - d: An integer. Only consider d-th realization within each group.
    """
    raise ## Stop fantasizing... Get feedback first.
    
    from sklearn.metrics import roc_curve, auc

    clim = np.mean(y_true)
    bin_edges = np.linspace(0, 1 + 1e-8, n_bins + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], ls=':', color='C7')
    ax.plot([0, 1], [clim/2, (1+clim)/2], ls=':', color='C7')
    ax.plot([0, 1], [clim, clim], ls=':', color='C7')
    ax.plot([clim, clim], [0, 1], ls=':', color='C7')

    for j, (label, pairs) in enumerate(data.items()):
        total = np.array([np.histogram(y_prob, bins=bin_edges)[0]
                          for y_prob in y_probs])
        positive = np.array([np.histogram([y for y, t in zip(y_prob, y_true) if t], bins=bin_edges)[0]
                             for y_prob in y_probs])

        if reduce == 'intra':
            posterior = (positive / total).mean(axis=0) # reliability curve is the mean of all curves
            std = (positive / total).std(axis=0)
            clim = np.mean(positive.sum(axis=1) / total.sum(axis=1))
            hist = total.mean(axis=0) # hist should also be the mean of all hists
        elif reduce == 'inter':
            posterior = positive.sum(axis=0) / total.sum(axis=0)
            # std doesn't make sense
            std = (positive / total).std(axis=0)
            clim = positive.sum() / total.sum()
            hist = total.sum(axis=0)
        elif reduce in cols:
            i = cols.index(reduce)
            posterior = positive[i] / total[i]
            clim = positive[i].sum() / total[i].sum()
            hist = total[i]
            # std to be verified
            std = np.sqrt(posterior * (1-posterior)/(hist + 3))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        ax.plot(
            bin_centers,
            posterior,
            label=label,
            color=f'C{j}',
        )
        ax.fill_between(
            fpr_grid,
            np.maximum(posterior - std, 0),
            np.minimum(posterior + std, 1),
            color=f'C{j}',
            alpha=0.2,
        )
    ax.set(xlim=[0,1], ylim=[0,1.0],
           xlabel='Predicted probability',
           ylabel='Observed rate')
    ax.set_xticks(np.linspace(0,1,11), minor=True)
    ax.set_yticks(np.linspace(0,1,11), minor=True)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

#     ax2 = ax1.twinx()
#     ax2.set_xlim([0,1])
#     #ax2.set_aspect('auto')  # equal aspect ratio won't work for twinx.
#     # "aspect" in matplotlib always refers to the data, not the axes box.
#     ax2.plot(prob_pred, bin_total, 's--', color='C1')  # red, square
#     ax2.set_ylabel('Number of samples', color='C1')

#     plt.tight_layout() # the figsize (canvas size) remains square, and the axis gets squeezed and become thinner because of the two ylabels
#     squarify(fig)
#     return fig
    plt.tight_layout()  # avoid xlabels being cut off, or use bbox_inches='tight' in savefig
    return fig


def plot_reliability(df, cols, reduce='intra'):
    """
    Deprecated in favor of draw_reliability_plot in arnet.utils.visualization.py

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
        posterior = (positive / total).mean(axis=0) # reliability curve is the mean of all curves
        std = (positive / total).std(axis=0)
        clim = np.mean(positive.sum(axis=1) / total.sum(axis=1))
        raise
        # clim should be weighted, then clim_intra == clim_inter
        hist = total.mean(axis=0) # hist should also be the mean of all hists
    elif reduce == 'inter':
        posterior = positive.sum(axis=0) / total.sum(axis=0)
        # std doesn't make sense
        std = (positive / total).std(axis=0)
        clim = positive.sum() / total.sum()
        hist = total.sum(axis=0)
    elif reduce in cols:
        i = cols.index(reduce)
        posterior = positive[i] / total[i]
        clim = positive[i].sum() / total[i].sum()
        hist = total[i]
        # std to be verified
        std = np.sqrt(posterior * (1-posterior)/(hist + 3))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ## Implementation 0: don't reinvent the wheel
    ## Should add shaded error to reliability
    #from arnet.utils import draw_reliability_plot
    #y_true = df['label']
    #y_prob = # should we average the prob in subset?
    #fig = draw_reliability_plot(y_true, y_prob, n_bins=10)
    
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
    fig.add_trace(go.Scatter(x=[0, 1],
                             y=[clim, clim],
                             mode='lines',
                             line={
                                 'color': 'rgba(150, 150, 150, 0.5)',
                                 'dash': 'dot',
                             },
                            ))
    fig.add_trace(go.Scatter(x=[clim, clim],
                             y=[0, 1],
                             mode='lines',
                             line={
                                 'color': 'rgba(150, 150, 150, 0.5)',
                                 'dash': 'dot',
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


def compute_scores(df, subset=None):
    from cotrain_helper import get_metrics
    if subset is None:
        subset = ['prob']
    metrics = {}
    for c in subset:
        metrics[c] = get_metrics(df['label'], df[c])
    df_metrics = pd.DataFrame(metrics)
    return df_metrics


def draw_roc_error(y_true, y_probs_dict, figsize=(3.75,3)):
    """
    Deprecated in favor of draw_roc in arnet.utils.visualization.py
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0,1], [0,1], ls=':', color='C7')

    for j, (label, y_probs) in enumerate(y_probs_dict.items()):
        fpr_grid = np.linspace(0, 1, 500)
        tpr_interp = [None] * len(y_probs)
        aucs = [None] * len(y_probs)
        for i, y_prob in enumerate(y_probs):
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            tpr_interp[i] = np.interp(fpr_grid, fpr, tpr)
            aucs[i] = auc(fpr, tpr)

        tpr_mean = np.mean(tpr_interp, axis=0)
        tpr_std = np.std(tpr_interp, axis=0)
        auc_mean = auc(fpr_grid, tpr_mean)
        auc_std = np.std(aucs)

        ax.plot(
            fpr_grid,
            tpr_mean,
            label=r'%s (AUC = %0.3f $\pm$ %0.3f)' % (label, auc_mean, auc_std),
            color=f'C{j}',
        )
        ax.fill_between(
            fpr_grid,
            np.maximum(tpr_mean - tpr_std, 0),
            np.minimum(tpr_mean + tpr_std, 1),
            color=f'C{j}',
            alpha=0.2,
        )
    ax.set(xlim=[0,1], ylim=[0,1.0],
           xlabel='FAR',
           ylabel='POD')
    ax.set_xticks(np.linspace(0,1,11), minor=True)
    ax.set_yticks(np.linspace(0,1,11), minor=True)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

    plt.tight_layout()  # avoid xlabels being cut off, or use bbox_inches='tight' in savefig
    return fig


def draw_diff_roc(y_true, y_prob1, y_prob2):
    from sklearn.metrics import roc_curve, auc
    from scipy.interpolate import interp1d

    fpr1, tpr1, thresholds1 = roc_curve(y_true, y_prob1)
    thresholds1[0] = thresholds1[0] - 1 + 0.001
    fpr2, tpr2, thresholds2 = roc_curve(y_true, y_prob2)
    thresholds2[0] = thresholds2[0] - 1 + 0.001    
    fpr1_func = interp1d(thresholds1, fpr1)
    tpr1_func = interp1d(thresholds1, tpr1)

    fig, ax = plt.subplots(figsize=(3.75,3))
    ax.plot(thresholds2, tpr2 - tpr1_func(thresholds2), label='POD2 - POD1')
    ax.plot(thresholds2, fpr2 - fpr1_func(thresholds2), label='FAR2 - FAR1')
    ax.legend()
    ax.grid()
    return fig


def plot_time_series(df,
                     prob_columns=None):
    """
    Args:
        df: has column 't_end', 'label', 'prob', ['prob2', ...].
        columns: columns in df to plot.
    
    Removed Args:
        arpnum (int): HARPNUM
        df_flares: has column 'start_time', 'intensity', 'flare_class'
        
    Note: We can visualize (1) mean/all pred prob of cnn/lstm (2) features, each a different yaxis
    Why do I set so many arguments, when many can be derived from df?
    """
    df['label'] = df['label'].astype(int) # so that label and plot can share axis
    arpnum = df['arpnum'].iloc[0]
    if prob_columns is None:
        val_split = df['model_query'].iloc[0].split('/')[-3]
        c1 = f'prob_CNN_{val_split}'
        c2 = 'prob'
        prob_columns = [c1 if c1 in df.columns else c2]
    df_flares = goes_df.loc[
        (goes_df['noaa_active_region'].isin(harpnum_to_noaa[arpnum])) &
        (goes_df['start_time'] >= df['t_end'].min()) &
        (goes_df['start_time'] - timedelta(hours=24)<= df['t_end'].max())]

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
    #sizes = df_melt['variable'].map({'label': 1.5, 'prob': 1}) # prob may not be in df, yielding nan
    #sizes = df_melt['variable'].apply(lambda v: 2 if v == 'label' else 1)
    # Problem is that the legend symbol is too small
    # Styling is difficult because all kinds of names (5982 uses cv2 uses prob)
    symbol_map = {'label': 'square-open', 'prob': 'cross'}
    symbol_map.update({f'prob_CNN_{i}': 'cross' for i in range(5)})
    color_discrete_map = {'label': 'blue', 'prob': 'red'}
    #color_discrete_map.update({f'prob_CNN_{i}': c
    #                          for i, c in zip(range(5), ['red'] * 5)})
    fig1 = px.scatter(df_melt,
                      x='t_end',
                      y='value',
                      #size=sizes, #pd.Series([1] * len(df_melt)),
                      #size_max=8, # used when deciding size_ref
                      symbol='variable',
                      symbol_map=symbol_map,
                      color='variable',
                      color_discrete_map=color_discrete_map,
                     )
    for trace in fig1.data:
        fig.add_trace(trace, secondary_y=False)
    
    ## Uncomment to draw class instead of intensity of flares.    
    # df_flares_dummy = pd.DataFrame(columns=df_flares.columns)
    # df_flares_dummy['flare_class'] = ['B', 'C', 'M']
    # df_flares = pd.concat([df_flares, df_flares_dummy])
    # y_col = 'flare_class'
    # yaxis2_range = [-0.5, 2.5], #'B', 'C', 'M']
    # yaxis2_title = 'Flare class'
    
    y_col = 'intensity'
    yaxis2_range = [-7, -4] #Warning: X flares > -4
    yaxis2_title = 'Log scale flare intensity'
    fig2 = px.scatter(df_flares,
                      x='start_time',
                      y=y_col,
                      color='flare_class',
                      #size=[0.001] * len(flares), # too long
                      #category_orders={
                      #    'flare_class': ['B', 'C', 'M'],
                      #}, # useless
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
        yaxis2_range = yaxis2_range,
        yaxis2_title=yaxis2_title,
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
