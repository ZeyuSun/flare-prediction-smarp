import pandas as pd
import matplotlib.pyplot as plt
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def get_scatter(df, keyword):
    rgba_scale = ['rgba({},{},{},0.5)'.format(*plotly.colors.hex_to_rgb(c))
                  for c in px.colors.qualitative.Plotly]
    colors = lambda labels: [rgba_scale[i] for i in labels.astype(int)] # faster than map
    cdata = df[['prefix', 'arpnum']].to_numpy()
    scatter = go.Scattergl(
        x=df['t_end'], y=df[keyword],
        customdata=cdata,
        hovertemplate=(
            "<b>%{y} </b><br><br>" +
            "time: %{x}<br>" +
            "%{customdata[0]} " + "%{customdata[1]}<br>"
        ),
        mode='markers',
        marker=dict(
            color=colors(df['label']),
            #opacity=0.5,
            line=dict(
                width=0,
                color='gold',
            )
        )
    )
    return scatter


def synoptic(dmf, widget=True):
    KEYWORDS = ['USFLUXL', 'AREA'] #['USFLUXL', 'MEANGBL', 'R_VALUE', 'AREA', 'FLARE_INDEX']

    fig = make_subplots(rows=len(KEYWORDS), cols=len(dmf),
                        shared_xaxes='all',
                        shared_yaxes='rows',
                        column_titles=list(dmf.keys()),
                        row_titles=KEYWORDS,
                        vertical_spacing=0.02)
    if widget:
        fig = go.FigureWidget(fig)

    for i, keyword in enumerate(KEYWORDS):
        for j, df in enumerate(dmf.values()):
            scatter = get_scatter(df, keyword)
            fig.add_trace(scatter, row=i+1, col=j+1)

    return fig


def highlight(fig, dfs):
    scatters = fig.data
    for j, split in enumerate(dfs):
        FN = dfs[split]['label'] & ~dfs[split]['pred']
        FP = ~dfs[split]['label'] & dfs[split]['pred']
        selected = FN | FP
        for i in range(len(KEYWORDS)):
            #fig.data[i*len(dfs)+j].marker.symbol = ['square' if s else 'circle' for s in selected]
            fig.data[i*len(dfs)+j].marker.size = [9 if s else 6 for s in selected]
            fig.data[i*len(dfs)+j].marker.line.width = selected.astype(int)*3