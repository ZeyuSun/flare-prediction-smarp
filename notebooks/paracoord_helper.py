import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def parallel_coordinates_and_hist(*args, **kwargs):
    """
    Augment parallel_coordinates with histograms
    
    Args:
        df: dataframe
        dimensions: column names
        color: the column name for coloring the parallel coordinate plot
        color_hist: the column name for coloring the hist
        
    TODO:
    1. Now we color hist by labels. We should generalize to any feature. See "Augmenting parallel coordinates plots with color-coded stacked histograms" (Bok 2020). One function to consider is go.Bar. See the following sample code from ["Continuous Color Scales and Color Bars in Python
"](https://plotly.com/python/colorscales/):
    ```python
    import plotly.express as px
    df = px.data.gapminder().query("year == 2007").sort_values(by="lifeExp")
    fig = px.bar(df, y="continent", x="pop", color="lifeExp", orientation="h",
                 color_continuous_scale='Bluered_r', hover_name="country")

    fig.show()
    ```
    2. Responsive histogram. Hover on one axis show its histogram. Show the histogram of only highlighted lines. Visualize the images of the highlighted images. One potential tool is FigureWidget.
    """
    df = args[0]
    #df['label'] = df['label'].astype(int) # should be done outside
    dimensions = kwargs['dimensions']
    color_hist = kwargs.pop('color_hist', 'label')

    # Make subplots
    cols = len(dimensions)
    fig = go.FigureWidget(make_subplots(rows=2, cols=cols, specs=[
        [{'type': 'domain', 'colspan': cols}] + [None] * (cols - 1),
        [{} for _ in range(cols)],
    ]))

    # Parallel coordinates
    pc = px.parallel_coordinates(*args, **kwargs)
    fig.add_trace(pc.data[0], row=1, col=1)

    # Histograms
    for j in range(cols):
        fig_hist = px.histogram(
            df,
            y=dimensions[j],
            color=color_hist,
            color_discrete_map={0: 'steelblue', 1: 'firebrick'}
        )
        fig_hist.data = sorted(fig_hist.data, key=lambda hist: hist.legendgroup)
        fig_hist.update_traces(bingroup=None)
        fig.add_traces(fig_hist.data, rows=2, cols=j+1)
        fig.update_xaxes(title_text=dimensions[j], row=2, col=j+1)
        fig.update_yaxes(title_text=None, row=2, col=j+1)
    fig.update_layout(barmode='stack', showlegend=False) # 'group', 'relative', 'overlay'

    def update_highlight(dimension, constraintrange):
        from pathos.multiprocessing import Pool
        import numpy as np
        masks = []
        for d in fig.data[0].dimensions:
            if d.constraintrange is not None:
                crs = np.array(d.constraintrange)
                if crs.ndim == 1:
                    crs = np.expand_dims(crs, axis=0)
                masks_dim = []
                for cr in crs:
                    #labels_rev = {v: k for k, v in labels.items()}
                    #key = labels_rev[d.label]
                    key = d.label
                    masks_dim.append(df[key].between(*cr))
                masks.append(np.logical_or.reduce(masks_dim))
        mask = np.logical_and.reduce(masks)
        # Pool doesn't work here probabily because fig are copied to each worker
        with fig.batch_update():
            for i, d in enumerate(fig.data[0].dimensions):
                fig.data[i*2+1].y = df.loc[mask & (~df['label'].astype(bool)), d.label]
                fig.data[i*2+2].y = df.loc[mask & (df['label'].astype(bool)), d.label]

    for d in fig.data[0].dimensions:
        d.on_change(update_highlight, 'constraintrange')

    return fig#, update_highlight (return the handle to debug)


def test_parallel_coordinates_and_hist(data_frame, columns):
    """
    assigngroup=True, use the same bin size
    ```python
    columns = ['USFLUXL', 'Ensemble prob']
    test_parallel_coordinates_and_hist(df_test, columns=columns)
    ```
    """
    fig = make_subplots(rows=1, cols=len(columns))
    for j in range(len(columns)):
        #breakpoint()
        hist = px.histogram(
            data_frame,
            x=columns[j],
            #color='label',
            nbins=10,
        )
        fig.add_trace(hist.data[0], row=1, col=j+1)
    return fig
