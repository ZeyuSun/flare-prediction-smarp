import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def parallel_coordinates_and_hist(*args, **kwargs):
    """
    Augment parallel_coordinates with histograms
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

    # Make subplots
    cols = len(dimensions)
    fig = make_subplots(rows=2, cols=cols, specs=[
        [{'type': 'domain', 'colspan': cols}] + [None] * (cols - 1),
        [{} for _ in range(cols)],
    ])

    # Parallel coordinates
    pc = px.parallel_coordinates(*args, **kwargs)
    fig.add_trace(pc.data[0], row=1, col=1)

    # Histograms
    for j in range(cols):
        # # How to not share binsize?
        # # hist.alignmentgroup = False # doesn't help
        # hist_fig = px.histogram(
        #     df,
        #     y=dimensions[j],
        #     color='label',
        # )
        # hist = hist_fig.data # hist is a tuple of two traces
        if dimensions[j] != 'labels':
            hist1 = go.Histogram(
                y=df.loc[~df['label'].astype(bool), dimensions[j]],
                showlegend=False,
                marker=dict(color='steelblue'),
            )
            hist2 = go.Histogram(
                y=df.loc[df['label'].astype(bool), dimensions[j]],
                showlegend=False,
                marker=dict(color='firebrick'),
            )
            fig.add_trace(hist1, row=2, col=j+1)
            fig.add_trace(hist2, row=2, col=j+1)
        else:
            bar = go.Bar(
                y=df['label']
            )
            fig.add_trace(bar, row=2, col=j+1)
        # yaxis_title overlaps with the plot on its left
        # automargin doesn't help
        #fig.update_yaxes(
        #    title_text=dimensions[j],
        #    automargin=True,
        #    row=2, col=j+1
        #)
        #fig.update_xaxes(title_text=dimensions[j], row=2, col=j+1)
    fig.update_layout(barmode='stack') # 'group', 'relative'

    return fig


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