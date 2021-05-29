import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


def get_label_stats(df):
    labels = df['label']
    stats = {
        'event': labels.sum(),
        'non-event': (~labels).sum(),
        'event rate': labels.sum() / len(labels)
    }
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    return stats_df


def plot_all_samples(dfs, names, dataset):
    KEYWORDS = ['USFLUXL', 'MEANGBL', 'R_VALUE', 'AREA', 'FLARE_INDEX']
    prefix = 'HARP' if dataset == 'sharp' else 'TARP'
    title = dataset.upper() + ' Keywords'

    fig = make_subplots(rows=len(KEYWORDS), cols=len(names),
                        shared_xaxes='all',
                        shared_yaxes='rows',
                        column_titles=names,
                        row_titles=KEYWORDS,
                        vertical_spacing=0.02)
    colors = lambda labels: [px.colors.qualitative.Plotly[i] for i in labels.astype(int)]

    def add_keyword(fig, keyword, row, col):
        fig.add_trace(go.Scattergl(x=df.t_end, y=df[keyword],
                                   customdata=df['arpnum'].to_numpy(),
                                   hovertemplate=("<b>%{y} </b><br><br>" +
                                                  "time: %{x}<br>" +
                                                  f"{prefix} " + "%{customdata}<br>"),
                                   mode='markers',
                                   marker=dict(color=colors(df.label),
                                               opacity=0.5)),
                      row=row, col=col)
        return fig

    for i, keyword in enumerate(KEYWORDS):
        for j, df in enumerate(dfs):
            fig = add_keyword(fig, keyword, i+1, j+1)

    fig.update_layout(title_text=title)

    return fig


def plot_selected_samples(X_train, X_test, y_train, y_test, features, title=None):
    df_train = pd.DataFrame(X_train, columns=features)
    df_train = df_train.assign(label=y_train)
    df_test = pd.DataFrame(X_test, columns=features)
    df_test = df_test.assign(label=y_test)
    dfs = [df_train, df_test]
    names = ['train', 'test']
    KEYWORDS = features

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px

    fig = make_subplots(rows=len(KEYWORDS), cols=len(names),
                        #shared_xaxes='all',
                        shared_yaxes='rows',
                        column_titles=names,
                        row_titles=KEYWORDS,
                        vertical_spacing=0.02)
    colors = lambda labels: [px.colors.qualitative.Plotly[i] for i in labels.astype(int)]

    def add_keyword(fig, keyword, row, col):
        fig.add_trace(go.Scattergl(y=df[keyword],
                                   # customdata=df['arpnum'].to_numpy(),
                                   # hovertemplate=("<b>%{y} </b><br><br>" +
                                   #                "time: %{x}<br>" +
                                   #                f"{prefix} " + "%{customdata}<br>"),
                                   mode='markers',
                                   marker=dict(color=colors(df.label),
                                               opacity=0.5)),
                      row=row, col=col)
        return fig

    for i, keyword in enumerate(KEYWORDS):
        for j, df in enumerate(dfs):
            fig = add_keyword(fig, keyword, i+1, j+1)

    fig.update_xaxes(title_text="dataframe index", row=len(KEYWORDS))

    if title:
        fig.update_layout(title_text=title)

    return fig


def plot_scatter_matrix(df):
    KEYWORDS = ['USFLUXL', 'MEANGBL', 'R_VALUE', 'AREA', 'FLARE_INDEX']
    fig = px.scatter_matrix(df,
                            #height=800,
                            dimensions=KEYWORDS,
                            color='label')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_data_dir', default='datasets')
    #parser.add_argument('--output_dir', default='outputs')
    args = parser.parse_args()

    df_dict = {}
    for d in sorted(os.listdir(args.processed_data_dir)):
        data_dir = os.path.join(args.processed_data_dir, d)
        if not os.path.isdir(data_dir):
            continue
        for dataset in ['smarp', 'sharp']:

            print(data_dir, dataset)

            df = pd.read_csv(os.path.join(data_dir, dataset+'.csv'))
            df_dict[d+dataset] = df
            stats_df = get_label_stats(df)
            print(stats_df.to_markdown(tablefmt='grid'))

            fig = plot_scatter_matrix(df)
            fig.show()

            df.hist(bins=20)
            plt.tight_layout()
            plt.show()

            #fig = plot_all_samples([df], [data_dir + ' ' + dataset], dataset)
            # filepath = os.path.join(output_dir,
            #                         os.path.basename(dataset_dir),
            #                         dataset + '.html')
            # if not os.path.exists(os.path.dirname(filepath)):
            #     os.makedirs(os.path.dirname(filepath))
            # fig.write_html(filepath)
            #fig.show()

    for dataset in ['smarp', 'sharp']:
        dirs = [d for d in sorted(os.listdir(args.processed_data_dir))
                if os.path.isdir(os.path.join(args.processed_data_dir, d))]
        dfs = [df_dict[d+dataset] for d in dirs]
        fig = plot_all_samples(dfs, dirs, dataset)
        fig.show()