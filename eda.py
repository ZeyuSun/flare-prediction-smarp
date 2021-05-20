import os
import argparse
import pandas as pd


def get_label_stats(dataset_dir):
    stats = {}
    for dataset in ['smarp', 'sharp']:
        for split in ['train', 'test', 'train_balanced', 'test_balanced']:
            df = pd.read_csv(os.path.join(dataset_dir, dataset, split+'.csv'))
            labels = df['label']
            stats[dataset+'_'+split] = {
                'event': labels.sum(),
                'non-event': (~labels).sum(),
                'event rate': labels.sum() / len(labels)
            }
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    print(dataset_dir)
    print(stats_df.to_markdown(tablefmt='grid'))
    return  stats_df


def plot_all_samples(dataset_dir):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px

    PREFIX = 'HARP'
    KEYWORDS = ['USFLUX', 'MEANGBZ', 'R_VALUE', 'AREA', 'FLARE_INDEX']

    for dataset in ['smarp', 'sharp']:
        df = {}
        for split in ['train_balanced', 'test_balanced']:
            df[split] = pd.read_csv(os.path.join(dataset_dir, dataset, split+'.csv'))

        fig = make_subplots(rows=len(KEYWORDS), cols=2,
                            shared_xaxes='all',
                            shared_yaxes='rows',
                            column_titles=['Training set', 'Testing set'],
                            row_titles=KEYWORDS,
                            vertical_spacing=0.02)
        colors = lambda labels: [px.colors.qualitative.Plotly[i] for i in labels.astype(int)]

        def add_keyword(fig, keyword, row):
            fig.add_trace(go.Scattergl(x=df['train_balanced'].t_end, y=df['train_balanced'][keyword],
                                       customdata=df['train_balanced']['arpnum'].to_numpy(),
                                       hovertemplate=("<b>%{y} </b><br><br>" +
                                                      "time: %{x}<br>" +
                                                      f"{PREFIX} " + "%{customdata}<br>"),
                                       mode='markers',
                                       marker=dict(color=colors(df['train_balanced'].label),
                                                   opacity=0.5)),
                          row=row, col=1)

            fig.add_trace(go.Scattergl(x=df['test_balanced'].t_end, y=df['test_balanced'][keyword],
                                       customdata=df['test_balanced']['arpnum'].to_numpy(),
                                       hovertemplate=("<b>%{y} </b><br><br>" +
                                                      "time: %{x}<br>" +
                                                      f"{PREFIX} " + "%{customdata}<br>"),
                                       mode='markers',
                                       marker=dict(color=colors(df['test_balanced'].label),
                                                   opacity=0.5)),
                          row=row, col=2)
            return fig

        for i, keyword in enumerate(KEYWORDS):
            fig = add_keyword(fig, keyword, i + 1)

        fig.update_layout(  # height=1000,# width=600, # extend to the entire webpage
            title_text=f"{PREFIX} Keywords")
        filepath = os.path.join(args.stats_dir,
                                os.path.basename(dataset_dir),
                                dataset+'.html')
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        fig.write_html(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_data_dir', default='datasets')
    parser.add_argument('--stats_dir', default='outputs')
    args = parser.parse_args()

    for d in os.listdir(args.processed_data_dir):
        data_dir = os.path.join(args.processed_data_dir, d)
        if not os.path.isdir(data_dir):
            continue
        get_label_stats(data_dir)
        plot_all_samples(data_dir)
