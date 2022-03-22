from functools import lru_cache
from ipdb import set_trace as breakpoint
from typing import Union
import pandas as pd
from uncertainties import ufloat
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri('file:///home/zeyusun/work/flare-prediction-smarp/mlruns')
client = MlflowClient()


def get_columns(name):
    columns = {
        'tags.database_name': 'database',
        'tags.dataset_name': 'dataset',
        'tags.estimator_name': 'estimator',
        'params.DATA.SEED': 'seed',
        'metrics.test/accuracy': 'ACC',
        'metrics.test/auc': 'AUC',
        'metrics.test/tss': 'TSS',
        'metrics.test/hss2': 'HSS',
        'metrics.test/bss': 'BSS',
    }
    return columns


@lru_cache  # lru_cached needed by all_in_one, retrieving all probs
def retrieve(experiment_name, parent_run_name, p=None):
    """
    Args:
        p=0: Union[int, list, tuple, slice].
            index the number of runs
            Invalid Syntax: list/slice is not hashable by lru_cache
    """
    # Get runs of an experiment
    exp_id = client.get_experiment_by_name(experiment_name).experiment_id
    runs = mlflow.search_runs(exp_id)
    #runs = mlflow.search_runs(“<experiment_id>”, “metrics.loss < 2.5”)

    # Select runs by parent run name
    parent_runs = runs.loc[runs['tags.mlflow.runName'] == parent_run_name]
    if len(parent_runs) == 0:
        unique_run_names = runs['tags.mlflow.runName'].unique()
        print(f"No parentRunName {parent_run_name} in {unique_run_names}")
        raise

    p = p or slice(None)
    p = [p] if isinstance(p, int) else p
    # print('Select iloc {} from \n{}'.format(
    #     p,
    #     parent_runs[['start_time', 'tags.mlflow.runName']])) #, 'tags.mlflow.source.git.commit']]))
    #     # may not be a git repo. Do not add it in.

    #import ipdb; ipdb.set_trace()
    parentRunId = parent_runs['run_id'].iloc[p]
    runs = runs.loc[(runs['tags.mlflow.parentRunId'].isin(parentRunId)) &
                    (runs['status'] == 'FINISHED')]
    return runs


def select(runs, columns=None, rows=None):
    rows = rows or {}
    columns = columns or get_columns('arnet')

    # Select and rename columns
    if columns is not None:
        try:
            runs = runs.loc[:, list(columns.keys())]
        except:
            print(runs.columns.values)
            raise
        runs = runs.rename(columns=lambda k: columns[k])

    # Rename rows
    for col, mapping in rows.items():
        mask = runs[col].isin(mapping)
        runs.loc[mask, col] = runs.loc[mask, col].map(mapping)

    return runs


def diff(runs_1, runs_2, subset=None):
    """
    The rows of the two dataframes must have the same setting.
    """
    subset = subset or ['ACC', 'AUC', 'TSS', 'HSS', 'BSS']
    runs_diff = runs_2.copy()
    runs_diff[subset] -= runs_1.loc[:, subset].values
    return runs_diff


# def compare(*runs, subset=None):
#     subset = subset or ['ACC', 'AUC', 'TSS', 'HSS', 'BSS']
#     runs_compare = runs[0].copy()
#     runs_compare[subset] =



def organize(runs, by=None, std=False):
    by = by or ['dataset', 'estimator']
    # sort:
    extract_hours = lambda s: s.str.split('_').str[2].str.replace('hr', '').astype(int)

    if std:
        df = (runs
            .groupby(by)
            .agg(lambda s: ufloat(s.mean(), s.std())) #['mean', 'std']) #FutureWarning: Dropping invalid columns in DataFrameGroupBy.agg is deprecated. In a future version, a TypeError will be raised. Before calling .agg, select only columns which should be valid for the aggregating function.
            .unstack(-1).T
            #.sort_values('database', axis=1, key=extract_hours)
            #.round(4)
            .applymap('{:.3f}'.format)
        )
    else:
        df = (runs
            .groupby(by)
            .agg('mean') #FutureWarning: Dropping invalid columns in DataFrameGroupBy.agg is deprecated. In a future version, a TypeError will be raised. Before calling .agg, select only columns which should be valid for the aggregating function.
            .unstack(-1).T
            #.sort_values('database', axis=1, key=extract_hours)
            .round(4)
            #.applymap('{:.3f}'.format)
        )
    return df


def style(runs, by=None):
    by = by or ['dataset', 'estimator']
    df = organize(runs, by=by, std=False)
    df_style = (df
            .style
            .background_gradient(axis=None)#, vmin=0.7)
            .set_precision(3))
    return df_style


def typeset(df, **kwargs):
    """
    Usage:
    ```python
    df = organize(runs, by=by, std=True)
    print(typeset(df))
    ```
    """
    df_latex = df.to_latex(
        #column_format='c' * df.shape[1], # index isn't counted as columns
        multicolumn_format='c',
        multirow=True,
        #escape=False,
        **kwargs,
    )
    return df_latex


def tensorboard(runs):
    tb = runs['artifact_uri'].str.replace('file://', '') + '/tensorboard'
    dirs = ','.join([f"{idx}_{runs.loc[idx, 'tags.dataset_name']}_{runs.loc[idx, 'tags.estimator_name']}:{tb[idx]}" for idx in tb.index])
    return dirs


def paired_ttest(a, b):
    """
    H0: a <= b
    H1: a > b
    Equivalent to: ttest_rel(a, b, alternative='greater')
    """
    import numpy as np
    from scipy.stats import t

    if isinstance(a, list) or isinstance(a, pd.Series):
        a = np.array(a)
    if isinstance(b, list) or isinstance(b, pd.Series):
        b = np.array(b)

    assert a.ndim == 1
    assert b.ndim == 1
    assert a.shape == b.shape

    x = a - b
    n = len(x)
    dof = n - 1

    mu = np.mean(x)
    std = np.std(x, ddof=1)
    statistic = mu / (std / np.sqrt(n))
    pvalue = t.sf(statistic, dof) # sf(x) = 1 - cdf(x)

    return statistic, pvalue


def get_mask(runs, dataset_names, estimator_names):
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    if isinstance(estimator_names, str):
        estimator_names = [estimator_names]
    mask = (
        runs['dataset'].isin(dataset_names) &
        runs['estimator'].isin(estimator_names)
    )
    return mask


def print_pvalues(runs, dataset_name):
    print(f'Is fused_{dataset_name} better than {dataset_name}?')
    for estimator_name in ['LSTM', 'CNN']:
        print(estimator_name)
        for metric in ['ACC', 'AUC', 'TSS', 'HSS', 'BSS']:
            # TODO: sort to make sure measurements are paired
            a = runs.loc[get_mask(runs, 'fused_'+dataset_name, estimator_name), metric].tolist()
            b = runs.loc[get_mask(runs, dataset_name, estimator_name), metric].tolist()
            print(metric, paired_ttest(a, b))


def tabulate_pvalues(runs, metrics=None):
    metrics = metrics or ['ACC', 'AUC', 'TSS', 'HSS', 'BSS']
    items = []
    for dataset_name in ['sharp', 'smarp']:
        for estimator_name in ['LSTM', 'CNN']:
            for metric in metrics:
                a = runs.loc[get_mask(runs, 'fused_'+dataset_name, estimator_name), metric].tolist()
                b = runs.loc[get_mask(runs, dataset_name, estimator_name), metric].tolist()
                statistic, pvalue = paired_ttest(a, b)
                items.append({
                    'S': metric,
                    'estimator': estimator_name,
                    'tested hypothesis': f'S(fused_{dataset_name}) > S({dataset_name})',
                    't': statistic,
                    'p-value': pvalue
                })
    df = pd.DataFrame(items)
    return df
    #df.set_index(['S', 'estimator', 'tested hypothesis'])
    
    
def organize_pvalues(df_pvalues, metrics=None):
    metrics = metrics or ['ACC', 'AUC', 'TSS', 'BSS']
    df_ttest = (df_pvalues
        .set_index(['S', 'estimator', 'tested hypothesis'])
        .sort_index() # group multiindex
        .unstack(-1)
        .swaplevel(axis=1)
        .sort_index(level=0, axis=1) # group column multiindex
        .loc[metrics] # sort index
    )
    return df_ttest


def style_pvalues(df_ttest):
    return df_ttest.style.applymap(
        lambda x: 'background-color : yellow' if x<0.05 else '',
        subset=(slice(None), [True, False, True, False]),
    )


def typeset_pvalues(df_ttest):
    df_ttest_print = (
        df_ttest
        .rename(columns={
            'S(fused_sharp) > S(sharp)': '$S_{\texttt{FUSED\_SHARP}}$ $>$ $S_{\texttt{SHARP\_ONLY}}$', # one $$ causes math processing error
            'S(fused_smarp) > S(smarp)': '$S_{\texttt{FUSED\_SMARP}}$ $>$ $S_{\texttt{SMARP\_ONLY}}$',
            'p-value': '$p$-value',
            't': '$t$',
        })
        .rename_axis(
            index=['Metric $S$', 'Estimator'],
            columns=['$H_1$', ''],
        )
    )
    print(typeset(df_ttest_print, escape=False))


def tabulate_pvalues_estimator(runs, metrics=None):
    metrics = metrics or ['ACC', 'AUC', 'TSS', 'HSS', 'BSS']
    items = []
    for dataset_name in ['fused_sharp', 'sharp', 'fused_smarp', 'smarp']:
        for metric in metrics:
            a = runs.loc[get_mask(runs, dataset_name, 'LSTM'), metric].tolist()
            b = runs.loc[get_mask(runs, dataset_name, 'CNN'), metric].tolist()
            statistic, pvalue = paired_ttest(a, b)
            items.append({
                'S': metric,
                'dataset': dataset_name,
                'tested hypothesis': f'S(LSTM) > S(CNN)',
                't': statistic,
                'p-value': pvalue
            })
    df = pd.DataFrame(items)
    return df
    #df.set_index(['S', 'estimator', 'tested hypothesis'])


def organize_pvalues_estimator(df_pvalues_est):
    df_ttest_est = (df_pvalues_est
     .drop(columns='tested hypothesis')
     .set_index(['S', 'dataset'])
     .sort_index()
     .unstack(-1)
     .swaplevel(axis=1)
     .sort_index(level=0, axis=1)
     [['fused_sharp', 'sharp', 'fused_smarp', 'smarp']]
     .rename(columns={
         'fused_sharp': '$\texttt{FUSED\_SHARP}$',
         'sharp': '$\texttt{SHARP\_ONLY}$',
         'fused_smarp': '$\texttt{FUSED\_SMARP}$',
         'smarp': '$\texttt{SMARP\_ONLY}$',
         'p-value': '$p$-value',
         't': '$t$',
     })
     .rename_axis(
         index='Metric $S$',
         columns=['Dataset', ''],
     )
    )
    return df_ttest_est


def style_pvalues_estimator(df_ttest_est):
    return df_ttest_est.style.applymap(
        lambda x: 'background-color : yellow' if x<0.05 else '',
        subset=(slice(None), [True, False] * 4),
    )


def download_figures(runs_raw, dataset_name, seed, estimator_name, output_dir=None):
    import os, shutil

    output_dir = output_dir or 'temp'
    os.makedirs(output_dir, exist_ok=True)

    artifact_uri = runs_raw.loc[
        (runs_raw['tags.dataset_name'] == dataset_name) &
        (runs_raw['params.DATA.SEED'] == str(seed)) &
        (runs_raw['tags.estimator_name'] == estimator_name),
        'artifact_uri'
    ].iloc[0]

    for figure in ['reliability', 'roc', 'ssp']:
        artifact_dir = artifact_uri.replace('file://', '')
        src = os.path.join(artifact_dir, 'test', figure, '0.png')
        dst = os.path.join(output_dir, f'{seed}_{estimator_name}_{dataset_name}_{figure}.png')
        # dst = 'temp/LSTM_fused_sharp_1_ssp.png'
        shutil.copy(src, dst)
        
        
def unstack_reps(runs_raw, index_cols=None, rep_col=None, metric_cols=None):
    #index_cols = index_cols or ['params.dataset0', 'params.estimator0', 'params.criterion']
    #rep_col = rep_col or 'params.seed0'
    #other_cols = ['metrics.tss_over_best']
    df = (runs_raw
          .loc[:, [*index_cols, rep_col, *metric_cols]]
          .set_index(index_cols + [rep_col])
          .unstack(-1)
         )
    return df


def get_labels_probs(query, split, correct_prob=None, return_df=False):
    """
    Args:
        query: 'experiment/run/dataset/seed/val_split/test_split/estimator'
    """
    import os
    import torch
    import pytorch_lightning as pl
    from arnet.modeling.learner import Learner
    from arnet.dataset import ActiveRegionDataModule

    correct_prob = correct_prob or (lambda probs, labels: probs)
    base_exp_name, base_run_name, dataset, seed, val_split, test_split, estimator = query.split('/')
    runs = retrieve(base_exp_name, base_run_name)
    selected = runs.loc[
        (runs['tags.dataset_name'] == dataset) &
        (runs['params.DATA.SEED'] == seed) &
        (runs['params.DATA.VAL_SPLIT'] == val_split) &
        (runs['params.DATA.TEST_SPLIT'] == test_split) &
        (runs['tags.estimator_name'] == estimator)
    ]
    if len(selected) != 1:
        print(f'WARNING: f{len(selected)} runs are selected')
    artifact_uri = selected['artifact_uri'].iloc[0][7:] # remove leading 'file://'
    ckpt_path = selected['tags.checkpoint'].iloc[0]
    ckpt_info = (ckpt_path
     .split('/')[-1]
     .replace('.ckpt', '')
     .replace('-', '=')
     .split('=')
    )
    epoch, step = int(ckpt_info[1]), int(ckpt_info[3])

    # Hotfix for val rus
    if split == 'train':
        csv_full = os.path.join(artifact_uri, 'train_predictions.csv')
        if not os.path.exists(csv_full):
            learner = Learner.load_from_checkpoint(ckpt_path)
            kwargs = learner.cfg.TRAINER.todict()
            # Saved under notebooks/mlruns and notebooks/lightning_logs
            trainer = pl.Trainer(**kwargs)
            dm = ActiveRegionDataModule(learner.cfg)
            _df = dm.df_train
            dataloader = dm.get_dataloader(_df)
            y_prob = trainer.predict(learner, dataloaders=dataloader)
            y_prob = torch.cat(y_prob).detach().cpu().numpy()
            df = _df.assign(prob=y_prob)
            df.to_csv(csv_full)
        df = pd.read_csv(csv_full, index_col=0)
        probs = df[f'prob'].values
        labels = df['label'].values.astype(int)
    elif split == 'val':
        ## Use the DataModules setting to decide if rus val
        #csv_full = os.path.join(artifact_uri, 'validation0', 'val_predictions_full.csv')
        #if not os.path.exists(csv_full):
        #    learner = Learner.load_from_checkpoint(ckpt_path)
        #    kwargs = learner.cfg.TRAINER.todict()
        #    # Saved under notebooks/mlruns and notebooks/lightning_logs
        #    trainer = pl.Trainer(**kwargs)
        #    dm = ActiveRegionDataModule(learner.cfg)
        #    _df = dm.df_vals[0]
        #    dataloader = dm.get_dataloader(_df)
        #    y_prob = trainer.predict(learner, dataloaders=dataloader)
        #    y_prob = torch.cat(y_prob).detach().cpu().numpy()
        #    df = _df.assign(prob=y_prob)
        #    df.to_csv(csv_full)

        ## Use original val_predictions.csv
        csv_full = os.path.join(artifact_uri, 'validation0', 'val_predictions.csv')

        df = pd.read_csv(csv_full, index_col=0)
        #probs = df[f'prob'].values # for val_predictions_full.csv
        probs = df[f'step-{step}'].values # for val_predictions.csv
        labels = df['label'].values.astype(int)
    elif split == 'test':
        csv = os.path.join(artifact_uri, 'validation1', 'val_predictions.csv')
        df = pd.read_csv(csv, index_col=0)
        probs = df[f'step-{step}'].values
        labels = df['label'].values.astype(int)
    probs = correct_prob(probs, labels)
    if return_df:
        return labels, probs, df
    else:
        return labels, probs


def graphical_compare_dataset(
        mlflow_experiment_name,
        mlflow_run_name,
        name_expr,
        name_ctrl,
        folder,
        correct_prob=None,
    ):
    """
    LevelOneData does a simple thing: retrieve predictions and labels.
    Why is the implementation so complicated?
    If MLflow retrieval speed is the iss
    """
    import os
    import matplotlib.pyplot as plt
    from arnet.utils import draw_roc, draw_ssp, draw_reliability_plot
    
    dataset1, estimator1, name1 = name_expr
    dataset2, estimator2, name2 = name_ctrl
    if not os.path.exists(folder):
        os.makedirs(folder)
    labels1, probs1 = zip(*[
        get_labels_probs(
            query=f'{mlflow_experiment_name}/{mlflow_run_name}/{dataset1}/{seed}/None/None/{estimator1}',
            split='test',
            correct_prob=correct_prob
        )
        for seed in range(10)
    ])

    labels2, probs2 = zip(*[
        get_labels_probs(
            query=f'{mlflow_experiment_name}/{mlflow_run_name}/{dataset2}/{seed}/None/None/{estimator2}',
            split='test',
            correct_prob=correct_prob
        )
        for seed in range(10)
    ])

    # Reliability diagram
    fig = draw_reliability_plot(
        labels1,
        probs1,
        name=name1,
    )
    fig = draw_reliability_plot(
        labels2,
        probs2,
        name=name2,
        marker='s',
        fig_ax_ax2=(fig, *fig.axes),
        offset=0.01,
    )
    fig.axes[0].legend(bbox_to_anchor=(0.4, 1), loc='upper center', framealpha=0.4)
    #fig.axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    from matplotlib.ticker import EngFormatter
    fig.axes[1].yaxis.set_major_formatter(EngFormatter())
    fig.set_size_inches(3.9, 3.5)
    fig.tight_layout()
    plt.savefig(os.path.join(folder, 'reliability.pdf'))
    plt.savefig(os.path.join(folder, 'reliability.png'), dpi=300)
    fig_rd = fig

    # SSP: TSS
    fig = draw_ssp(
        labels1,
        probs1,
        name=name1,
        scores=['tss'],
    )
    fig = draw_ssp(
        labels2,
        probs2,
        name=name2,
        scores=['tss'],
        fig_ax=(fig, *fig.axes),
    )
    #fig.axes[0].legend(loc='upper center')
    fig.set_size_inches(3.5, 3.5)
    fig.tight_layout()
    plt.savefig(os.path.join(folder, 'ssp_tss.pdf'))
    plt.savefig(os.path.join(folder, 'ssp_tss.png'), dpi=300)
    fig_ssp_tss = fig
    
    # SSP: HSS
    fig = draw_ssp(
        labels1,
        probs1,
        name=name1,
        scores=['hss'],
    )
    fig = draw_ssp(
        labels2,
        probs2,
        name=name2,
        scores=['hss'],
        fig_ax=(fig, *fig.axes),
    )
    #fig.axes[0].legend(loc='upper center')
    fig.set_size_inches(3.5, 3.5)
    fig.tight_layout()
    plt.savefig(os.path.join(folder, 'ssp_hss.pdf'))
    plt.savefig(os.path.join(folder, 'ssp_hss.png'), dpi=300)
    fig_ssp_hss = fig

    # ROC
    fig = draw_roc(
        labels1,
        probs1,
        name=name1,
    )
    fig = draw_roc(
        labels2,
        probs2,
        name=name2,
        fig_ax=(fig, *fig.axes),
    )
    #fig.axes[0].legend(loc='upper center')
    fig.set_size_inches(3.5, 3.5)
    fig.tight_layout()
    plt.savefig(os.path.join(folder, 'roc.pdf'))
    plt.savefig(os.path.join(folder, 'roc.png'), dpi=300)
    fig_roc = fig
    return fig_rd, fig_ssp_tss, fig_ssp_hss, fig_roc
