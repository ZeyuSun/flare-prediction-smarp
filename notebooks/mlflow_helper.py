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


def retrieve(experiment_name, parent_run_name, p=None):
    """p=0: Union[int, list, tuple, slice] doesn't work: Invalid Syntax"""
    """list/slice is not hashable by lru_cache"""
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
    print('Select iloc {} from \n{}'.format(
        p,
        parent_runs[['start_time', 'tags.mlflow.runName']])) #, 'tags.mlflow.source.git.commit']]))
        # may not be a git repo. Do not add it in.

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


def tabulate_pvalues_estimator(runs):
    items = []
    for dataset_name in ['fused_sharp', 'sharp', 'fused_smarp', 'smarp']:
        for metric in ['ACC', 'AUC', 'TSS', 'HSS', 'BSS']:
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


def run_test():
    # Arguments:
    experiment_name, run_name = 'leaderboard3', 'val_tss_2'
    datasets = ['sharp', 'fused_sharp', 'smarp', 'fused_smarp']
    seeds = [str(i) for i in range(5,10)]
    estimators = ['LSTM', 'CNN']

    from functools import lru_cache
    from itertools import product
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    import torch
    import pytorch_lightning as pl
    from tqdm import tqdm

    from arnet import utils
    from arnet.dataset import ActiveRegionDataModule
    from arnet.modeling.learner import Learner
    from cotrain_helper import get_learner_by_query

    runs_raw = retrieve(experiment_name, run_name)
    runs = []
    for dataset, seed, estimator in tqdm(product(datasets, seeds, estimators)):
        selected = runs_raw.loc[
            (runs_raw['tags.dataset_name'] == dataset) &
            (runs_raw['params.DATA.SEED'] == seed) &
            (runs_raw['tags.estimator_name'] == estimator)
        ]

        ckpt_path = selected['tags.checkpoint'].iloc[0]
        learner = Learner.load_from_checkpoint(ckpt_path)
        learner.eval()
        learner.to('cuda:0')
        # hotfix
        learner.cfg.DATA.NUM_WORKERS = 8

        kwargs = learner.cfg.TRAINER.todict()
        kwargs['default_root_dir'] = 'lightning_logs_dev'
        trainer = pl.Trainer(**kwargs)

        dm = ActiveRegionDataModule(learner.cfg)
        df = dm.df_test
        dl = dm.get_dataloader(df)

        y_prob = trainer.predict(learner, dataloaders=dl)
        y_prob = torch.cat(y_prob).detach().cpu().numpy()
        y_true = df['label'].to_numpy()
        df['prob'] = y_prob
        df['pred'] = (y_prob > 0.5)
        df[['label', 'pred']] = df[['label', 'pred']].astype(int)
        metrics, _, _ = utils.get_metrics_probabilistic(
            df['label'].values,
            df['prob'].values,
            criterion=None,
        )
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics.update({
            'dataset': dataset,
            'seed': seed,
            'estimator': estimator,
        })
        runs.append(metrics)

    ## Metric statistics
    df_runs = pd.DataFrame(runs)
    df_runs = df_runs.rename(columns={
        'accuracy': 'ACC',
        'precision': 'PRECISION',
        'recall': 'RECALL',
        'f1': 'F1',
        'auc': 'AUC',
        'tss': 'TSS',
        'hss2': 'HSS',
        'bss': 'BSS',
    })
    #df_results = organize(df_runs, std=True)
    """Typeset can be cumbersome and subject to change. And the format is only important to writing paper, not reproducing results. Therefore, put it in a function and call it outside"""
    #print(df_results)

    ## Paired t-tests of datasets
    # df_ttest = tabulate_pvalues(df_runs)
    # Typeset is bad
    # df_ttest = (df_ttest
    #  .set_index(['S', 'estimator', 'tested hypothesis'])
    #  .sort_index() # group multiindex
    #  .unstack(-1)
    #  .swaplevel(axis=1)
    #  .sort_index(level=0, axis=1) # group column multiindex
    # ).loc[['ACC', 'AUC', 'TSS', 'HSS', 'BSS']]
    # print(typeset(df_ttest))

    ## Paired t-tests of models

    ## Stacking
    """
    Always do hierarchical. Seperate the experiments that don't belong together. You always have the option to organize the experiments with a higher level function.
    """
    # for dataset in datasets:
    #     for seed in seeds:
    #         members = [
    #             f'{experiment_name}/{run_name}/{dataset}/{seed}/0/0/LSTM',
    #             f'//////CNN'
    #         ]
    #         axis_titles = ['LSTM predicted probability', 'CNN predicted probability']
    #         levelone = LevelOneData(members, get_train=True)
    #         meta_learn(levelone, train=True, axis_titles=axis_titles, run_name='FN_bug_reproduce')

    return df_runs
