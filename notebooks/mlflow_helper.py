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


def retrieve(experiment_name, parent_run_name, p=0):
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
    else:
        print('Select iloc {} from \n{}'.format(
            p,
            parent_runs[['start_time', 'tags.mlflow.runName', 'tags.mlflow.source.git.commit']]))
        parentRunId = parent_runs['run_id'].iloc[p]

    runs = runs.loc[(runs['tags.mlflow.parentRunId'] == parentRunId) &
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
            .agg(lambda s: ufloat(s.mean(), s.std())) #['mean', 'std'])
            .unstack(-1).T
            #.sort_values('database', axis=1, key=extract_hours)
            #.round(4)
            .applymap('{:.3f}'.format)
        )
    else:
        df = (runs
            .groupby(by)
            .agg('mean')
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


def typeset(df):
    df_latex = df.to_latex(multicolumn_format='c')
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
            a = runs.loc[get_mask(runs, 'fused_'+dataset_name, estimator_name), metric].tolist()
            b = runs.loc[get_mask(runs, dataset_name, estimator_name), metric].tolist()
            print(metric, paired_ttest(a, b))