import pandas as pd
from uncertainties import ufloat
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri('file:///home/zeyusun/work/flare-prediction-smarp/mlruns')
client = MlflowClient()


def get_columns(name):
    tags = {
        #'tags.database_name': 'database',
        'tags.dataset_name': 'dataset',
        #'tags.balanced': 'balanced',
        'tags.estimator_name': 'estimator',
        #'tags.seed': 'seed',
    }
    if name == 'arnet':
        metrics = {'metrics.test/' + m: m.upper() for m in [
            'auc',
            'tss',
            #'hss2',
            #'precision',
            #'recall',
        ]}
    elif name == 'sklearn':
        metrics = {'metrics.' + m: new_m.upper() for m, new_m in [
            ['auc', 'auc'],
            ['tss_opt', 'tss'],
            #'tss',
            #'hss2',
            #'precision',
            #'recall',
        ]}
    else:
        raise
    columns = {**tags, **metrics}
    return columns


def retrieve(experiment_name, parent_run_name):
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
    elif len(parent_runs) == 1:
        parentRunId = parent_runs['run_id'].item()
    else:
        print('Select the first from \n{}'.format(
            parent_runs[['start_time', 'tags.mlflow.runName', 'tags.mlflow.source.git.commit']]))
        parentRunId = parent_runs['run_id'].iloc[0]

    runs = runs.loc[(runs['tags.mlflow.parentRunId'] == parentRunId) &
                    (runs['status'] == 'FINISHED')]
    return runs


def select(runs, columns, rows):
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


def organize(runs, columns=None, std=False):
    columns = columns or ['dataset', 'estimator']
    # sort:
    extract_hours = lambda s: s.str.split('_').str[2].str.replace('hr', '').astype(int)

    if std:
        df = (runs
            .groupby(columns)
            .agg(lambda s: ufloat(s.mean(), s.std())) #['mean', 'std'])
            .unstack(-1).T
            #.sort_values('database', axis=1, key=extract_hours)
            #.round(4)
            .applymap('{:.3f}'.format)
        )
    else:
        df = (runs
            .groupby(columns)
            .agg('mean')
            .unstack(-1).T
            #.sort_values('database', axis=1, key=extract_hours)
            .round(4)
            #.applymap('{:.3f}'.format)
        )
    return df


def style(runs, columns=None):
    columns = columns or ['dataset', 'estimator']
    df = organize(runs, columns=columns, std=False)
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
    dirs = ','.join([f'{idx}:{tb[idx]}' for idx in tb.index])
    return dirs