import os
from pathlib import Path
from itertools import product
from functools import lru_cache
from ipdb import set_trace as breakpoint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, hinge_loss
import requests
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import mlflow
from tqdm import tqdm

from arnet.modeling.learner import Learner
from arnet.dataset import ActiveRegionDataModule
from mlflow_helper import retrieve, paired_ttest, get_labels_probs
retrieve = lru_cache(retrieve)
from dashboard_helper import get_learner, inspect_runs, predict, get_transform_from_learner


# Plot function
def plot_level_one_naive(X, y,
                         X_err=None,
                         axis_titles=None, alpha=None, meta=None):
    axis_titles = axis_titles or [None, None]
    if X_err is None:
        X_err = [None, None]
    else:
        X_err = [X_err[:, 0], X_err[:, 1]]
    if meta is not None:
        df = meta.copy()
        kwargs = {
            #'hover_name': '',
            'hover_data': ['prefix', 'arpnum', 't_end', 'AREA'],
        }
    else:
        df = pd.DataFrame()
        kwargs = {}

    df = (df
          .assign(**{'prob0': X[:, 0]})
          .assign(**{'prob1': X[:, 1]}))
    fig = px.scatter(df, x='prob0', y='prob1',
                     error_x=X_err[0], error_y=X_err[1],
                     color=['Positive' if i else 'Negative' for i in y],
                     opacity=0.8, size_max=3,
                     **kwargs)    

    # Draw separating line
    if alpha is not None:
        add_separating_line(fig, alpha)
    fig.update_layout(
        xaxis_title=axis_titles[0],
        xaxis_range=[-0.1,1.1],
        yaxis_title=axis_titles[1],
        yaxis_scaleanchor='x',
        yaxis_tickmode='linear',
        yaxis_tick0=0,
        yaxis_dtick=0.5,
        yaxis_range=[-0.1,1.1],
        height=300,
        width=450,
        margin=dict(
            l=0, #left margin
            r=0, #right margin
            b=0, #bottom margin
            t=0, #top margin
        )
    )
    return fig


def add_separating_line(fig, alpha, dash='solid'):
    if alpha == 1:
        xx = [0.5, 0.5]
        yy = [-1, 2]
    else:
        intercept = 0.5 / (1-alpha)
        xx = [0, 1]
        yy = [intercept, 1-intercept]
    fig.add_trace(
        go.Scatter(
            x=xx,
            y=yy,
            mode='lines',
            #showlegend=False,
            line=dict(
                dash=dash,
                #color=px.colors.qualitative.Plotly[2]
            ),
            # Color adjustment needed for go.Scatter but not for px.scatter for data points
            #name='Ensemble decision border'
            name=f'alpha = {alpha:.3f}'
        )
    )


def get_split(dfs, split):
    """Used in paracoord"""
    df_fig = dfs['LSTM'][split].copy()
    df_fig = df_fig.rename(columns={'prob': 'LSTM prob'})
    df_fig['label'] = df_fig['label'].astype(bool)
    df_fig['CNN prob'] = dfs['CNN'][split]['prob']
    return df_fig


def get_learner_by_query(query, eval_mode=False, device=None):
    s = {
        'experiment': 'cv',
        'run': 'cv',
        'dataset': 'fused_sharp',
        'seed': '0',
        'val_split': '0',
        'test_split': '0',
        'estimator': 'LSTM',
    }
    for (k, old), new in zip(s.items(), query.split('/')):
        s[k] = new or s[k] # update if specified
    print(s.values())
    runs = retrieve(s['experiment'], s['run'])
    selected = runs.loc[
        (runs['tags.dataset_name'] == s['dataset']) &
        (runs['params.DATA.SEED'] == s['seed']) &
        (runs['params.DATA.VAL_SPLIT'] == s['val_split']) &
        (runs['params.DATA.TEST_SPLIT'] == s['test_split']) &
        (runs['tags.estimator_name'] == s['estimator'])
    ]
    if len(selected) > 1:
        print('WARNING: more than 1 runs')
    ckpt_path = selected['tags.checkpoint'].iloc[0]
    learner = Learner.load_from_checkpoint(ckpt_path)
    if eval_mode:
        learner.eval()
    if device:
        learner.to(device)
    return learner


def get_val_csv(run, get_train: bool, rus=True, discard=True):
    artifact_uri = run['artifact_uri']
    ckpt_path = run['tags.checkpoint']
    ckpt_info = (ckpt_path
     .split('/')[-1]
     .replace('.ckpt', '')
     .replace('-', '=')
     .split('=')
    )
    epoch, step = int(ckpt_info[1]), int(ckpt_info[3])

    # df_val
    # csv format:
    # ,prefix,arpnum,...
    # 0,HARP,338,...
    # 1,HARP,366,...
    if get_train or rus == False:
        learner = Learner.load_from_checkpoint(ckpt_path)
        kwargs = learner.cfg.TRAINER.todict()
        # Saved under notebooks/mlruns and notebooks/lightning_logs
        trainer = pl.Trainer(**kwargs)
        if discard == False:
            learner.cfg.DATA.DATABASE = '/home/zeyusun/work/flare-prediction-smarp/datasets/M_QSL_24hr'
        if rus == False:
            learner.cfg.DATA.BALANCED = False
        dm = ActiveRegionDataModule(learner.cfg)

    def add_prob_col(df_sample):
        dataloader = dm.get_dataloader(df_sample)
        y_prob = trainer.predict(learner, dataloaders=dataloader)
        y_prob = torch.cat(y_prob).detach().cpu().numpy()
        df = df_sample.assign(prob=y_prob)
        return df

    if get_train:
        df_train = add_prob_col(dm.df_train)
    else:
        df_train = None
        
    if rus == False:
        df_val = add_prob_col(dm.df_vals[0])
        df_test = add_prob_col(dm.df_test)
    else:
        df_val = pd.read_csv(Path(artifact_uri) / 'validation0' / 'val_predictions.csv', index_col=0)
        df_val = df_val.rename(columns={f'step-{step}': 'prob'})
        df_val = df_val[[col for col in df_val.columns if 'step-' not in col]]
        
        df_test = pd.read_csv(Path(artifact_uri) / 'validation1' / 'val_predictions.csv', index_col=0)
        df_test = df_test.rename(columns={f'step-{step}': 'prob'})
        df_test = df_test[[col for col in df_test.columns if 'step-' not in col]]

    return df_train, df_val, df_test


# The followings are used for cross-entropy minimization
def grad(X, y, alpha):
    """"""
    v1 = [1, -1]
    v2 = [alpha, 1-alpha]
    num = X.dot(v1)
    den = 1 - y - X.dot(v2)
    result = np.mean(num / den)
    return result


def hessian(X, y, alpha):
    v1 = [1, -1]
    v2 = [alpha, 1-alpha]
    num = X.dot(v1) ** 2
    den = (X.dot(v2) - 1 + y) ** 2
    result = np.mean(num / den)
    return result


def gd(grad, step, proj, x0,
       niter=100, tol=1e-6, fun=None):
    x = x0
    fun = fun or (lambda x: None)
    out = [fun(x)]
    for i in range(1, niter+1):
        x -= step * grad(x)
        x = proj(x)
        out.append(fun(x))
        if i > 5: # diff([x]) = []
            dx = np.diff(np.array(out[-5:]), axis=0)
            if np.abs(dx).sum() < tol:
                break
    return x, out


def newton(grad, hessian, proj, x0,
           niter=100, tol=1e-6, fun=None):
    x = x0
    fun = fun or (lambda x: None)
    out = [fun(x)]
    for i in range(1, niter+1):
        x -= grad(x) / hessian(x)
        x = proj(x)
        out.append(fun(x))
        if i > 5:
            dx = np.diff(np.array(out[-5:]), axis=0)
            if np.abs(dx).sum() < tol:
                break
    return x, out


def fun(X, y, alpha):
    v1 = [1, -1]
    v2 = [alpha, 1-alpha]
    assert 0 <= alpha <= 1, 'alpha = {}'.format(alpha)
    r = X.dot(v2)
    result = -np.mean(np.log(np.where(y, r, 1-r).astype(float)))
    return result


# Meta-learner
def logit(x):
    return np.log(x / (1-x))


# Meta-learner
def logit(x):
    return np.log(x / (1-x))


def get_metrics(y_true, y_prob):
    y_pred = y_prob > 0.5
    y_true = y_true.astype(bool)

    TP = np.sum(np.logical_and(y_true, y_pred))
    TN = np.sum(np.logical_and(~y_true, ~y_pred))
    FP = np.sum(np.logical_and(~y_true, y_pred))
    FN = np.sum(np.logical_and(y_true, ~y_pred))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    pod = TP / (TP + FN)
    far = FP / (FP + TN)
    tss = pod - far

    auc = roc_auc_score(y_true, y_prob)

    y_clim = np.mean(y_true)
    bss = 1 - np.mean((y_prob - y_true)**2) / np.mean((y_prob - y_clim)**2)

    cross_entropy = log_loss(y_true, y_prob, normalize=True)
    # takes care of 0log0 and 1log0

    hinge = np.mean(
        np.maximum(0, 1 - np.where(y_true, y_prob, -y_prob)))
    hinge_2 = hinge_loss(y_true * 2 - 1, y_prob)
    assert(np.abs(hinge - hinge_2) < 1e-10)

    metrics = {
        #'precision': precision,
        #'recall': recall,
        'acc': acc,
        'auc': auc,
        'f1': f1,
        #'tss': tss,
        'bss': bss,
        'cross_entropy': cross_entropy,
        'hinge': hinge,
    }
    return metrics


class MetaLearner:
    def __init__(self,
                 criterion='cross_entropy',
                 mode='min',
                 alpha=0.5):
        """
        Args:
            criterion: 'cross_entropy', 'acc', 'auc', 'tss', 'hss', 'bss'
            mode: 'min' or 'max'
            alpha: parameter for convex combination
        """
        self.criterion = criterion
        self.mode = mode
        self.alpha = alpha 

    def fit(self, X, y):
        """
        X, y: level-1 data
        """
        # We also need self.metrics in cross_entropy
        #if self.criterion == 'cross_entropy':
        #    g = lambda alpha: grad(X, y, alpha)
        #    h = lambda alpha: hessian(X, y, alpha)
        #    f = lambda alpha: (alpha, fun(X, y, alpha))
        #    proj = lambda alpha: np.clip(alpha, 0, 1)
        #    # (1) bisection-search the 1d cvx obj
        #    # (2) gradient descient
        #    #step = 5e-1
        #    #self.alpha, self.out = gd(
        #    #    g, step, proj, self.alpha, fun=f)
        #    # (3) Newton's method
        #    self.alpha, self.out = newton(
        #        g, h, proj, self.alpha, fun=f)
        #else:
        N = 100
        alphas = np.linspace(0, 1, N) #+1)
        self.metrics = []
        for alpha in alphas:
            y_prob = X.dot([alpha, 1-alpha])
            self.metrics.append(get_metrics(y, y_prob))

        selector = {'max': np.argmax, 'min': np.argmin}[self.mode]
        i = selector([m[self.criterion] for m in self.metrics])
        self.i = i
        self.alpha = alphas[i]
        return self

    def predict_proba(self, X):
        v = [self.alpha, 1-self.alpha]
        proba = X.dot(v)
        return proba

    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def scores(self, X, y):
        y_prob = self.predict_proba(X)
        metrics = get_metrics(y, y_prob)
        return metrics

    def evaluate(self, X, y):
        """scores + comparisons with best and avg"""
        metrics = self.scores(X, y)
        m0 = get_metrics(y, X[:, 0])
        m1 = get_metrics(y, X[:, 1])
        best = {}
        avg = {}
        for k in metrics:
            avg[k] = np.mean([m0[k], m1[k]])
            if k in ['cross_entropy', 'hinge']:
                best[k] = np.minimum(m0[k], m1[k])
            elif k in ['acc', 'auc', 'f1', 'bss']:
                best[k] = np.maximum(m0[k], m1[k])
            else:
                raise
        metrics.update({k+'_over_best': metrics[k] - best[k] for k in best})
        metrics.update({k+'_over_avg': metrics[k] - avg[k] for k in avg})
        return metrics

    def inspect(self, what, *args, **kwargs):
        """
        Args:
            what: What to inspect.
                - 'convergence': Training convergence.
                - 'levelone': Wraps plot_level_one_naive to unify the interface
                for meta-learner training and evaluation (and hence static
                method).  We don't want to make it obj-specific because it is
                constantly modified: val has fitted alpha, test has fitted and
                oracle, and train has neither and no obj available.
        Returns:
            fig: allows user to modify the figure though it's been shown or saved

        Note: An example of over-abstraction. 'levelone' is optionally draw with
        multiple alpha. Writing them ad-hoc is shorter, easier, more flexible,
        saving you lots of time coding, debugging, and thinking.
        """
        pass


def meta_learn(queries, train=False, axis_titles=None, run_name='temp', correct_prob=None):
    mlflow.set_experiment('stacking')
    with mlflow.start_run(run_name=run_name, nested=True):
        # Load data
        X, y = {}, {}
        splits = ['train', 'val', 'test'] if train else ['val', 'test']
        for split in splits:
            y[split], X[split] = zip(*[get_labels_probs(q, split, correct_prob) for q in queries])
            y[split], X[split]  = y[split][0], np.stack(X[split]).T

        if train:
            fig = plot_level_one_naive(X['train'], y['train'], axis_titles=axis_titles)
            mlflow.log_figure(fig, 'data_train.html')

        settings = [
            #['log_likelihood', 'max'],
            #['neg_hinge', 'max'],
            ['cross_entropy', 'min'],
            ['hinge', 'min'],
            ['acc', 'max'],
            ['auc', 'max'],
            ['f1', 'max'],
            ['bss', 'max'],
        ]
        
        for setting in settings:
            criterion, mode = setting
            tag = f'{criterion}/'
            with mlflow.start_run(run_name=run_name, # same as parent run_name. Easy to retrieve
                                  nested=True): # The idea was to retrieve all and select
                keys = ['experiment', 'run', 'dataset', 'seed', 'val_split', 'test_split', 'estimator']
                for g, q in enumerate(queries):
                    params = dict(zip(keys, q.split('/')))
                    mlflow.log_params({k+str(g): v for k, v in params.items()})
                mlflow.log_param('criterion', criterion)
                #parameter_names = [
                #    'experiment0',
                #    'experiment1',
                #    'run0',
                #    'run1',
                #    'dataset0',
                #    'dataset1',
                #    'seed0',
                #    'seed1',
                #    'estimator0',
                #    'estimator1',
                #    'criterion',
                #]

                # Meta learner training
                ml = MetaLearner(
                    criterion=criterion,
                    mode=mode,
                )
                ml.fit(X['val'], y['val'])

                mlflow.log_metric('alpha', ml.alpha)
                for i, metrics in enumerate(ml.metrics):
                    mlflow.log_metrics({'all_'+k: v for k, v in metrics.items()}, step=i)

                metrics = ml.evaluate(X['test'], y['test'])
                mlflow.log_metrics(metrics, step=ml.i)

                #fig = ml.inspect('convergence', filename=None)
                #mlflow.log_figure(fig, tag + 'convergence_val.png')

                fig = plot_level_one_naive(X['val'], y['val'], axis_titles=axis_titles)
                add_separating_line(fig, ml.alpha, dash='solid')
                mlflow.log_figure(fig, tag + 'data_val.html') #png doesn't work

                # Meta learner evaluation
                ml_test = MetaLearner(
                    criterion=criterion,
                    mode=mode,
                )
                ml_test = ml_test.fit(X['test'], y['test'])

                for i, metrics in enumerate(ml_test.metrics):
                    mlflow.log_metrics({'all_oracle_'+k: v for k, v in metrics.items()}, step=i)

                metrics = ml_test.evaluate(X['test'], y['test'])
                mlflow.log_metrics({'oracle_' + k: v for k, v in metrics.items()}, step=ml_test.i)

                #fig = ml_test.inspect('convergence', filename=None)
                #mlflow.log_figure(fig, tag + 'convergence_test.png')

                fig = plot_level_one_naive(X['test'], y['test'], axis_titles=axis_titles)
                add_separating_line(fig, ml.alpha, dash='solid')
                add_separating_line(fig, ml_test.alpha, dash='dash')
                mlflow.log_figure(fig, tag + 'data_test.html')
                
                alphas = np.linspace(0, 1, 100)
                val_col, test_col = f'val_{criterion}', f'test_{criterion}'
                df_alpha = pd.DataFrame({
                    'alpha': alphas,
                    val_col: [m[criterion] for m in ml.metrics],
                    test_col: [m[criterion] for m in ml_test.metrics],
                })
                fig = px.line(df_alpha,
                              x='alpha',
                              y=[val_col, test_col],
                              labels={'alpha': r'$\alpha$', 'value': criterion})
                idx = np.argmax(df_alpha[val_col])
                fig.add_trace(go.Scatter(x=[alphas[idx]], y=[ml.metrics[idx][criterion]], marker_size=10, showlegend=False))
                fig.add_vline(x=alphas[idx], line_color=px.colors.qualitative.Plotly[2], line_dash='dot')
                mlflow.log_figure(fig, tag + 'alpha.html')


def run_meta_learn(
        base_experiment_name='leaderboard8', base_run_name='reproduce_2',
        #meta_experiment_name='stacking',
        meta_run_name='reproduce_2'
    ):
    mlflow.set_experiment('stacking') # all runs at one time are collected together to not crowd mlflow ui
    with mlflow.start_run(run_name=meta_run_name, nested=True):
        for dataset_name in tqdm(['sharp', 'fused_sharp', 'smarp', 'fused_smarp']):
            for seed in tqdm(range(10)):
                queries = [
                    f'{base_experiment_name}/{base_run_name}/{dataset_name}/{seed}/None/None/LSTM',
                    f'{base_experiment_name}/{base_run_name}/{dataset_name}/{seed}/None/None/CNN',
                ]
                axis_titles = ['LSTM predicted probability', 'CNN predicted probability']
                meta_learn(queries, train=False, axis_titles=axis_titles, run_name=meta_run_name)
                
                
def retrieve_run(query):
    """
    Args:
        members (List[str]): Each string is of format
                'experiment/run/dataset/seed/val_split/estimator'. If any of the field is
                empty, it defaults to the previous member (or the default
                setting for the first member).
                
    Returns:
        run (pd.Series)
    """
    s = dict(zip(['experiment', 'run', 'dataset', 'seed', 'val_split', 'estimator'],
                 query.split('/')))
    runs = retrieve(s['experiment'], s['run'])
    selected = runs.loc[
        (runs['tags.dataset_name'] == s['dataset']) &
        (runs['params.DATA.SEED'] == s['seed']) &
        (runs['params.DATA.VAL_SPLIT'] == s['val_split']) &
        (runs['tags.estimator_name'] == s['estimator'])
    ]
    if len(selected) > 1:
        print('WARNING: more than 1 runs')
    run = selected.iloc[0]
    return run


@lru_cache
def retrieve_metrics(run_id, metric_key, steps=None):
    api_url = f'http://localhost:5000/api/2.0/mlflow/metrics/get-history?run_id={run_id}&metric_key={metric_key}'
    response = requests.get(api_url)
    resp = response.json()
    if steps is None:
        metrics = [item['value'] for item in resp['metrics']]
    else:
        values = {item['step']: item['value'] for item in resp['metrics']}
        metrics = [values.get(step, None) for step in steps]
    return metrics


def retrieve_metrics_all(run_id, metric_key):
    api_url = f'http://localhost:5000/api/2.0/mlflow/metrics/get-history?run_id={run_id}&metric_key={metric_key}'
    response = requests.get(api_url)
    resp = response.json()
    metrics = resp['metrics']
    return metrics


def meta_learn_show_results(run_name='cv',
                            group_col='params.dataset0',
                            rep_col='params.test_split0',
                            criterion_eval='acc',
                            criteria_opt=None,
                            to_replace=None,
                            to_rename=None,
                            return_dfs=None,
                            folder=None):
    """
    Args:
        run_name: mlflow runName (in experiment 'stacking').
        group_col: column used to group.
        rep_col: repetition column.
        criterion_eval: criterion to use in evaluation.
        criteria_opt: subset of optimization criteria to consider.
        to_replace (dict): a map to replace dataframe values by column. Used in visualization.
        to_rename (dict): a map to rename dataframe columns. Used in visualization.
    """
    # Output dir
    import os
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
        
    # Process arguments
    if isinstance(run_name, str):
        runs = retrieve('stacking', run_name, p=None)
    elif isinstance(run_name, list):
        runs = pd.concat([retrieve('stacking', r, p=None) for r in run_name])
    elif isinstance(run_name, pd.DataFrame):
        runs = run_name
    else:
        raise
    runs = runs[~runs['params.criterion'].isna()] # get rid of the middle generation, who don't have params or metrics
    
    criteria_opt = criteria_opt or [
        'CNN', 'LSTM',
        'AVG', 'BEST',
        #'hinge',
        'cross_entropy',
        'bss', 'auc', 'tss',
        #'acc', 'f1', 'bss', 'auc',
    ]
    
    _delta = to_replace or {}
    to_replace = {
        'params.criterion': {c: c.upper() for c in criteria_opt},
        'params.dataset0': {
            'fused_sharp': 'FUSED_SHARP',
            'fused_smarp': 'FUSED_SMARP',
            'sharp': 'SHARP_ONLY',
            'smarp': 'SMARP_ONLY',
        },
    }
    to_replace.update(_delta)

    _delta = to_rename or {}
    to_rename = {
        'params.criterion': 'Criterion',
        'params.dataset0': 'Dataset',
        f'metrics.{criterion_eval}': criterion_eval.upper(),
        'metrics.alpha': 'alpha',
    }
    to_rename.update(_delta)

    return_dfs = return_dfs or []

    # Collect mlflow records of stacking ensembles
    _df = (runs
           .loc[:, [group_col, 'params.criterion', rep_col, f'metrics.{criterion_eval}_over_best']]
           .set_index([group_col, 'params.criterion', rep_col])
           .sort_index()  # group multiindex
           .unstack(-1)
         )
    ## The above selection rule is abstracted in unstack_reps:
    #index_cols = ['params.dataset0', 'params.criterion']
    #rep_col = 'params.seed0'
    #metric_cols = ['metrics.tss_over_best']
    #_df = unstack_reps(runs, index_cols=index_cols, rep_col=rep_col, metric_cols=metric_cols)
    df_styled = (_df
           .assign(p_val=[paired_ttest(x, np.zeros(len(_df.columns)))[1]
                          for x in _df.values])
           .style
           .background_gradient(subset=['p_val'])
           .background_gradient(subset=[f'metrics.{criterion_eval}_over_best'], cmap='coolwarm', vmin=-0.02, vmax=0.02)
          )

    # Get base learners and baseline ensembles (AVG and BEST)
    rows = []
    for (group_val, rep_val), subdf in runs.groupby([group_col, rep_col]):
        #display(subdf[['params.criterion']])
        run_id = subdf['run_id'].iloc[0]
        spectrum = {
            'val': retrieve_metrics(run_id, f'all_{criterion_eval}'),
            'test': retrieve_metrics(run_id, f'all_oracle_{criterion_eval}'),
        }
        metric_LSTM = spectrum['test'][-1] # 0, alpha=0, CNN?
        metric_CNN = spectrum['test'][0]
        metric_AVG = np.mean([spectrum['test'][0], spectrum['test'][-1]])
        if spectrum['val'][-1] > spectrum['val'][0]:
            model_BEST = 'LSTM'
            metric_BEST = metric_LSTM
        else:
            model_BEST = 'CNN'
            metric_BEST = metric_CNN

        rows.extend([
            {
                group_col: group_val,
                'params.criterion': 'LSTM',
                rep_col: rep_val,
                f'metrics.{criterion_eval}': metric_LSTM
            },
            {
                group_col: group_val,
                'params.criterion': 'CNN',
                rep_col: rep_val,
                f'metrics.{criterion_eval}': metric_CNN
            },
            {
                group_col: group_val,
                'params.criterion': 'AVG',
                rep_col: rep_val,
                f'metrics.{criterion_eval}': metric_AVG
            },
            {
                group_col: group_val,
                'params.criterion': 'BEST',
                rep_col: rep_val,
                f'metrics.{criterion_eval}': metric_BEST,
                'model_BEST': model_BEST,
            },
        ])

    # Combine base learners, baseline ensembles, and stacking ensembles
    columns = [group_col, rep_col, 'params.criterion', f'metrics.{criterion_eval}', 'metrics.alpha']
    df = pd.concat((runs[columns], pd.DataFrame(rows)))
    ## Inspect the selections of BEST
    #df_rows = pd.DataFrame(rows)
    #df_rows = (df_rows
    #           [(df_rows['params.criterion'] == 'BEST')]
    #           .drop(columns='params.criterion')
    #           .set_index(['params.dataset0', 'params.seed0'])
    #           ['model_BEST']
    #           .unstack(-1))
    ## Inspect df in Jupyter notebook
    #with pd.option_context('display.max_rows', None):
    #    display(df
    #     .set_index(['params.dataset0', 'params.criterion', 'params.test_split0'])
    #     [[f'metrics.{criterion}', 'metrics.alpha']]
    #     .unstack(-1)
    #    )
    
    # Prepare for plot
    sorter = dict(zip(criteria_opt, range(len(criteria_opt))))
    df_replaced = (df
                  .replace(to_replace)
                  #.rename(columns=to_rename)
                  .assign(sorter=df['params.criterion'].map(sorter))
                  [df['params.criterion'].isin(criteria_opt)] # after assign since the assigned series has to have the same length
                  .sort_values(by=[group_col, 'sorter', rep_col])
                 )

    # Plot
    fig = px.box(
        df_replaced,
        y=f'metrics.{criterion_eval}',
        x=group_col,
        color='params.criterion',
        #facet_col='Dataset',
        #x=list(reversed(['LSTM', 'CNN', 'BEST', 'AVG', 'tss'])),
        #points='all',
        labels=to_rename,
    )
    if folder:
        fig.update_layout(
           width=1500/1.5,
           height=400/1.2,
        )
        fig.write_image(os.path.join(folder, 'stacking.png'), scale=2)

    # Plot weight alpha
    is_stacking = df_replaced['params.criterion'].isin([
        c.upper() for c in [ # all possible stacking criteria
            'hinge', 'cross_entropy',
            'acc', 'f1',
            'acc', 'tss', 'hss',
            'bss', 'auc'
        ]])
    fig_alpha = px.box(
        df_replaced[is_stacking],
        y='metrics.alpha',
        x=group_col,
        color='params.criterion',
        color_discrete_sequence=px.colors.qualitative.Plotly[4:], # skip 4 baselines
        #facet_col='Dataset',
        #x=list(reversed(['LSTM', 'CNN', 'BEST', 'AVG', 'tss'])),
        labels=to_rename,
        points='all',
    )
    for trace in fig_alpha.data:
        trace.marker.size = 4
    fig_alpha.update_layout(
        #width=1500,
        #height=400,
        yaxis_title=r'$\alpha$', # write_image to png works; write_html and show in jupyter not working.
        yaxis_range=[-0.05, 1.05],
    )
    if folder:
        fig_alpha.update_layout(
           width=1500/1.5,
           height=400/1.3,
        )
        fig_alpha.write_image(os.path.join(folder, 'stacking_weights.png'), scale=2)
    
    dfs_table = {
        '_df': _df,
        'df_styled': df_styled,
        'df': df,
        'df_replaced': df_replaced,
    }
    dfs = {name: dfs_table[name] for name in return_dfs}
    
    return fig, fig_alpha, dfs


def meta_learn_show_instance(
        base_experiment_name='leaderboard8',
        base_run_name='reproduce_2',
        query='smarp/0/acc', #dataset, seed, criterion
        correct_prob=None,
    ):
    dataset, seed, criterion = query.split('/')
    queries = [f'{base_experiment_name}/{base_run_name}/{dataset}/{seed}/None/None/LSTM',
               f'{base_experiment_name}/{base_run_name}/{dataset}/{seed}/None/None/CNN']
    # Load data
    X, y, df = {}, {}, {}
    splits = ['train', 'val', 'test']  # if train else ['val', 'test']
    for split in splits:
        y[split], X[split], df[split] = zip(*[get_labels_probs(q, split, correct_prob, return_df=True)
                                              for q in queries])
        y[split], X[split], df[split] = y[split][0].astype(int), np.stack(X[split]).T, df[split][0]

    # Train
    fig_train = plot_level_one_naive(X['train'], y['train'],
                                     axis_titles=['LSTM predicted probability', 'CNN predicted probability'],
                                     #alpha=ml.alpha,
                                     meta=df['train'])
    # Val
    ml = MetaLearner(criterion=criterion, mode='max')
    ml.fit(X['val'], y['val'])
    fig_val = plot_level_one_naive(X['val'], y['val'],
                                   axis_titles=['LSTM predicted probability', 'CNN predicted probability'],
                                   alpha=ml.alpha, meta=df['val'])
    
    # Test
    ml_test = MetaLearner(criterion=criterion, mode='max')
    ml_test.fit(X['test'], y['test'])
    fig_test = plot_level_one_naive(X['test'], y['test'],
                                    #X_err=X_test_std, # err is too large, especially points near 0.5
                                    axis_titles=['LSTM predicted probability', 'CNN predicted probability'],
                                    alpha=ml.alpha, meta=df['test'])
    add_separating_line(fig_test, ml_test.alpha, dash='dash')
    
    # alpha
    alphas = np.linspace(0, 1, 100)
    df_alpha = pd.DataFrame({
        'alpha': alphas,
        'val_metric': [m[criterion] for m in ml.metrics],
        'test_metric': [m[criterion] for m in ml_test.metrics],
    })
    fig_alpha_grid = px.line(df_alpha, x='alpha', y=['val_metric', 'test_metric'], labels={'value': criterion.upper()})
    idx = np.argmax(df_alpha['val_metric'])
    fig_alpha_grid.add_trace(go.Scatter(x=[alphas[idx]], y=[ml.metrics[idx][criterion]], marker_size=10, showlegend=False))
    fig_alpha_grid.add_vline(x=alphas[idx], line_color=px.colors.qualitative.Plotly[2], line_dash='dot')
    fig_alpha_grid.update_layout(
        height=300,
        width=450,
        margin=dict(l=0,r=0,b=0,t=0),
    )
    return fig_train, fig_val, fig_test, fig_alpha_grid


if __name__ == '__main__':
    # Fit stacking weights
    base_exp_name, base_run_name, meta_run_name = (
        'leaderboard8', 'reproduce_3', 'smoke_test', #'reproduce_3_transfer'
    )
    run_meta_learn(
        base_experiment_name=base_exp_name, base_run_name=base_run_name,
        meta_run_name=meta_run_name,
    )
