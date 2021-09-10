import os
from pathlib import Path
from itertools import product
from functools import lru_cache

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

from arnet.modeling.learner import Learner
from arnet.dataset import ActiveRegionDataModule
from mlflow_helper import retrieve
retrieve = lru_cache(retrieve)
from dashboard_helper import get_learner, inspect_runs, predict, get_transform_from_learner


# Plot function
def plot_level_one_naive(X, y, axis_titles=None, alpha=None, meta=None):
    axis_titles = axis_titles or [None, None]
    if meta is not None:
        kwargs = {
            #'hover_name': '',
            'hover_data': ['prefix', 'arpnum', 't_end', 'AREA'],
        }
    else:
        kwargs = {}

    df = (meta
          .assign(**{'prob0': X[:, 0]})
          .assign(**{'prob1': X[:, 1]}))
    fig = px.scatter(df, x='prob0', y='prob1', color=y, **kwargs)    
    fig.update_layout(
        xaxis_title=axis_titles[0],
        yaxis_title=axis_titles[1],
        yaxis_scaleanchor='x',
        yaxis_tickmode='linear',
        yaxis_tick0=0,
        yaxis_dtick=0.5,
    )

    # Draw separating line
    if alpha is not None:
        add_separating_line(fig, alpha)
    fig.update_layout(
        yaxis = dict(scaleanchor = 'x'),
        xaxis_range=[-0.1,1.1],
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


def get_val_csv(run, get_train: bool):
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
    df_val = pd.read_csv(Path(artifact_uri) / 'validation0' / 'val_predictions.csv')
    df_val = df_val.rename(columns={f'step-{step}': 'prob'})
    df_val = df_val[[col for col in df_val.columns if 'step-' not in col]]

    # df_test
    df_test = pd.read_csv(Path(artifact_uri) / 'validation1' / 'val_predictions.csv')
    df_test = df_test.rename(columns={f'step-{step}': 'prob'})
    df_test = df_test[[col for col in df_test.columns if 'step-' not in col]]

    if get_train:
        learner = Learner.load_from_checkpoint(ckpt_path)
        kwargs = learner.cfg.TRAINER.todict()
        # Saved under notebooks/mlruns and notebooks/lightning_logs
        trainer = pl.Trainer(**kwargs)
        dm = ActiveRegionDataModule(learner.cfg)

        # df_train
        dl_train = dm.get_dataloader(dm.df_train)
        y_prob = trainer.predict(learner, dataloaders=dl_train)
        y_prob = torch.cat(y_prob).detach().cpu().numpy()
        df_train = dm.df_train.assign(prob=y_prob)
    else:
        df_train = None

    return df_train, df_val, df_test


class LevelOneData:
    """Level-1 data class"""
    def __init__(self, members=None, get_train=False):
        """
        Args:
            members (List[str]): Each string is of format
                'experiment/run/dataset/seed/estimator'. If any of the field is
                empty, it defaults to the previous member (or the default
                setting for the first member).
            get_train: Whether to retrieve the train set, which is a bit more
                time consumuing.
        """
        self.members = members
        self.dfs = []
        # selector s
        s = {
            'experiment': 'leaderboard3',
            'run': 'val_tss',
            'dataset': 'sharp',
            'seed': '0',
            'estimator': 'LSTM',
        }
        self.selectors = []
        for member in members:
            for (k, old), new in zip(s.items(), member.split('/')):
                s[k] = new or s[k] # update if specified
            self.selectors.append(s.copy())
            print(s.values())
            runs = retrieve(s['experiment'], s['run'])
            selected = runs.loc[
                (runs['tags.dataset_name'] == s['dataset']) &
                (runs['params.DATA.SEED'] == s['seed']) &
                (runs['tags.estimator_name'] == s['estimator'])
            ]
            if len(selected) > 1:
                print('WARNING: more than 1 runs')
            df_train, df_val, df_test  = get_val_csv(
                selected.iloc[0],
                get_train,
            )
            self.dfs.append({
                'train': df_train,
                'val': df_val,
                'test': df_test
            })

    def get_split(self, split, return_df=False):
        X = np.vstack([member[split]['prob'] for member in self.dfs]).T
        
        labels = np.vstack([member[split]['label'] for member in self.dfs]).T
        assert np.all(labels[:,0] == labels[:,1])
        y = labels[:,0]
        
        cols = [c for c in self.dfs[0][split].columns if c != 'prob']
        dataframes = [member[split][cols] for member in self.dfs]
        df = dataframes[0]
        assert(all([df.equals(dataframe) for dataframe in dataframes]))
        
        if return_df:
            return X, y, df
        else:
            return X, y


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

    TP = np.sum(np.logical_and(y_true, y_pred))
    TN = np.sum(np.logical_and(~y_true, ~y_pred))
    FP = np.sum(np.logical_and(~y_true, y_pred))
    FN = np.sum(np.logical_and(y_true, ~y_pred))

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
        'acc': acc,
        'tss': tss,
        'auc': auc,
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
        alphas = np.linspace(0, 1, N)
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
            elif k in ['acc', 'auc', 'tss', 'bss']:
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

def meta_learn(levelone, train=False, axis_titles=None, run_name='temp'):
    mlflow.set_experiment('stacking')
    with mlflow.start_run(run_name=run_name):
        # Load data
        if train:
            X_train, y_train, df_train = levelone.get_split('train', return_df=True)
        X_val, y_val, df_val = levelone.get_split('val', return_df=True)
        X_test, y_test, df_test = levelone.get_split('test', return_df=True)

        if train:
            fig = plot_level_one_naive(X_train, y_train, axis_titles=axis_titles, meta=df_train)
            mlflow.log_figure(fig, 'data_train.html')

        settings = [
            ['cross_entropy', 'min'],
            ['hinge', 'min'],
            ['acc', 'max'],
            ['auc', 'max'],
            ['tss', 'max'],
            ['bss', 'max'],
        ]
        
        for setting in settings:
            criterion, mode = setting
            tag = f'{criterion}/'
            with mlflow.start_run(run_name=run_name, # same as parent run_name. Easy to retrieve
                                  nested=True):
                mlflow.log_params({
                    'experiment0': levelone.selectors[0]['experiment'],
                    'experiment1': levelone.selectors[1]['experiment'],
                    'run0': levelone.selectors[0]['run'],
                    'run1': levelone.selectors[1]['run'],
                    'dataset0': levelone.selectors[0]['dataset'],
                    'dataset1': levelone.selectors[1]['dataset'],
                    'seed0': levelone.selectors[0]['seed'],
                    'seed1': levelone.selectors[1]['seed'],
                    'estimator0': levelone.selectors[0]['estimator'],
                    'estimator1': levelone.selectors[1]['estimator'],
                    'criterion': criterion,
                })

                # Meta learner training
                ml = MetaLearner(
                    criterion=criterion,
                    mode=mode,
                )
                ml.fit(X_val, y_val)

                mlflow.log_metric('alpha', ml.alpha)
                for i, metrics in enumerate(ml.metrics):
                    mlflow.log_metrics({'all_'+k: v for k, v in metrics.items()}, step=i)

                metrics = ml.evaluate(X_test, y_test)
                mlflow.log_metrics(metrics, step=ml.i)

                #fig = ml.inspect('convergence', filename=None)
                #mlflow.log_figure(fig, tag + 'convergence_val.png')

                fig = plot_level_one_naive(X_val, y_val, axis_titles=axis_titles, meta=df_val)
                add_separating_line(fig, ml.alpha, dash='solid')
                mlflow.log_figure(fig, tag + 'data_val.html') #png doesn't work

                # Meta learner evaluation
                ml_test = MetaLearner(
                    criterion=criterion,
                    mode=mode,
                )
                ml_test = ml_test.fit(X_test, y_test)

                for i, metrics in enumerate(ml_test.metrics):
                    mlflow.log_metrics({'all_oracle_'+k: v for k, v in metrics.items()}, step=i)

                metrics = ml_test.evaluate(X_test, y_test)
                mlflow.log_metrics({'oracle_' + k: v for k, v in metrics.items()}, step=ml_test.i)

                #fig = ml_test.inspect('convergence', filename=None)
                #mlflow.log_figure(fig, tag + 'convergence_test.png')

                fig = plot_level_one_naive(X_test, y_test, axis_titles=axis_titles, meta=df_test)
                add_separating_line(fig, ml.alpha, dash='solid')
                add_separating_line(fig, ml_test.alpha, dash='dash')
                mlflow.log_figure(fig, tag + 'data_test.html')


def retrieve_metrics(run_id, metric_key):
    api_url = f'http://localhost:5000/api/2.0/mlflow/metrics/get-history?run_id={run_id}&metric_key={metric_key}'
    response = requests.get(api_url)
    resp = response.json()
    metrics = [m['value'] for m in resp['metrics']]
    return metrics


if __name__ == '__main__':
    for dataset_name in ['sharp', 'fused_sharp', 'smarp', 'fused_smarp']:
        for seed in range(5):
            members = [
                f'leaderboard3/val_tss/{dataset_name}/{seed}/LSTM',
                f'////CNN'
            ]
            axis_titles = ['LSTM predicted probability', 'CNN predicted probability']
            levelone = LevelOneData(members, get_train=True)
            meta_learn(levelone, train=True, axis_titles=axis_titles, run_name='estimator')

    for dataset_name in ['sharp', 'fused_sharp', 'smarp', 'fused_smarp']:
        for seed in range(5, 10):
            members = [
                f'leaderboard3/val_tss_2/{dataset_name}/{seed}/LSTM',
                f'////CNN'
            ]
            axis_titles = ['LSTM', 'CNN']
            levelone = LevelOneData(members, get_train=True)
            meta_learn(levelone, train=True, axis_titles=axis_titles, run_name='estimator_2')

    # Figure remaking
    #members = [
    #    f'leaderboard3/val_tss_2/fused_sharp/8/LSTM',
    #    f'////CNN'
    #]
    #axis_titles = ['LSTM predicted probability', 'CNN predicted probability']
    #levelone = LevelOneData(members, get_train=True)
    #meta_learn(levelone, train=True, axis_titles=axis_titles, run_name='estimator_2_seed_8')
