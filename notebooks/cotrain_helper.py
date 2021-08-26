import os
from pathlib import Path
from itertools import product
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch

from arnet.modeling.learner import Learner
from arnet.dataset import ActiveRegionDataModule
from mlflow_helper import retrieve
retrieve = lru_cache(retrieve)
from dashboard_helper import get_learner, inspect_runs, predict, get_transform_from_learner


class LevelOneData:
    """
    * We don't want seed in __init__. Everytime we change seed we have to retrieve runs again.
        * But we already have lru_cache around retrieve? And we need to call get_data in __init__, which needs seed.
    * Seed shouldn't be a state attribute.
    * I'm going to call get_data separately. Atom. Explicit.
    * self.dfs is bound to a certain seed. That's why I set it not as an attr. But we would like the class to include dfs because that is the level one data. So I set it as an attr set by get_data(seed). Should it be a dict indexed by seed?
    """
    def __init__(self, dataset_names, estimator_names, experiment_name='leaderboard3', run_name='val_tss'):
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        if isinstance(estimator_names, str):
            estimator_names = [estimator_names]
        self.selectors = list(product(dataset_names, estimator_names))
        assert len(self.selectors) == 2
        self.runs = retrieve(experiment_name, run_name)

    def get_data(self, seed):
        self.dfs = []
        for s in self.selectors:
            selected = self.runs.loc[
                (self.runs['tags.dataset_name'] == s[0]) &
                (self.runs['params.DATA.SEED'] == str(seed)) &
                (self.runs['tags.estimator_name'] == s[1])
            ]
            if len(selected) > 1:
                print('WARNING: more than 1 runs')
            df_train, df_val, df_test  = get_val_csv(selected.iloc[0])
            self.dfs.append({
                'train': df_train,
                'val': df_val,
                'test': df_test
            })

    def get_split(self, split):
        X = np.vstack([member[split]['prob'] for member in self.dfs]).T
        labels = np.vstack([member[split]['label'] for member in self.dfs]).T
        assert np.all(labels[:,0] == labels[:,1])
        y = labels[:,0]
        return X, y


def get_val_csv(run):
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

    return df_train, df_val, df_test


def get_split(dfs, split):
    df_fig = dfs['LSTM'][split].copy()
    df_fig = df_fig.rename(columns={'prob': 'LSTM prob'})
    df_fig['label'] = df_fig['label'].astype(bool)
    df_fig['CNN prob'] = dfs['CNN'][split]['prob']
    return df_fig


def evaluate(df, alpha=None):
    lstm = ((df['LSTM prob'] > 0.5) ^ df['label']).mean()
    cnn = ((df['CNN prob'] > 0.5) ^ df['label']).mean()
    errors = [lstm, cnn]
    if alpha is not None:
        v2 = [alpha, 1-alpha]
        prob = df[['LSTM prob', 'CNN prob']].to_numpy().dot(v2)
        meta = np.mean((prob > 0.5) ^ df['label'])
        errors.append(meta)
    return errors


def evaluate_naive(X, y, metric=None, alpha=None):
    """X is prob"""
    from arnet.utils.metrics import accuracy

    metric = metric or accuracy
    m0 = metric(y, X[:, 0]>0.5)
    m1 = metric(y, X[:, 1]>0.5)
    if alpha is not None:
        v2 = [alpha, 1-alpha]
        m2 = metric(y, X.dot(v2)>0.5)
    return [m0, m1, m2]

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


# Visualize grad and hessian
# from cotrain_helper import hessian
# alphas = np.linspace(0, 1, 101)
# X, y = X_test, y_test
# plt.plot(alphas, [fun(X, y, a) for a in alphas])
# plt.plot(alphas, [grad(X, y, a) for a in alphas])
# plt.plot(alphas, [hessian(X, y, a) for a in alphas])
# plt.grid()


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

class MetaLearner:
    def __init__(self, step=5e-1, alpha=None):
        self.alpha = alpha or 0.5
        self.step = step

    def fit(self, X, y):
        """
        X, y: level-1 data
        """
        g = lambda alpha: grad(X, y, alpha)
        h = lambda alpha: hessian(X, y, alpha)
        f = lambda alpha: (alpha, fun(X, y, alpha))
        proj = lambda alpha: np.clip(alpha, 0, 1)
        # We can even bisection-search the 1d cvx obj
        #self.alpha, self.out = gd(
        #    g, self.step, proj, self.alpha, fun=f)
        self.alpha, self.out = newton(
            g, h, proj, self.alpha, fun=f)
        return self

    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def predict_proba(self, X):
        v = [self.alpha, 1-self.alpha]
        proba = X.dot(v)
        return proba


# Plot function

def plot_level_one(df, alpha=None):
    #### Option 1: px.scatter
    #fig = px.scatter(df, x='LSTM prob', y='CNN prob', color='label')

    #### Option 2: single go.Scatter (not good)
    #### only 1 trace messes up color order; no legend
    #colors = [px.colors.qualitative.Plotly[i] for i in df['label']]
    #data = go.Scatter(x=df['LSTM prob'], y=df['CNN prob'],
    #                  mode="markers",
    #                  marker=dict(color=colors),
    #                  #line=dict(width=0),
    #                  customdata=df[['prefix', 'arpnum', 't_end', 'AREA']].to_numpy(),
    #                  hovertemplate=(
    #                      "<b>x=%{x}, y=%{y} </b><br><br>" +
    #                      "%{customdata[0]} %{customdata[1]}<br>" +
    #                      "time: %{customdata[2]}<br>" +
    #                      "AREA: %{customdata[3]}"
    #                  )
    #)

    #### Option 3: two go.Scatter
    df_neg = df.loc[~df['label']]
    trace1 = go.Scatter(x=df_neg['LSTM prob'],
                        y=df_neg['CNN prob'],
                        mode="markers",
                        marker=dict(opacity=0.8),
                        name='Negative',
                        customdata=df_neg[['prefix', 'arpnum', 't_end', 'AREA']].to_numpy(),
                        hovertemplate=(
                            "<b>x=%{x}, y=%{y} </b><br><br>" +
                            "%{customdata[0]} %{customdata[1]}<br>" +
                            "time: %{customdata[2]}<br>" +
                            "AREA: %{customdata[3]}"
                        )
    )
    df_pos = df.loc[df['label']]
    trace2 = go.Scatter(x=df_pos['LSTM prob'],
                        y=df_pos['CNN prob'],
                        mode="markers",
                        marker=dict(opacity=0.8),
                        name='Positive',
                        customdata=df_pos[['prefix', 'arpnum', 't_end', 'AREA']].to_numpy(),
                        hovertemplate=(
                            "<b>x=%{x}, y=%{y} </b><br><br>" +
                            "%{customdata[0]} %{customdata[1]}<br>" +
                            "time: %{customdata[2]}<br>" +
                            "AREA: %{customdata[3]}"
                        )
    )
    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(
        xaxis_title='LSTM predicted probability',
        yaxis_title='CNN predicted probability',
        yaxis = dict(
            scaleanchor = 'x',
            tickmode = 'linear',
            tick0 = 0,
            dtick = 0.5,
        )
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


def plot_level_one_naive(X, y, columns=None, alpha=None):
    columns = columns or [None, None]
    fig = px.scatter(x=X[:,0], y=X[:,1], color=y)
    fig.update_layout(
        xaxis_title=columns[0],
        yaxis_title=columns[1],
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
    ###############WIP
    trace1 = go.Scatter(x=df_neg['LSTM prob'],
                        y=df_neg['CNN prob'],
                        mode="markers",
                        marker=dict(opacity=0.8),
                        name='Negative',
                        customdata=df_neg[['prefix', 'arpnum', 't_end', 'AREA']].to_numpy(),
                        hovertemplate=(
                            "<b>x=%{x}, y=%{y} </b><br><br>" +
                            "%{customdata[0]} %{customdata[1]}<br>" +
                            "time: %{customdata[2]}<br>" +
                            "AREA: %{customdata[3]}"
                        )
    )
    df_pos = df.loc[df['label']]
    trace2 = go.Scatter(x=df_pos['LSTM prob'],
                        y=df_pos['CNN prob'],
                        mode="markers",
                        marker=dict(opacity=0.8),
                        name='Positive',
                        customdata=df_pos[['prefix', 'arpnum', 't_end', 'AREA']].to_numpy(),
                        hovertemplate=(
                            "<b>x=%{x}, y=%{y} </b><br><br>" +
                            "%{customdata[0]} %{customdata[1]}<br>" +
                            "time: %{customdata[2]}<br>" +
                            "AREA: %{customdata[3]}"
                        )
    )
    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(
        xaxis_title='LSTM predicted probability',
        yaxis_title='CNN predicted probability',
        yaxis = dict(scaleanchor = 'x')
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


# Plot 0-1 loss
def plot_alpha(df):
    aa = np.linspace(0, 1, 50)
    err = np.array([evaluate(df, a) for a in aa])

    plt.figure(figsize=(4,4))
    plt.plot(aa, err[:,0], label='LSTM')
    plt.plot(aa, err[:,1], label='CNN')
    plt.plot(aa, err[:,2], label='Ensemble')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Error rate')
    plt.ylim([0, 0.1])
    plt.tight_layout()
    #plt.savefig('alpha_val.png')


def plot_alpha_naive(X, y, metric=None, columns=None):
    from arnet.utils.metrics import accuracy

    metric = metric or accuracy
    columns = columns or ['member0', 'member1', 'ensemble']

    aa = np.linspace(0, 1, 50)
    scores = np.array([evaluate_naive(X, y, metric=metric, alpha=a) for a in aa])

    plt.figure(figsize=(4,4))
    plt.plot(aa, scores[:,0], label=columns[0])
    plt.plot(aa, scores[:,1], label=columns[1])
    plt.plot(aa, scores[:,2], label=columns[2])
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('score')
    plt.ylim([0.9, 1])
    plt.tight_layout()


def parse_meta(meta):
    from datetime import datetime
    meta_list = meta.replace('.npy', '').split('_')
    prefix = meta_list[0][:4]
    arpnum = int(meta_list[0][4:])
    t_rec = datetime.strptime(meta_list[1], '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S TAI')
    flares = meta_list[-1]
    title = f'{prefix} {arpnum} @ {t_rec}\nFlares: {flares}'
    return title


def meta_learn(levelone, train=False, output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if train:
        X_train, y_train = levelone.get_split('train')
    X_val, y_val = levelone.get_split('val')
    X_test, y_test = levelone.get_split('test')

    if train:
        fig = plot_level_one_naive(X_train, y_train)
        if output_dir is None:
            fig.show()
        else:
            fig.write_image(str(output_dir / 'data_train.png'))

    # Meta learner training
    ml = MetaLearner(step=0.1)
    ml.fit(X_val, y_val)
    alphas, funs = zip(*ml.out)

    scores = evaluate_naive(X_test, y_test, alpha=ml.alpha)
    if output_dir is None:
        print('Super learning: ', scores)
    else:
        with open(output_dir / 'log.txt', 'a') as fd:
            fd.write(f'Super learning: {scores}\n')

    plt.figure()
    plt.subplot(121); plt.plot(alphas); plt.ylabel(r'$\alpha$')
    plt.subplot(122); plt.plot(funs); plt.ylabel('Cross-entropy loss')
    plt.tight_layout()
    if output_dir is None:
        plt.show()
    else:
        plt.savefig(output_dir / 'convergence_val.png')

    fig = plot_level_one_naive(X_val, y_val, alpha=ml.alpha)
    if output_dir is None:
        fig.show()
    else:
        fig.write_image(str(output_dir / 'data_val.png'))

    # Meta learner evaluation
    ml_test = MetaLearner()
    ml_test = ml_test.fit(X_test, y_test)
    alphas, funs = zip(*ml_test.out)

    scores = evaluate_naive(X_test, y_test, alpha=ml_test.alpha)
    if output_dir is None:
        print('Oracle: ', scores)
    else:
        with open(output_dir / 'log.txt', 'a') as fd:
            fd.write(f'Oracle: {scores}\n')

    plt.figure()
    plt.subplot(121); plt.plot(alphas); plt.ylabel(r'$\alpha$')
    plt.subplot(122); plt.plot(funs); plt.ylabel('Cross-entropy loss')
    plt.tight_layout()
    if output_dir is None:
        plt.show()
    else:
        plt.savefig(output_dir / 'convergence_test.png')

    fig = plot_level_one_naive(X_test, y_test, alpha=ml.alpha)
    add_separating_line(fig, ml_test.alpha, dash='dash')
    if output_dir is None:
        fig.show()
    else:
        fig.write_image(str(output_dir / 'data_test.png'))

    # 0-1 loss
    if train:
        plot_alpha_naive(X_train, y_train)
        if output_dir is None:
            plt.show()
        else:
            plt.savefig(output_dir / 'alpha_train.png')

    plot_alpha_naive(X_val, y_val)
    if output_dir is None:
        plt.show()
    else:
        plt.savefig(output_dir / 'alpha_val.png')

    plot_alpha_naive(X_test, y_test)
    if output_dir is None:
        plt.show()
    else:
        plt.savefig(output_dir / 'alpha_test.png')

    plt.close() # savefig won't close
