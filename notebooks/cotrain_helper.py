import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dashboard_helper import get_learner, inspect_runs, predict, get_transform_from_learner


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


def parse_meta(meta):
    from datetime import datetime
    meta_list = meta.replace('.npy', '').split('_')
    prefix = meta_list[0][:4]
    arpnum = int(meta_list[0][4:])
    t_rec = datetime.strptime(meta_list[1], '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S TAI')
    flares = meta_list[-1]
    title = f'{prefix} {arpnum} @ {t_rec}\nFlares: {flares}'
    return title
