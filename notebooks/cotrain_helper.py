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
    v1 = [1, -1]
    v2 = [alpha, 1-alpha]
    num = X.dot(v1)
    den = 1 - y - X.dot(v2)
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
        if np.abs(np.diff(out[-3:])).sum() < tol:
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
        f = lambda alpha: (alpha, fun(X, y, alpha))
        proj = lambda alpha: np.clip(alpha, 0, 1)
        self.alpha, self.out = gd(
            g, self.step, proj, self.alpha, fun=f)
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
    trace1 = go.Scatter(x=df.loc[~df['label'], 'LSTM prob'],
                        y=df.loc[~df['label'], 'CNN prob'],
                        mode="markers",
                        name='Negative',
                        customdata=df[['prefix', 'arpnum', 't_end', 'AREA']].to_numpy(),
                        hovertemplate=(
                            "<b>x=%{x}, y=%{y} </b><br><br>" +
                            "%{customdata[0]} %{customdata[1]}<br>" +
                            "time: %{customdata[2]}<br>" +
                            "AREA: %{customdata[3]}"
                        )
    )
    trace2 = go.Scatter(x=df.loc[df['label'], 'LSTM prob'],
                        y=df.loc[df['label'], 'CNN prob'],
                        mode="markers",
                        name='Positive',
                        customdata=df[['prefix', 'arpnum', 't_end', 'AREA']].to_numpy(),
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
        intercept = 0.5 / (1-alpha)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[intercept, 1-intercept],
                mode='lines',
                showlegend=False,
                #line=dict(color=px.colors.qualitative.Plotly[2]),
                # Color adjustment needed for go.Scatter but not for px.scatter for data points
                #name='Ensemble decision border'
            )
        )
    fig.update_layout(
        yaxis = dict(scaleanchor = 'x'),
        xaxis_range=[-0.1,1.1],
        yaxis_range=[-0.1,1.1],
        height=300,
        width=400,
        margin=dict(
            l=0, #left margin
            r=0, #right margin
            b=0, #bottom margin
            t=0, #top margin
        )
    )
    return fig


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