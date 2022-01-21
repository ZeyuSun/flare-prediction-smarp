import time
import logging
import argparse
from pathlib import Path
import cProfile, pstats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import mlflow
import graphviz
from skopt import BayesSearchCV
from skopt.plots import plot_objective
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold, LeavePGroupsOut, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, make_scorer, plot_confusion_matrix, plot_roc_curve

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

from metrics import tss, hss2, roc_auc_score, get_scores_from_cm, optimal_tss, draw_ssp
from utils import get_output
from arnet.fusion import get_datasets


def standardize_data(X_train, X_test):
    X_mean = X_train.mean(0)
    X_std = X_train.std(0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    return X_train, X_test


def get_dataset_from_df(df):
    X = df[cfg['features']].to_numpy()
    y = df['label'].to_numpy()
    groups = (df['prefix'] + df['arpnum'].apply(str)).to_numpy()
    return X, y, groups


def get_dataset_numpy(database, dataset, auxdata, balanced=False, seed=None):
    if cfg['smoke']:
        balanced = {0: 50, 1: 50}

    df_train, df_test = get_datasets(database, dataset, auxdata,
                                     balanced=balanced, validation=False, shuffle=True, seed=seed)
    X_train, y_train, g_train = get_dataset_from_df(df_train)
    X_test, y_test, g_test = get_dataset_from_df(df_test)

    X_train, X_test = standardize_data(X_train, X_test)

    return X_train, X_test, y_train, y_test, g_train, g_test


def evaluate(X_test, y_test, model, save_dir='outputs'):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(model, X_test, y_test)
    save_path = save_dir / 'confusion_matrix.png'
    plt.savefig(save_path)
    mlflow.log_artifact(save_path)
    #mlflow.log_figure(plt.gcf(), 'confusion_matrix_figure.png')
    #plt.show()

    scorer = make_scorer(roc_auc_score, needs_threshold=True)
    auc = scorer(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test) #TODO: mark decision threshold
    plt.savefig(save_dir / 'roc.png')
    mlflow.log_figure(plt.gcf(), 'roc.png')
    y_score = get_output(model, X_test)
    tss_opt = optimal_tss(y_test, y_score)
    plt.savefig(save_dir / 'ssp.png')
    mlflow.log_figure(draw_ssp(y_test, y_score), 'ssp.png')
    #plt.show()

    scores = get_scores_from_cm(cm)
    scores.update({
        'auc': auc,
        'tss_opt': tss_opt,
    })
    save_path = save_dir / 'best_model_test_scores.md'
    pd.DataFrame(scores, index=[0,]).to_markdown(save_path, tablefmt='grid')

    # Inspect
    estimator = model.best_estimator_['model']
    if isinstance(estimator, DecisionTreeClassifier):
        dot_data = export_graphviz(estimator, out_file=None,
                                   max_depth=3,
                                   feature_names=cfg['features'],
                                   class_names=True,
                                   filled=True)
        graph = graphviz.Source(dot_data, format='png')
        save_path = save_dir / 'tree_graphviz.png'
        graph.render(save_path)
        mlflow.log_artifact(save_path)
    if isinstance(estimator, RandomForestClassifier):
        # Feature importance based on mean decrease in impurity
        fig, ax = plt.subplots()
        forest_importances = pd.Series(estimator.feature_importances_,
                                       index=cfg['features'])
        std = np.std([tree.feature_importances_ for tree in estimator.estimators_], axis=0)
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        mlflow.log_figure(fig, 'forest_importances.png')
    #from sklearn.inspection import permutation_importance
    #r = permutation_importance(model, X_test, y_test,
    #                           n_repeats=10,
    #                           random_state=0)

    #for i in r.importances_mean.argsort()[::-1]:
    #    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
    #        print(f"{diabetes.feature_names[i]:<8}"
    #              f"{r.importances_mean[i]:.3f}"
    #              f" +/- {r.importances_std[i]:.3f}")

    plt.close('all')
    return scores


def tune(X_train, y_train, groups_train,
         Model, param_space, method='grid', save_dir='outputs'):
    #scorer = make_scorer(hss2)
    scorer = make_scorer(roc_auc_score, needs_threshold=True)

    pipe = Pipeline([
        #('rus', RandomUnderSampler()),
        #('scaler', StandardScaler()), # already did it in loading
        ('model', Model())
    ])
    pipe_space = {'model__' + k: v for k, v in param_space.items()}
    pipe_space.update({
        #'rus__sampling_strategy': [1, 0.5, 0.1]  # desired ratio of minority:majority
        #'rus__sampling_strategy': (0.1, 1.0, 'uniform')
    })

    if method == 'grid':
        search = GridSearchCV(pipe,
                              pipe_space,
                              scoring=scorer,
                              n_jobs=1,
                              cv=GroupKFold(cfg['bayes']['n_splits']),
                              refit=True, # default True
                              verbose=1)
        search.fit(X_train, y_train, groups_train)
    elif method == 'bayes':
        search = BayesSearchCV(pipe,
                               pipe_space,
                               n_iter=cfg['bayes']['n_iter'], # default 50 # 8 cause out of range
                               scoring=scorer,
                               n_jobs=cfg['bayes']['n_jobs'], # at most n_points * cv jobs
                               n_points=cfg['bayes']['n_points'], # number of points to run in parallel
                               #pre_dispatch default to'2*n_jobs'. Can't be None. See joblib
                               cv=GroupKFold(cfg['bayes']['n_splits']), # if integer, StratifiedKFold is used by default
                               refit=True, # default True
                               verbose=0)
        search.fit(X_train, y_train, groups_train)
        # Partial Dependence plots of the (surrogate) objective function
        # Not working for smoke test
        #_ = plot_objective(search.optimizer_results_[0],  # index out of range for QDA? If search space is empty, then the optimizer_results_ has length 1, but in plot_objective, optimizer_results_.models[-1] is called but models is an empty list. This should happen for all n_jobs though. Why didn't I come across it?
        #                   dimensions=list(pipe_space.keys()),
        #                   n_minimum_search=int(1e8))
        #plt.tight_layout()
        #plt.savefig(os.path.join(save_dir, 'parallel_dependence.png'))
        #plt.show()
    else:
        raise

    df = pd.DataFrame(search.cv_results_['params'])
    df = df.rename(columns=lambda p: p.split('__')[1])
    df = df.assign(**{new_k: search.cv_results_[k] for k, new_k in
                   [['mean_fit_time', 'fit_time'],
                    ['std_test_score', 'score_std'],
                    ['mean_test_score', 'score_mean'],
                    ['rank_test_score', 'rank']]})

    save_path = save_dir / 'cv_results.csv'
    df.to_csv(save_path)
    mlflow.log_artifact(save_path)

    save_path = save_dir / 'cv_results.md'
    df.to_markdown(save_path, tablefmt='grid')
    mlflow.log_artifact(save_path)

    fig = px.parallel_coordinates(df, color="score_mean",
                                  dimensions=df.columns,
                                  #color_continuous_scale=px.colors.diverging.Tealrose,
                                  #color_continuous_midpoint=2
                                 )
    save_path = save_dir / 'parallel_coordinates.html'
    fig.write_html(save_path.open(mode='w')) # alternatively, str(path.resolve())
    mlflow.log_artifact(save_path)
    #fig.show()

    joblib.dump(search, save_dir / 'model.joblib')
    mlflow.sklearn.log_model(search, 'model')

    return search, df


def sklearn_main(database_dir):
    """
    We sweep both dataset and model in this function because that's the key comparisons
    made by the paper. Databases, on the other hand, is iterated outside this function.
    """
    Models = [
        #KNeighborsClassifier,
        #QuadraticDiscriminantAnalysis,
        SGDClassifier,
        #SVC,
        #DecisionTreeClassifier,
        RandomForestClassifier,
        #ExtraTreesClassifier,
        #AdaBoostClassifier,
        #GradientBoostingClassifier,
        HistGradientBoostingClassifier,
    ]

    grids = {
        'SGDClassifier': {
            'loss': [
                'hinge', # linear SVM
                'log', # logistic regression
            ],
            'alpha': [1e-6, 1e-4, 1e-2],
            'class_weight': 'balanced', # default to None (all classes are assumed to have weight one)
        },
        'QuadraticDiscriminantAnalysis': {
            # priors=None, # By default, the class proportions are inferred from training data
        },
        'SVC': {
            'C': [0.1, 1, 10],
            'class_weight': [
                {0: 1, 1: 1},
                {0: 1, 1: 2},
                {0: 1, 1: 10},
            ],
        },
        'DecisionTreeClassifier': {
            'max_depth': [1, 2, 4, 8], # default None
            'min_samples_leaf': [1, 0.00001, 0.0001, 0.001, 0.01], # 1 and 1.0 are different. Default 1
            'class_weight': 'balanced', # default None (all classes are assumed to have weight one)
        },
        'RandomForestClassifier': {
            'n_estimators': [10, 100, 1000],
            'max_depth': [None, 2, 4, 8],  # weak learners
            #'min_samples_split': 2,
            'class_weight': ['balanced', 'balanced_subsample'],
        },
        'ExtraTreesClassifier': {
        },
        'AdaBoostClassifier': {
        },
        'GradientBoostingClassifier': {
        },
        'HistGradientBoostingClassifier': {
        },
        #'XGBClassifier': {},
    }

    distributions = {
        'SGDClassifier': {
            'loss': [
                #'hinge', # linear SVM
                'log', # logistic regression
            ],
            'alpha': (1e-6, 1e-1, 'log-uniform'),
            'class_weight': ['balanced'], # default to None (all classes are assumed to have weight one)
        },
        'QuadraticDiscriminantAnalysis': {
            'reg_param': [0],  # BayesSearchCV require
            # priors=None, # By default, the class proportions are inferred from training data
        },
        'DecisionTreeClassifier': {
            'max_depth': [8, 16, 32, 64, None], # default None
            #'min_samples_leaf': (0.000001, 0.01, 'log-uniform'),
            # 1 and 1.0 are different. Default 1
            'class_weight': ['balanced'], # default to None (all classes are assumed to have weight one)
        },
        'RandomForestClassifier': {
            'n_estimators': [300], #[50, 100, 300], 300 better than 50 and 100
            #'max_depth': [None, 1, 2, 4, 8], # RF doesn't use weak learner
            'class_weight': ['balanced', 'balanced_subsample'], # default to None (all classes are assumed to have weight one)
            'oob_score': [True],
        },
        'ExtraTreesClassifier': {
            'n_estimators': [100, 300, 1000],
        },
        'AdaBoostClassifier': {
            'n_estimators': [50],
            'learning_rate': [1],
        },
        'GradientBoostingClassifier': {
            'learning_rate': [0.1],
        },
        'HistGradientBoostingClassifier': {
            'learning_rate': (0.0001, 0.1, 'log-uniform'),
            'max_iter': [50, 100, 200, 400, 1000],
            'max_depth': [None, 2, 4, 6],
        },
    }

    results = []
    for dataset in ['smarp', 'sharp', 'fused_smarp', 'fused_sharp']:
        for balanced in [True]:
            for cfg['seed'] in range(5):
                dataset_blc = dataset + '_' + ('balanced' if balanced else 'raw')
                X_train, X_test, y_train, y_test, groups_train, _ = get_dataset_numpy(
                    database_dir, dataset, cfg['auxdata'], balanced=balanced, seed=cfg['seed'])
                # # Visualize processed train and test splits
                # from eda import plot_selected_samples
                # title = database_dir.name + ' ' + dataset_blc
                # fig = plot_selected_samples(X_train, X_test, y_train, y_test, cfg['features'],
                #                             title=title)
                # fig.show()
                # continue
                for Model in Models:
                    t_start = time.time()
                    param_space = distributions[Model.__name__]

                    run_name = '_'.join([database_dir.name, dataset_blc, Model.__name__])
                    run_dir = Path(cfg['output_dir']) / run_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    with mlflow.start_run(run_name=run_name, nested=True) as run:

                        best_model, df = tune(X_train, y_train, groups_train,
                                              Model, param_space, method='bayes',
                                              save_dir=run_dir)
                        # Alternatively, param_space = grids[Model.__name__] and use 'grid' method
                        print(f'\nCV results of {Model.__name__} on {database_dir} {dataset_blc}:')
                        print(df.to_markdown(tablefmt='grid'))

                        scores = evaluate(X_test, y_test, best_model, save_dir=run_dir)

                        #mlflow.log_param('sampling_strategy', best_model.best_params_['rus__sampling_strategy'])
                        mlflow.log_params({k.replace('model__', ''): v for k, v in
                            best_model.best_params_.items() if k.startswith('model__')})
                        mlflow.set_tag('database_name', database_dir.name)
                        mlflow.set_tag('dataset_name', dataset)
                        mlflow.set_tag('balanced', balanced)
                        mlflow.set_tag('estimator_name', Model.__name__)
                        mlflow.set_tag('seed', cfg['seed'])
                        mlflow.log_metrics(scores)
                        #mlflow.sklearn.log_model(best_model, 'mlflow_model')

                    r = {
                        'database': database_dir.name,
                        'dataset': dataset_blc,
                        'model': Model.__name__,
                        'time': time.time() - t_start,
                        'seed': cfg['seed'],
                    }
                    r.update(scores)
                    r.update({
                        'params': dict(best_model.best_params_),
                    })
                    results.append(r)

    results_df = pd.DataFrame(results)
    save_path = Path(cfg['output_dir']) / f'{database_dir.name}_results.md'
    results_df.to_markdown(save_path, tablefmt='grid')
    results_df.to_csv(save_path.with_suffix('.csv'))
    print(results_df.to_markdown(tablefmt='grid'))


def test_seed():
    np.random.seed(0)
    a = np.random.randint(0, 65536, 10)
    assert np.all(a == [2732, 43567, 42613, 52416, 45891, 21243, 30403, 32103, 41993, 57043])
    np.random.seed(None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', default='datasets')
    parser.add_argument('-a', '--auxdata', default='datasets/sharp2smarp.npy')
    parser.add_argument('-s', '--smoke', action='store_true')
    parser.add_argument('-e', '--experiment_name', default='leaderboard2')
    parser.add_argument('-r', '--run_name', default='sklearn')
    parser.add_argument('-o', '--output_dir', default='outputs')
    parser.add_argument('--seed', default=0)
    args = parser.parse_args()

    cfg = {
        'features': ['AREA', 'USFLUXL', 'MEANGBL', 'R_VALUE'],
        'bayes': {
            'n_iter': 10, # light computation until the final stage
            'n_jobs': 20,
            'n_points': 4,
            'n_splits': 5,
        },
    }
    cfg.update(vars(args))
    if args.smoke:
        cfg.update({
            'experiment_name': 'smoke',
            'output_dir': 'outputs_smoke',
            'bayes': {
                'n_iter': 6,
                'n_jobs': 2,
                'n_points': 1,
                'n_splits': 2,
            },
        })

    test_seed()

    t_start = time.time()
    mlflow.set_experiment(cfg['experiment_name'])
    with mlflow.start_run(run_name=cfg['run_name']) as run:
        databases = [p for p in Path(cfg['data_root']).iterdir() if p.is_dir()]
        databases = [Path(cfg['data_root']) / d for d in [
            'M_Q_24hr',
            'M_QS_24hr',
        ]]
        logging.info(databases)
        for database in databases:
            sklearn_main(database)

    print('Run time: {} s'.format(time.time() - t_start))
