import os
import time
import argparse
import cProfile, pstats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import wandb
import joblib
import mlflow
from skopt import BayesSearchCV
from skopt.plots import plot_objective
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, make_scorer, roc_curve, auc, plot_confusion_matrix, plot_roc_curve

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

from metrics import tss, hss2, roc_auc_score, get_scores_from_cm


def get_data(filepath):
    df = pd.read_csv(filepath)
    df['flares'].fillna('', inplace=True)
    assert df.isnull().any(axis=None) == False

    if 'sharp' in filepath:
        sharp2smarp = np.load('datasets/sharp2smarp.npy', allow_pickle=True).item()
        for k, v in sharp2smarp.items():
            df[k] = df[k] * v['coef'] + v['intercept']

    X = df[cfg['features']].to_numpy()
    y = df['label'].to_numpy()

    return X, y


def load_dataset(dataset):
    if dataset == 'combined':
        X_train1, y_train1 = get_data('datasets_quiet/smarp/train.csv')
        X_train2, y_train2 = get_data('datasets_quiet/sharp/train.csv')
        X_test1, y_test1 = get_data('datasets_quiet/smarp/test.csv')
        X_test2, y_test2 = get_data('datasets_quiet/sharp/test.csv')

        X_train = np.concatenate((X_train1, X_test1, X_train2))
        y_train = np.concatenate((y_train1, y_test1, y_train2))
        X_test = X_test2
        y_test = y_test2
    elif dataset == 'smarp':
        X_train, y_train = get_data('datasets_quiet/smarp/train.csv')
        X_test, y_test = get_data('datasets_quiet/smarp/test.csv')
    elif dataset == 'sharp':
        X_train, y_train = get_data('datasets_quiet/sharp/train.csv')
        X_test, y_test = get_data('datasets_quiet/sharp/test.csv')
    else:
        raise

    # standardization
    X_mean = X_train.mean(0)
    X_std = X_train.std(0)
    #print(X_mean, X_std)
    Z_train = (X_train - X_mean) / X_std
    Z_test = (X_test - X_mean) / X_std

    if cfg['smoke']:
        N = 1000
        Z_train = Z_train[:N]
        y_train = y_train[:N]
        Z_test = Z_test[:N]
        y_test = y_test[:N]

    return Z_train, Z_test, y_train, y_test


def evaluate(dataset, model, save_dir='outputs'):
    if isinstance(model, str):
        import pickle
        model = pickle.load(model)

    X_train, X_test, y_train, y_test = load_dataset(dataset)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(model, X_test, y_test)
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    mlflow.log_artifact(save_path)
    #mlflow.log_figure(plt.gcf(), 'confusion_matrix_figure.png')
    #plt.show()

    scorer = make_scorer(roc_auc_score, needs_threshold=True)
    auc = scorer(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test) #TODO: mark decision threshold
    plt.savefig(os.path.join(save_dir, 'roc.png'))
    mlflow.log_artifact(os.path.join(save_dir, 'roc.png'))
    mlflow.log_figure(plt.gcf(), 'roc_figure.png')
    #plt.show()

    scores = get_scores_from_cm(cm)
    scores.update({'auc': auc})
    save_path = os.path.join(save_dir, 'best_model_test_scores.md')
    pd.DataFrame(scores, index=[0,]).to_markdown(save_path, tablefmt='grid')

    # Inspect
    estimator = model.best_estimator_['model']
    if isinstance(estimator, DecisionTreeClassifier):
        if estimator.tree_.node_count < 16: # max depth 4
            plot_tree(estimator,
                      feature_names=cfg['features'],
                      filled=True)
            save_path = os.path.join(save_dir, 'tree.png')
            plt.savefig(save_path)
            mlflow.log_artifact(save_path)
            mlflow.log_figure(plt.gcf(), 'tree_figure.png')
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

    return scores


def tune(dataset, Model, param_space, method='grid', save_dir='outputs'):
    X_train, X_test, y_train, y_test = load_dataset(dataset)

    scorer = make_scorer(hss2)
    #scorer = make_scorer(roc_auc_score, needs_threshold=True)

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
                              cv=5,
                              refit=True, # default True
                              verbose=1)
        search.fit(X_train, y_train)
    elif method == 'bayes':
        search = BayesSearchCV(pipe,
                               pipe_space,
                               n_iter=cfg['bayes']['n_iter'], # default 50 # 8 cause out of range
                               scoring=scorer,
                               n_jobs=cfg['bayes']['n_jobs'], # at most n_points * cv jobs
                               n_points=cfg['bayes']['n_points'], # number of points to run in parallel
                               #pre_dispatch default to'2*n_jobs'. Can't be None. See joblib
                               cv=cfg['bayes']['cv'], # if integer, StratifiedKFold is used by default
                               refit=True, # default True
                               verbose=1)
        search.fit(X_train, y_train)
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

    save_path = os.path.join(save_dir, 'cv_results.csv')
    df.to_csv(save_path)
    mlflow.log_artifact(save_path)

    save_path = os.path.join(save_dir, 'cv_results.md')
    df.to_markdown(save_path, tablefmt='grid')
    mlflow.log_artifact(save_path)

    print(f'CV results of {Model.__name__} on {dataset}:')
    print(df.to_markdown(tablefmt='grid'))
    print()

    fig = px.parallel_coordinates(df, color="score_mean",
                                  dimensions=df.columns,
                                  #color_continuous_scale=px.colors.diverging.Tealrose,
                                  #color_continuous_midpoint=2
                                 )
    save_path = os.path.join(save_dir, 'parallel_coordinates.html')
    fig.write_html(save_path)
    mlflow.log_artifact(save_path)
    #fig.show()

    joblib.dump(search, os.path.join(save_dir, 'model.joblib'))
    mlflow.sklearn.log_model(search, 'model')

    return search


def sklearn_main(output_dir='outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    Models = [
        #KNeighborsClassifier,
        #QuadraticDiscriminantAnalysis,
        SGDClassifier,
        #SVC,
        #DecisionTreeClassifier,
        #RandomForestClassifier,
        #ExtraTreesClassifier,
        #AdaBoostClassifier,
        #GradientBoostingClassifier,
        #HistGradientBoostingClassifier,
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
            # priors=None, # By default, the class proportions are inferred from training data
        },
        'DecisionTreeClassifier': {
            'max_depth': [8, 16, 32, 64, None], # default None
            #'min_samples_leaf': (0.000001, 0.01, 'log-uniform'),
            # 1 and 1.0 are different. Default 1
            'class_weight': ['balanced'], # default to None (all classes are assumed to have weight one)
        },
        'RandomForestClassifier': {
            'n_estimators': [100, 300, 1000],
            #'max_depth': [None, 1, 2, 4, 8], # RF doesn't use weak learner
            'class_weight': ['balanced', 'balanced_subsample'], # default to None (all classes are assumed to have weight one)
            'oob_score': [True],
        },
#         'ExtraTreesClassifier': {
#         },
#         'AdaBoostClassifier': {
#         },
#         'GradientBoostingClassifier': {
#         },
        'HistGradientBoostingClassifier': {
            'learning_rate': (0.0001, 0.1, 'log-uniform'),
            'max_iter': [50, 100, 200, 400, 1000],
            'max_depth': [None, 2, 4, 6],
        },
    }

    results = []
    for dataset in ['sharp', 'combined']:
        for Model in Models:
            t_start = time.time()
            run_dir = os.path.join(output_dir, f'{Model.__name__}_{dataset}')
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)

            param_space = distributions[Model.__name__]

            with mlflow.start_run(run_name=cfg['run_name']) as run:
                best_model = tune(dataset, Model, param_space, method='bayes', save_dir=run_dir)
                # Alternatively, param_space = grids[Model.__name__] and use 'grid' method

                scores = evaluate(dataset, best_model, save_dir=run_dir)

                #mlflow.log_param('sampling_strategy', best_model.best_params_['rus__sampling_strategy'])
                mlflow.log_params({k.replace('model__', ''): v for k, v in
                    best_model.best_params_.items() if k.startswith('model__')})
                mlflow.set_tag('estimator_name', Model.__name__)
                mlflow.set_tag('dataset_name', dataset)
                mlflow.log_metrics(scores)
                #mlflow.sklearn.log_model(best_model, 'mlflow_model')

            r = {
                'dataset': dataset,
                'model': Model.__name__,
                'time': time.time() - t_start,
            }
            r.update(scores)
            r.update({
                'params': dict(best_model.best_params_),
            })
            results.append(r)

    results_df = pd.DataFrame(results)
    save_path = os.path.join(output_dir, f'results')
    results_df.to_markdown(save_path+'.md', tablefmt='grid')
    results_df.to_csv(save_path+'.csv')
    print(results_df.to_markdown(tablefmt='grid'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--smoke', action='store_true',
                        help='Smoke test')
    parser.add_argument('-r', '--run_name', default='baseline_quiet',
                        help='MLflow run name')
    args = parser.parse_args()

    cfg = {
        'features': ['AREA', 'USFLUX', 'MEANGBZ', 'R_VALUE', 'FLARE_INDEX'],
        'smoke': args.smoke,
        'run_name': args.run_name,
    }
    if args.smoke:
        cfg.update({
            'experiment_name': 'smoke',
            'output_root': 'outputs_smoke',
            'bayes': {
                'n_iter': 6,
                'n_jobs': 2,
                'n_points': 1,
                'cv': 2,
            },
        })
    else:
        cfg.update({
            'experiment_name': 'experiment',
            'output_root': 'outputs',
            'bayes': {
                'n_iter': 50,
                'n_jobs': 20,
                'n_points': 4,
                'cv': 10,
            },
        })

    mlflow.set_experiment(cfg['experiment_name'])
    output_dir = os.path.join(cfg['output_root'], cfg['run_name'])
    with cProfile.Profile() as p:
        sklearn_main(output_dir)

    pstats.Stats(p).sort_stats('cumtime').print_stats(50)
