import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


sharp2smarp = np.load('datasets/sharp2smarp.npy', allow_pickle=True).item()
FEATURES = ['AREA', 'USFLUX', 'MEANGBZ', 'R_VALUE', 'FLARE_INDEX']
EXPERIMENT_NAME ='experiment'
mlflow.set_experiment(EXPERIMENT_NAME)
RUN_NAME = 'baseline'


def get_data(filepath):
    df = pd.read_csv(filepath)
    df['flares'].fillna('', inplace=True)
    assert df.isnull().any(axis=None) == False

    if 'sharp' in filepath:
        for k, v in sharp2smarp.items():
            df[k] = df[k] * v['coef'] + v['intercept']
    
    X = df[FEATURES].to_numpy()
    y = df['label'].to_numpy()

    return X, y

def load_dataset(dataset):
    if dataset == 'combined':
        X_train1, y_train1 = get_data('datasets/smarp/train.csv')
        X_train2, y_train2 = get_data('datasets/sharp/train.csv')
        X_test1, y_test1 = get_data('datasets/smarp/test.csv')
        X_test2, y_test2 = get_data('datasets/sharp/test.csv')

        X_train = np.concatenate((X_train1, X_test1, X_train2))
        y_train = np.concatenate((y_train1, y_test1, y_train2))
        X_test = X_test2
        y_test = y_test2
    elif dataset == 'smarp':
        X_train, y_train = get_data('datasets/smarp/train.csv')
        X_test, y_test = get_data('datasets/smarp/test.csv')
    elif dataset == 'sharp':
        X_train, y_train = get_data('datasets/sharp/train.csv')
        X_test, y_test = get_data('datasets/sharp/test.csv')
    else:
        raise
    
    # random undersampling
    #X_train, y_train = rus(X_train, y_train)
    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    #print(y_train.mean(), y_test.mean())

    # standardization
    X_mean = X_train.mean(0)
    X_std = X_train.std(0)
    #print(X_mean, X_std)
    Z_train = (X_train - X_mean) / X_std
    Z_test = (X_test - X_mean) / X_std
    
    return Z_train, Z_test, y_train, y_test

def evaluate(dataset, model, save_dir=None):
    if isinstance(model, str):
        import pickle
        model = pickle.load(model)
    
    X_train, X_test, y_train, y_test = load_dataset(dataset)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plot_confusion_matrix(model, X_test, y_test)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        mlflow.log_artifact(os.path.join(save_dir, 'confusion_matrix.png'))
        mlflow.log_figure(plt.gcf(), 'confusion_matrix_figure.png')
    #plt.show()
    
    scorer = make_scorer(roc_auc_score, needs_threshold=True)
    auc = scorer(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test) #TODO: mark decision threshold
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'roc.png'))
        mlflow.log_artifact(os.path.join(save_dir, 'roc.png'))
        mlflow.log_figure(plt.gcf(), 'roc_figure.png')
    #plt.show()

    scores = get_scores_from_cm(cm)
    scores.update({'auc': auc})
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'best_model_test_scores.md')
        pd.DataFrame(scores, index=[0,]).to_markdown(save_path, tablefmt='grid')

    # Inspect
    estimator = model.best_estimator_['model']
    if isinstance(estimator, DecisionTreeClassifier):
        plot_tree(estimator,
                  feature_names=FEATURES,
                  filled=True)
        save_path = os.path.join(save_dir, 'tree.png')
        plt.savefig(save_path)
        mlflow.log_artifact(save_path)
        mlflow.log_figure(plt.gcf(), 'tree_figure.png')
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


def tune(dataset, Model, param_space, method='grid', save_dir=None):
    X_train, X_test, y_train, y_test = load_dataset(dataset)

    scorer = make_scorer(hss2)
    #scorer = make_scorer(roc_auc_score, needs_threshold=True)
    
    pipe = Pipeline([
        ('rus', RandomUnderSampler()),
        #('scaler', StandardScaler()), # already did it in loading
        ('model', Model())
    ])
    
    pipe_space = {'model__' + k: v for k, v in param_space.items()}
    pipe_space.update({
        'rus__sampling_strategy': [  # desired ratio of minority:majority
            1,
            0.5,
            0.1]
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
                               n_iter=20, # default 50 # 8 cause out of range
                               scoring=scorer,
                               n_jobs=20, # at most n_points * cv jobs
                               n_points=4, # number of points to run in parallel
                               #pre_dispatch=2*njobs by default
                               cv=5, # if integer, StratifiedKFold is used by default
                               refit=True, # default True
                               verbose=1)
        search.fit(X_train, y_train)
        _ = plot_objective(search.optimizer_results_[0],  # index out of range for QDA? If search space is empty, then the optimizer_results_ has length 1, but in plot_objective, optimizer_results_.models[-1] is called but models is an empty list. This should happen for all n_jobs though. Why didn't I come across it?
                           dimensions=list(pipe_space.keys()), # ["C", "degree", "gamma", "kernel"],
                           n_minimum_search=int(1e8)) # Partial Dependence plots of the objective function
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'parallel_dependence.png'))
        #plt.show()
    
    else:
        raise
    
    import plotly.express as px
    df = pd.DataFrame(search.cv_results_)
    df = pd.DataFrame(list(search.cv_results_['params']))
    df = df.assign(**{k: search.cv_results_[k]
                      for k in ['mean_fit_time', 'std_test_score', 'mean_test_score']})
    if save_dir is not None:
        df.to_csv(os.path.join(save_dir, 'cv_results.csv'))
        
    fig = px.parallel_coordinates(df, color="mean_test_score",
                                  dimensions=df.columns,
                                  #color_continuous_scale=px.colors.diverging.Tealrose,
                                  #color_continuous_midpoint=2
                                 )
    if save_dir is not None:
        filepath = os.path.join(save_dir, 'parallel_coordinates.html')
        fig.write_html(filepath)
        mlflow.log_artifact(filepath)
    #fig.show()
    
    if save_dir is not None:
        joblib.dump(search, os.path.join(save_dir, 'model.joblib'))
    
    print(f'CV results of {Model.__name__} on {dataset}:')
    cv_results = {
        'score mean': search.cv_results_['mean_test_score'],
        'score std': search.cv_results_['std_test_score'],
        'rank': [int(i) for i in search.cv_results_['rank_test_score']],
        # BayesSearchCV rank uses numpy.int32, which is not json serializable
    }
    trials = search.cv_results_['params']
    cv_results.update({p.split('__')[1]: [t[p] for t in trials] for p in trials[0]})
    mlflow.log_dict(cv_results, 'cv_results.json')

    cv_df = pd.DataFrame(cv_results)
    cv_file = os.path.join(save_dir, 'cv_results.md')
    cv_df.to_markdown(cv_file, tablefmt='grid')
    print(cv_df.to_markdown(tablefmt='grid'))
    mlflow.log_artifact(cv_file)

    return search


def sklearn_main(output_dir='outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    Models = [
        #KNeighborsClassifier,
        #QuadraticDiscriminantAnalysis,
        SGDClassifier,
        #SVC,
        DecisionTreeClassifier,
        RandomForestClassifier,
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
        },
        'QuadraticDiscriminantAnalysis': {},
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
        },
        'RandomForestClassifier': {
            'n_estimators': [10, 100, 1000],
            'max_depth': [None, 2, 4, 8],  # weak learners
            #'min_samples_split': 2,
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
                'hinge', # linear SVM
                'log', # logistic regression
            ],
            'alpha': (1e-6, 1e-2, 'log-uniform'),
        },
        'QuadraticDiscriminantAnalysis': {},
        'DecisionTreeClassifier': {
            'max_depth': [1, 2, 4, 8, 16], # default None
            #'min_samples_leaf': (0.000001, 0.01, 'log-uniform'),
            # 1 and 1.0 are different. Default 1
        },
        'RandomForestClassifier': {
            'n_estimators': [10, 100, 1000],
            'max_depth': [None, 1, 2, 4, 8],
            # Split a node if # of samples in the node >= min_samples_split and
            # # of samples in each children after split >= min_samples_leaf (??)
            #'min_samples_leaf': [1, 2],
            #'min_samples_split': [4, 5, 6],
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
            model_dir = os.path.join(output_dir, f'{Model.__name__}_{dataset}')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            param_space = distributions[Model.__name__]

            with mlflow.start_run(run_name=RUN_NAME) as run:
                best_model = tune(dataset, Model, param_space, method='bayes', save_dir=model_dir)
                # Alternatively, param_space = grids[Model.__name__] and use 'grid' method

                scores = evaluate(dataset, best_model, save_dir=model_dir)

                mlflow.log_param('sampling_strategy', best_model.best_params_['rus__sampling_strategy'])
                mlflow.log_params({k.replace('model__', ''): v for k, v in
                    best_model.best_params_.items() if k.startswith('model__')})
                mlflow.set_tag('estimator_name', Model.__name__)
                mlflow.set_tag('dataset_name', dataset)
                mlflow.log_metrics(scores)
                #mlflow.sklearn.log_model(best_model, 'mlflow_model')

            r = {
                'dataset': dataset,
                'model': Model.__name__,
                'params': dict(best_model.best_params_),
            }
            r.update(scores)
            results.append(r)
            
            print('***********************************************')
            print(pd.DataFrame(results).to_markdown(tablefmt='grid'))

    print('***********************************************')
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, f'results')
    results_df.to_markdown(results_path+'.md', tablefmt='grid')
    results_df.to_csv(results_path+'.csv')

    
if __name__ == '__main__':
    import cProfile, pstats
    with cProfile.Profile() as p:
        sklearn_main(f'outputs/{RUN_NAME}')
    
    pstats.Stats(p).sort_stats('cumtime').print_stats(50)
