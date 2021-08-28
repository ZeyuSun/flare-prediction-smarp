def get_learner(dataset_name, seed: int, estimator_name,
                experiment_name='leaderboard3', run_name='LSTM_CNN'):
    import os
    from mlflow_helper import retrieve
    from arnet.modeling.learner import Learner

    # MLflow run info
    runs = retrieve(experiment_name, run_name)
    runs = runs.loc[
        (runs['tags.dataset_name'] == dataset_name) &
        (runs['params.DATA.SEED'] == str(seed)) &
        (runs['tags.estimator_name'] == estimator_name)
    ]
    if len(runs) > 1:
        print('WARNING: more than 1 runs')
    run_info = runs.iloc[0]
    ckpt_path = run_info['tags.checkpoint']
    # ckpt_path = os.path.join(os.path.dirname(run_info['tags.checkpoint']), 'last.ckpt')
    # print(ckpt_path)

    learner = Learner.load_from_checkpoint(ckpt_path)
    return learner


def inspect_runs(dataset_name, seed: int, estimator_name,
                 experiment_name='leaderboard3', run_name='LSTM_CNN'):
    from arnet.config import cfg
    from arnet.modeling.learner import Learner
    from arnet.dataset import ActiveRegionDataModule
    from mlflow_helper import retrieve, tensorboard

    if not isinstance(dataset_name, list):
        dataset_name = [dataset_name]

    if not isinstance(seed, list):
        seed = [seed]
    seed = [str(s) for s in seed]

    if not isinstance(estimator_name, list):
        estimator_name = [estimator_name]

    # MLflow run info
    runs_raw = retrieve(experiment_name, run_name)
    runs = runs_raw.loc[
        (runs_raw['tags.dataset_name'].isin(dataset_name)) &
        (runs_raw['params.DATA.SEED'].isin(seed)) &
        (runs_raw['tags.estimator_name'].isin(estimator_name)),
        ['tags.dataset_name', 'params.DATA.SEED', 'tags.estimator_name', 'artifact_uri', 'run_id']
    ]

    #print(runs) # output is ugly

    dirs = tensorboard(runs)
    return runs, dirs


def numpy_get_thresh(y_true, y_prob, criterion=None):
    if criterion is None:
        return 0.5

    from sklearn.metrics import roc_curve
    if y_true.sum() == 0:
        print('Return thresh 0.5, because no positive samples in targets, true positive value should be meaningless')
        return 0.5 # ValueError: No positive samples in targets, true positive value should be meaningless
    fpr, tpr, thresholds = roc_curve(y_true, y_prob) #, num_classes=2)
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]  # remove added point
    if criterion == 'tss':
        TSS = tpr - fpr
        return thresholds[TSS.argmax()]

    P = y_true.sum()
    N = len(y_true) - P
    FP, TP = N * fpr, P * tpr
    TN, FN = N - FP, P - TP
    if criterion == 'hss2':
        HSS2 = 2 * (TP * TN - FN * FP) / (P * (FN + TN) + (TP + FP) * N)
        return thresholds[HSS2.argmax()]


def score2prob(scores):
    probs = F.softmax(scores, dim=-1)
    return probs[:,1]


def predict(dataset_name, seed: int, estimator_name,
            experiment_name='leaderboard3', run_name='LSTM_CNN'):
    import torch
    import pytorch_lightning as pl
    from arnet.dataset import ActiveRegionDataModule

    learner = get_learner(dataset_name, seed, estimator_name)
    # hotfix
    learner.cfg.DATA.DATABASE = '/home/zeyusun/work/flare-prediction-smarp/' + str(learner.cfg.DATA.DATABASE)
    learner.cfg.DATA.AUXDATA = '/home/zeyusun/work/flare-prediction-smarp/' + str(learner.cfg.DATA.AUXDATA)

    kwargs = learner.cfg.TRAINER.todict()
    kwargs['default_root_dir'] = 'lightning_logs_dev'
    trainer = pl.Trainer(**kwargs)

    dm = ActiveRegionDataModule(learner.cfg)

    # Predict train/val/test dataset
    dfs = {
        'train': dm.df_train,
        'val': dm.df_vals[0],
        'test': dm.df_test,
    }
    for split in dfs.keys():
        dl = dm.get_dataloader(dfs[split])
        y_prob = trainer.predict(learner, dataloaders=dl)
        y_prob = torch.cat(y_prob).detach().cpu().numpy()
        y_true = dfs[split]['label'].to_numpy()
        thresh = numpy_get_thresh(y_true, y_prob, None) #'tss')
        dfs[split]['prob'] = y_prob
        dfs[split]['pred'] = (y_prob > thresh)
        dfs[split][['label', 'pred']] = dfs[split][['label', 'pred']].astype(int)

    return dfs


def get_learners_cv(dataset_name, seed: int, estimator_name,
                    experiment_name='CNN', run_name='cv'):
    from mlflow_helper import retrieve
    from arnet.modeling.learner import Learner

    # MLflow run info
    runs = retrieve(experiment_name, run_name)
    runs = runs.loc[
        (runs['tags.dataset_name'] == dataset_name) &
        (runs['params.DATA.SEED'] == str(seed)) &
        (runs['tags.estimator_name'] == estimator_name)
    ]
    if len(runs) > 1:
        print('WARNING: more than 1 runs')
    run_info = runs.iloc[0]
    ckpt_path = run_info['tags.checkpoint']

    learner = Learner.load_from_checkpoint(ckpt_path)
    return learner


# def predict_cv(dataset_name, seed: int, estimator_name,
#                experiment_name='CNN', run_name='cv'):
#     import torch
#     import pytorch_lightning as pl
#     from arnet.dataset import ActiveRegionDataModule, CrossValidationDataModule

#     learner = get_learner(dataset_name, seed, estimator_name)
#     # hotfix
#     learner.cfg.DATA.DATABASE = '/home/zeyusun/work/flare-prediction-smarp/' + str(learner.cfg.DATA.DATABASE)
#     learner.cfg.DATA.AUXDATA = '/home/zeyusun/work/flare-prediction-smarp/' + str(learner.cfg.DATA.AUXDATA)

#     kwargs = learner.cfg.TRAINER.todict()
#     kwargs['default_root_dir'] = 'lightning_logs_dev'
#     trainer = pl.Trainer(**kwargs)

#     #dm = ActiveRegionDataModule(learner.cfg)
#     cv_dm = CrossValidationDataModule(learner.cfg)

#     for fold_idx, loaders in
#         c
#     # Predict train/val/test dataset
#     dfs = {
#         'train': dm.df_train,
#         'val': dm.df_val,
#         'test': dm.df_test,
#     }
#     for split in dfs.keys():
#         dl = dm.get_dataloader(dfs[split])
#         y_prob = trainer.predict(learner, dataloaders=dl)
#         y_prob = torch.cat(y_prob).detach().cpu().numpy()
#         y_true = dfs[split]['label'].to_numpy()
#         thresh = numpy_get_thresh(y_true, y_prob, None) #'tss')
#         dfs[split]['prob'] = y_prob
#         dfs[split]['pred'] = (y_prob > thresh)
#         dfs[split][['label', 'pred']] = dfs[split][['label', 'pred']].astype(int)

#     return dfs


def get_transform_from_learner(learner):
    from torchvision.transforms import Compose
    from arnet.transforms import get_transform
    transforms = [get_transform(t, learner.cfg) for t in learner.cfg.DATA.TRANSFORMS]
    transform = Compose(transforms)
    return transform


def get_gradcam_from_learner(learner, target_layer):
    from arnet import utils
    gradcam = utils.GradCAM(learner.model, target_layers=[target_layer],
                            data_mean=0, data_std=1)
    return gradcam


def function(gradcam_layer, seed, dataset):
    dfs = {estimator: predict (dataset, seed, estimator)
           for estimator in ['CNN', 'LSTM']}
