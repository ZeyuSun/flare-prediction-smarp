import time
import argparse
import cProfile, pstats
from pathlib import Path
import mlflow
import pytorch_lightning as pl

from arnet import utils
from arnet.dataset import ActiveRegionDataModule
from arnet.modeling.learner import Learner, build_test_logger
from arnet.config import cfg

# TODO: calling getLogger repeatedly somehow creates multiple loggers
logger = utils.setup_logger('outputs')


def train(cfg, dm, resume=False):
    pl.utilities.seed.seed_everything(seed=cfg.DATA.SEED, workers=True)
    callbacks = [
        pl.callbacks.early_stopping.EarlyStopping(
            monitor='validation0/tss',
            patience=cfg.LEARNER.PATIENCE,
            mode='max',
            #verbose=True,
        ),
        pl.callbacks.ModelCheckpoint(
            monitor='validation0/tss',
            save_top_k=1,
            mode='max',
            #verbose=True,
        ),
        pl.callbacks.LearningRateMonitor(
            logging_interval=None,
            log_momentum=True,
        ),
        #pl.callbacks.ModelPruning("l1_unstructured", amount=0.5),
    ]
    # log_hparams in tensorboard
    #tb_logger = pl.loggers.TensorBoardLogger(save_dir, default_hp_metric=False)
    #how to get save_dir before init trainer?

    kwargs = cfg.TRAINER.todict()
    kwargs.setdefault('callbacks', []).extend(callbacks)
    #kwargs['logger'] = tb_logger
    trainer = pl.Trainer(**kwargs)
    if resume:
        learner = Learner.load_from_checkpoint(resume, cfg=cfg)
    else:
        learner = Learner(cfg)
    trainer.validate(learner, datamodule=dm) # mlflow log before training
    trainer.fit(learner, datamodule=dm)
    return trainer.checkpoint_callback.best_model_path


def test(cfg, dm):
    learner = Learner.load_from_checkpoint(cfg.LEARNER.CHECKPOINT, cfg=cfg)
    logger = build_test_logger(learner)
    trainer = pl.Trainer(logger=logger, **cfg.TRAINER.todict())
    trainer.test(learner, datamodule=dm)


def launch(config, modes, resume, opts):
    """Perform training, testing, and/or visualization"""
    logger.info("======== LAUNCH ========")
    global cfg  # If not stated, cfg is seen as local due to in-function assignment.
    if config is not None:
        cfg.merge_from_file(config)
    cfg.merge_from_list(opts)
    # cfg.freeze()

    dm = ActiveRegionDataModule(cfg) # datamodule construction also changes transformation params
    cfg = dm.set_class_weight(cfg)

    mlflow.log_params({key: val
                       for key, val in cfg.flatten().items()
                       if key != 'LEARNER.CHECKPOINT'})
    logger.info(cfg)
    logger.info("{} {} {}".format(
        cfg.DATA.DATABASE,
        config,
        cfg.DATA.DATASET,
    ))

    if 'train' in modes:
        logger.info("======== TRAIN ========")
        cfg.LEARNER.CHECKPOINT = train(cfg, dm, resume)
        mlflow.set_tag('checkpoint', cfg.LEARNER.CHECKPOINT)
        mlflow.log_param('LEARNER.CHECKPOINT', cfg.LEARNER.CHECKPOINT) # update
        logger.info("Checkpoint saved at %s" % cfg.LEARNER.CHECKPOINT)

    if 'test' in modes:
        logger.info("======== TEST ========")
        test(cfg, dm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--smoke', action='store_true',
                        help='Smoke test')
    parser.add_argument('-e', '--experiment_name', default='arnet',
                        help='MLflow experiment name')
    parser.add_argument('-r', '--run_name', default='c3d',
                        help='MLflow run name')
    parser.add_argument('--config', metavar='FILE',
                        help="Path to a yaml formatted config file")
    parser.add_argument('--modes', default='train|test',
                        help="Perform training and/or testing")
    parser.add_argument('--resume', metavar='CHECKPOINT',
                        help="Resume training from checkpoint. Valid only in training mode.")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options. Use dot(.) to indicate hierarchy.")
    args = parser.parse_args()
    args.modes = args.modes.split('|')
    accepted_modes = ['train', 'test']
    if any([m not in accepted_modes for m in args.modes]):
        raise AssertionError('Mode {} is not accepted'.format(args.modes))
    if 'train' not in args.modes and 'LEARNER.CHECKPOINT' not in args.opts:
        raise ValueError('LEARNER.CHECKPOINT must be specified in the absence of training mode.')
    if args.smoke:
        args.experiment_name = 'smoke_arnet'
        args.opts.extend([
            'TRAINER.limit_train_batches', '10',
            'TRAINER.limit_val_batches', '2',
            'TRAINER.limit_test_batches', '2',
            'TRAINER.max_epochs', '1',
            'TRAINER.default_root_dir', 'lightning_logs_dev'
        ])

    mlflow.set_experiment(experiment_name=args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        launch(args.config, args.modes, args.resume, args.opts)


def sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', default='datasets')
    parser.add_argument('-c', '--config_root', default='arnet/configs')
    parser.add_argument('-s', '--smoke', action='store_true')
    parser.add_argument('-e', '--experiment_name', default='leaderboard8')
    parser.add_argument('-r', '--run_name', default='reproduce')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.smoke:
        args.experiment_name = 'smoke_arnet'
        args.opts.extend([
            'TRAINER.limit_train_batches', '2',
            'TRAINER.limit_val_batches', '2',
            'TRAINER.limit_test_batches', '2',
            'TRAINER.max_epochs', '1',
            'TRAINER.default_root_dir', 'lightning_logs_dev'
        ])
        num_seeds = 1
        configs = [Path('arnet/configs').absolute() / f'{c}.yaml' for c in ['MLP', 'LSTM', 'CNN', 'C3D', 'FusionC3D']]
        datasets = ['fused_sharp']
    else:
        num_seeds = 10
        # configs = [c for c in Path(args.config_root).iterdir()]
        configs = [Path('arnet/configs').absolute() / f'{c}.yaml' for c in ['LSTM', 'CNN']]
        datasets = ['sharp', 'fused_sharp', 'smarp', 'fused_smarp']
    t_start = time.time()
    databases = [p for p in Path(args.data_root).iterdir() if p.is_dir()]
    databases = [Path(args.data_root).absolute() / d for d in ['M_Q_24hr']]
    test_splits = [None] #range(5)
    val_splits = [None] #range(5)
    seeds = range(num_seeds)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name):
        for database in databases:
            for balanced in [True]:
                for dataset in datasets:
                    for config in configs:
                        for seed in seeds:
                            for test_split in test_splits:
                                for val_split in val_splits:
                                    opts = [
                                        'DATA.DATABASE', database,
                                        'DATA.DATASET', dataset,
                                        'DATA.BALANCED', balanced,
                                        'DATA.SEED', seed, # used in data rus and training
                                        'DATA.TEST_SPLIT', test_split,
                                        'DATA.VAL_SPLIT', val_split,
                                    ]
                                    run_name = '_'.join([database.name, config.stem, dataset])
                                    with mlflow.start_run(run_name=run_name, nested=True):
                                        tt = time.time()
                                        launch(config, 'train|test', False, args.opts + opts)
                                        mlflow.log_metric('time', time.time() - tt)
                                        mlflow.set_tag('database_name', database.name)
                                        mlflow.set_tag('balanced', balanced)
                                        mlflow.set_tag('estimator_name', config.stem)
                                        mlflow.set_tag('dataset_name', dataset)

    print('Run time: {} s'.format(time.time() - t_start))


if __name__ == '__main__':
    main()
