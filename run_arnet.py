import argparse
import cProfile, pstats
import mlflow
import pytorch_lightning as pl


from arnet import utils
from arnet.dataset import ActiveRegionDataModule
from arnet.modeling.learner import Learner, build_test_logger
from arnet.config import cfg


def train(cfg, dm, resume=False):
    trainer = pl.Trainer(**cfg.TRAINER.todict())
    if resume:
        learner = Learner.load_from_checkpoint(cfg.LEARNER.CHECKPOINT, cfg=cfg)
    else:
        learner = Learner(cfg)
    trainer.fit(learner, datamodule=dm)
    return trainer.checkpoint_callback.best_model_path


def test(cfg, dm):
    learner = Learner.load_from_checkpoint(cfg.LEARNER.CHECKPOINT, cfg=cfg)
    logger = build_test_logger(learner, testmode='test')
    trainer = pl.Trainer(logger=logger, **cfg.TRAINER.todict())
    trainer.test(learner, datamodule=dm)


def visualize(cfg, dm):
    learner = Learner.load_from_checkpoint(cfg.LEARNER.CHECKPOINT, cfg=cfg)
    logger = build_test_logger(learner, testmode='visualize')
    trainer = pl.Trainer(logger=logger, **cfg.TRAINER.todict())

    learner.testmode = dm.testmode = 'visualize_predictions'
    trainer.test(learner, datamodule=dm)

    learner.testmode = dm.testmode = 'visualize_features'
    trainer.test(learner, datamodule=dm)


def main(args):
    """Perform training, testing, and/or visualization"""
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        if args.config is not None:
            cfg.merge_from_file(args.config)
        cfg.merge_from_list(args.opts)
        # cfg.freeze()
        mlflow.log_params(cfg.flatten())

        logger = utils.setup_logger(cfg.MISC.OUTPUT_DIR)
        logger.info(cfg)

        dm = ActiveRegionDataModule(cfg)

        if 'train' in args.modes:
            logger.info("======== TRAIN ========")
            if args.resume:
                cfg.LEARNER.CHECKPOINT = args.resume
            cfg.LEARNER.CHECKPOINT = train(cfg, dm)

        if 'test' in args.modes:
            logger.info("======== TEST ========")
            test(cfg, dm)

        #TODO: visualization for LSTM
        #if 'visualize' in args.modes:
        #    logger.info("======== VISUALIZE ========")
        #    visualize(cfg, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--smoke', action='store_true',
                        help='Smoke test')
    parser.add_argument('-e', '--experiment_name', default='experiment',
                        help='MLflow experiment name')
    parser.add_argument('-r', '--run_name', default='arnet',
                        help='MLflow run name')
    parser.add_argument('--config', metavar='FILE',
                        help="Path to a yaml formatted config file")
    parser.add_argument('--modes', default='train|test|visualize',
                        help="Perform training, testing, and/or visualization")
    parser.add_argument('--resume', metavar='CHECKPOINT',
                        help="Resume training from checkpoint. Valid only in training mode.")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options. Use dot(.) to indicate hierarchy.")
    args = parser.parse_args()
    args.modes = args.modes.split('|')
    accepted_modes = ['train', 'test', 'visualize']
    if any([m not in accepted_modes for m in args.modes]):
        raise AssertionError('Mode {} is not accepted'.format(args.modes))
    if 'train' not in args.modes and 'LEARNER.CHECKPOINT' not in args.opts:
        raise ValueError('LEARNER.CHECKPOINT must be specified in the absence of training mode.')
    if args.smoke:
        args.experiment_name = 'arnet_smoke'
        args.opts.extend([
            'TRAINER.limit_train_batches', '10',
            'TRAINER.limit_val_batches', '2',
            'TRAINER.limit_test_batches', '2',
            'TRAINER.max_epochs', '1',
            'TRAINER.default_root_dir', 'lightning_logs_dev'
        ])

    with cProfile.Profile() as p:
        main(args)

    pstats.Stats(p).sort_stats('cumtime').print_stats(50)
