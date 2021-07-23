from copy import deepcopy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection
from arnet.dataset import CrossValidationDataModule


class CrossValidationTrainer:
    """
    Adapted from: https://github.com/PyTorchLightning/pytorch-lightning/issues/839#issuecomment-817658495
    """
    def __init__(self, *args, **kwargs):
        self.trainer_args = args
        self.trainer_kwargs = kwargs

    @staticmethod
    def update_modelcheckpoint(model_ckpt_callback, fold_idx):
        _default_filename = '{epoch}-{step}'
        _suffix = f'_fold{fold_idx}'
        if model_ckpt_callback.filename is None:
            new_filename = _default_filename + _suffix
        else:
            new_filename = model_ckpt_callback.filename + _suffix
        setattr(model_ckpt_callback, 'filename', new_filename)

    def update_logger(self, trainer: Trainer, fold_idx: int):
        if fold_idx == 0:
            self.logger_version = trainer.logger.version
        trainer.logger._version = f'version_{self.logger_version}_{fold_idx}'

    def fit(self, model: LightningModule, cv_datamodule: CrossValidationDataModule):
        splits = cv_datamodule.get_splits()
        for fold_idx, loaders in enumerate(splits):
            # loaders = (train_dataloader, val_dataloader)

            # Clone model & instantiate a new trainer:
            _model = deepcopy(model)
            trainer = Trainer(*self.trainer_args, **self.trainer_kwargs)

            # Update loggers and callbacks:
            self.update_logger(trainer, fold_idx)
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    self.update_modelcheckpoint(callback, fold_idx)

            # Fit:
            trainer.fit(_model, *loaders)
