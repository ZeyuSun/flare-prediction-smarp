""" The one-stop reference point for ALL configurable options
Restrictions:
    Argument can't be None
    Cmd line / yaml argument type must match this file
"""
from arnet.utils import CfgNode as CN

cfg = CN()

cfg.DATA = CN()
cfg.DATA.DATASET = 'smarp'
cfg.DATA.NUM_CLASSES = 2
# Input dimensions
cfg.DATA.CHANNELS = [0]
cfg.DATA.NUM_FRAMES = 16
cfg.DATA.HEIGHT = 64
cfg.DATA.WIDTH = 128
# Dataloader config
cfg.DATA.BATCH_SIZE = 32
cfg.DATA.NUM_WORKERS = 8

cfg.LEARNER = CN()
cfg.LEARNER.LOSS_TYPE = 'weighted_negative_log_likelihood'
cfg.LEARNER.LOSS_PN_RATIO = 1 # Positive/Negative weight ratio
cfg.LEARNER.LEARNING_RATE = 1e-4 # Don't change it here!
### For visualization
cfg.LEARNER.VIS = CN()
cfg.LEARNER.VIS.GRADCAM_LAYERS = ['convs.conv3']
cfg.LEARNER.VIS.ACTIVATIONS = ['convs.conv3']
cfg.LEARNER.VIS.HISTOGRAM = ['convs.conv3', 'linear1']

cfg.LEARNER.MODEL = CN()
cfg.LEARNER.MODEL.NAME = 'SimpleC3D_v4'
cfg.LEARNER.MODEL.WEIGHTS = "" # path to checkpoint file to read from
cfg.LEARNER.MODEL.SETTINGS = 'c3d'

cfg.TRAINER = CN()
cfg.TRAINER.distributed_backend = None #"ddp" #None
cfg.TRAINER.gpus = 1 #2 #AssertionError: Invalid type <class 'NoneType'> for key gpus; valid types = {<class 'float'>,
# If gpus = 2, can't debug inside training_step, where it shows two outputs and gives a warning: WARNING: your terminal doesn't support cursor position requests (CPR).
cfg.TRAINER.fast_dev_run = False
cfg.TRAINER.log_every_n_steps = 1
cfg.TRAINER.track_grad_norm = 0 #1
cfg.TRAINER.limit_train_batches = 1.0
cfg.TRAINER.limit_val_batches = 1.0
cfg.TRAINER.limit_test_batches = 1.0
cfg.TRAINER.num_sanity_val_steps = 0
cfg.TRAINER.val_check_interval = 1.0
cfg.TRAINER.check_val_every_n_epoch = 1
cfg.TRAINER.max_epochs = 20
cfg.TRAINER.precision = 16
cfg.TRAINER.default_root_dir = 'lightning_logs_c3d' #None
#TODO: a more convenient way to add the signature of pl.Trainer()

cfg.MISC = CN()
cfg.MISC.OUTPUT_DIR = "outputs_c3d"
