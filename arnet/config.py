""" The one-stop reference point for ALL configurable options
Restrictions:
    Argument can't be None
    Cmd line / yaml argument type must match this file
    Default config:
        * Task: Predict MX vs Quiet in future 24 hr
        * Dataset: class-balanced sharp, 24 hr observation
        * Model: 3D CNN
"""
from arnet.utils import CfgNode as CN

cfg = CN()

cfg.DATA = CN()
cfg.DATA.DATABASE = 'datasets/M_Q_24hr'
cfg.DATA.DATASET = 'sharp'
cfg.DATA.AUXDATA = 'datasets/sharp2smarp.npy'
cfg.DATA.BALANCED = True
cfg.DATA.SEED = None
cfg.DATA.FEATURES = [
    'MAGNETOGRAM',
    #'AREA', 'USFLUXL', 'MEANGBL', 'R_VALUE',
    #'FLARE_INDEX',
]
# Input dimensions
cfg.DATA.NUM_FRAMES = 16
cfg.DATA.HEIGHT = 64
cfg.DATA.WIDTH = 128
# Dataloader config
cfg.DATA.BATCH_SIZE = 64
cfg.DATA.NUM_WORKERS = 8
cfg.DATA.TRANSFORMS = [
    'Resize', # 'CenterCropPad'
    #'ValueTransform',
    'Standardize',
    #'Reverse',
]
cfg.DATA.SHRINKAGE = 'log'
cfg.DATA.THRESH = 150
cfg.DATA.IMAGE_MEAN = 0
cfg.DATA.IMAGE_STD = 90

cfg.LEARNER = CN()
cfg.LEARNER.CLASS_WEIGHT = [1, 1] # list or 'balanced', default=None
cfg.LEARNER.LEARNING_RATE = 1e-4 # Don't change it here!
cfg.LEARNER.CHECKPOINT = "" # path to checkpoint file to read from
cfg.LEARNER.PATIENCE = 8
### For visualization
cfg.LEARNER.VIS = CN()
cfg.LEARNER.VIS.GRADCAM_LAYERS = ['convs.conv5']
cfg.LEARNER.VIS.ACTIVATIONS = ['convs.conv1', 'convs.conv2']
cfg.LEARNER.VIS.HISTOGRAM = ['convs.conv3', 'linears.linear1']
### For model
cfg.LEARNER.MODEL = CN()
cfg.LEARNER.MODEL.NAME = 'SimpleC3D'
cfg.LEARNER.MODEL.SETTINGS = 'c3d'

cfg.TRAINER = CN()
cfg.TRAINER.distributed_backend = None #"ddp" #None
cfg.TRAINER.gpus = 1 #2 #AssertionError: Invalid type <class 'NoneType'> for key gpus; valid types = {<class 'float'>,
# If gpus = 2, can't debug inside training_step, where it shows two outputs and gives a warning: WARNING: your terminal doesn't support cursor position requests (CPR).
cfg.TRAINER.fast_dev_run = False
cfg.TRAINER.log_every_n_steps = 1
cfg.TRAINER.track_grad_norm = 0#1
cfg.TRAINER.limit_train_batches = 1.0
cfg.TRAINER.limit_val_batches = 1.0
cfg.TRAINER.limit_test_batches = 1.0
cfg.TRAINER.num_sanity_val_steps = 0
cfg.TRAINER.val_check_interval = 1.0
cfg.TRAINER.check_val_every_n_epoch = 1
cfg.TRAINER.max_epochs = 5
cfg.TRAINER.precision = 16
cfg.TRAINER.default_root_dir = None #'lightning_logs_cnn'
# root_dir defaults to be '.'
cfg.TRAINER.profiler = None
#TODO: a more convenient way to add the signature of pl.Trainer()

cfg.MISC = CN()
cfg.MISC.OUTPUT_DIR = "outputs"
