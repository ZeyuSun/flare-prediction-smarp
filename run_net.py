import os
from datetime import datetime, timedelta
import argparse
import cProfile, pstats
import pandas as pd
import mlflow
import drms
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms
import pytorch_lightning as pl

from arnet.data.datamodule import ARVideoDataModule as _ARVideoDataModule
from arnet.run_net import train, test, visualize
from arnet import utils
from arnet import const
from config import cfg
from data import query


SPLIT_DIRS = {
    'HARP': '/home/zeyusun/work/flare-prediction-smarp/datasets/sharp/',
    'TARP': '/home/zeyusun/work/flare-prediction-smarp/datasets/smarp/',
}
DATA_DIRS = {
    'HARP': '/data2/SHARP/image/',
    'TARP': '/data2/SMARP/image/',
}
SERIES = {
    'HARP': 'hmi.sharp_cea_720s',
    'TARP': 'su_mbobra.smarp_cea_96m',
}


class ARVideoDataset(Dataset):
    def __init__(self, df_sample, transform=None):
        self.df_sample = df_sample
        self.df_sample['flares'].fillna('', inplace=True)
        self.transform = transform

    def __len__(self):
        return len(self.df_sample)

    def __getitem__(self, idx):
        current_sample = self.df_sample.iloc[idx]
        t_start = current_sample['t_start']
        t_end = current_sample['t_end']
        prefix = current_sample['prefix']
        arpnum = current_sample['arpnum']
        flares = current_sample['flares']

        # video
        video = self.load_transform(prefix, arpnum, t_start, t_end)

        # label
        label = int(current_sample['label'])

        # meta
        t_start_str = datetime.strptime(t_start, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')
        largest_flare = max(flares.split('|')) #WARNING: X10+
        meta = f'{prefix}{arpnum:06d}_{t_start_str}_H0_W0_{largest_flare}.npy'
        return video, label, meta

    def load_transform(self, prefix, arpnum, t_start, t_end):
        t_steps = pd.date_range(drms.to_datetime(t_start),
                                drms.to_datetime(t_end),
                                freq='96min').strftime('%Y%m%d_%H%M%S_TAI')
        filenames = (f"{SERIES[prefix]}.{arpnum}.{t}.magnetogram.fits"
                     for t in t_steps)
        filepaths = [os.path.join(DATA_DIRS[prefix], f'{arpnum:06d}', filename)
                     for filename in filenames]
        filepaths = [p if os.path.exists(p) else filepaths[i+1]
                     for i, p in enumerate(filepaths)]
        video = query(filepaths)
        video = torch.from_numpy(video)
        video = torch.unsqueeze(video, 0) # C,T,H,W
        if self.transform:
            video = self.transform(video)
        if prefix == 'HARP':
            pass # accounted for in fits_open
            #video = F.interpolate(video, scale_factor=1/4, mode='nearest')
        return video


class ARVideoDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        T, H, W = cfg.DATA.NUM_FRAMES, cfg.DATA.HEIGHT, cfg.DATA.WIDTH
        self.transform = transforms.Compose([
            utils.CenterCropPad3D(target_size=(None, H, W)),
            #utils.ValueTransform(),
            transforms.Normalize(mean=0, std=const.STD),
        ])
        self.dims = (1, T, H, W)
        self.num_classes = self.cfg.DATA.NUM_CLASSES
        self.testmode = 'test'

        smarp_train = pd.read_csv(os.path.join(SPLIT_DIRS['TARP'], 'train.csv'))
        smarp_test  = pd.read_csv(os.path.join(SPLIT_DIRS['TARP'], 'test.csv'))
        sharp_train = pd.read_csv(os.path.join(SPLIT_DIRS['HARP'], 'train.csv'))
        sharp_test  = pd.read_csv(os.path.join(SPLIT_DIRS['HARP'], 'test.csv'))
        if self.cfg.DATA.DATASET == 'sharp':
            self.df_train = sharp_train
            self.df_val = sharp_test
            self.df_test = sharp_test
        elif self.cfg.DATA.DATASET == 'smarp':
            self.df_train = smarp_train
            self.df_val = smarp_test
            self.df_test = smarp_test
        elif self.cfg.DATA.DATASET == 'combined':
            self.df_train = pd.concat((smarp_train, smarp_test, sharp_train)).reset_index(drop=True)
            self.df_val = sharp_test
            self.df_test = sharp_test
        else:
            raise
        self.df_train = self.df_train.sample(frac=1)
        self.df_vis = sharp_test.iloc[:10]

    def train_dataloader(self):
        dataset = ARVideoDataset(self.df_train,
                                 transform=self.transform)
        #sampler = RandomSampler(dataset, len(dataset) // 2)
        loader = DataLoader(dataset,
                            batch_size=self.cfg.DATA.BATCH_SIZE,
                            #sampler=sampler,
                            drop_last=True,
                            num_workers=self.cfg.DATA.NUM_WORKERS,
                            pin_memory=True)
        return loader

    def val_dataloader(self):
        dataset = ARVideoDataset(self.df_val,
                                 transform=self.transform)
        loader = DataLoader(dataset,
                            batch_size=self.cfg.DATA.BATCH_SIZE,
                            num_workers=self.cfg.DATA.NUM_WORKERS,
                            pin_memory=True)
        return loader

    def test_dataloader(self):
        if self.testmode == 'test':
            dataset = ARVideoDataset(self.df_test,
                                     transform=self.transform)
            loader = DataLoader(dataset,
                                batch_size=self.cfg.DATA.BATCH_SIZE,
                                num_workers=self.cfg.DATA.NUM_WORKERS,
                                pin_memory=True)
        elif self.testmode == 'visualize_predictions':
            dataset = ARVideoDataset(self.df_vis,
                                     transform=self.transform)
            loader = DataLoader(dataset,
                                batch_size=self.cfg.DATA.BATCH_SIZE,
                                num_workers=0,
                                pin_memory=False)
        elif self.testmode == 'visualize_features':
            dataset = ARVideoDataset(self.df_vis,
                                     transform=self.transform)
            loader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=0,
                                pin_memory=False)
        return loader


def main(args):
    """Perform training, testing, and/or visualization"""
    if args.smoke:
        args.opts.extend([
            'TRAINER.limit_train_batches', '10',
            'TRAINER.limit_val_batches', '2',
            'TRAINER.limit_test_batches', '2',
            'TRAINER.max_epochs', '1',
            'TRAINER.default_root_dir', 'lightning_logs_c3d_dev'
        ])
        experiment_name = 'c3d_smoke'
    else:
        experiment_name = 'c3d'
    mlflow.set_experiment(experiment_name)

    if args.config is not None:
        cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()

    logger = utils.setup_logger(cfg.MISC.OUTPUT_DIR)
    logger.info(cfg)

    dm = ARVideoDataModule(cfg)

    if 'train' in args.modes:
        logger.info("======== TRAIN ========")
        if args.resume:
            cfg.LEARNER.MODEL.WEIGHTS
        best_model_path = train(cfg, dm)
        cfg.LEARNER.MODEL.WEIGHTS = best_model_path

    if 'test' in args.modes:
        logger.info("======== TEST ========")
        test(cfg, dm)

    if 'visualize' in args.modes:
        logger.info("======== VISUALIZE ========")
        visualize(cfg, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--smoke', action='store_true',
                        help='Smoke test')
    parser.add_argument('-e', '--experiment_name', default='experiment',
                        help='MLflow experiment name')
    parser.add_argument('-r', '--run_name', default='c3d',
                        help='MLflow run name')
    parser.add_argument('--config', metavar='FILE',
                        help="Path to a yaml formatted config file")
    parser.add_argument('--modes', default='train|test|visualize',
                        help="Perform training, testing, and/or visualization")
    parser.add_argument('--resume', default=False,
                        help="Resume training. Valid only in training mode.") #TODO
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options. Use dot(.) to indicate hierarchy.")
    args = parser.parse_args()
    args.modes = args.modes.split('|')
    accepted_modes = ['train', 'test', 'visualize']
    if any([m not in accepted_modes for m in args.modes]):
        raise AssertionError('Mode {} is not accepted'.format(args.modes))
    if 'train' not in args.modes and 'LEARNER.MODEL.WEIGHTS' not in args.opts:
        raise ValueError('LEARNER.MODEL.WEIGHTS must be specified in the absence of training mode.')

    with cProfile.Profile() as p:
        main(args)

    pstats.Stats(p).sort_stats('cumtime').print_stats(50)
