import os
import functools
from datetime import datetime, timedelta
import pandas as pd
import drms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import pytorch_lightning as pl

from arnet.utils import query, query_parameters, read_header
from arnet.transforms import get_transform
from arnet.constants import CONSTANTS


PROCESSED_DATA_DIR = '/home/zeyusun/work/flare-prediction-smarp/datasets/M1.0_24hr_balanced'
SPLIT_DIRS = {
    'HARP': os.path.join(PROCESSED_DATA_DIR, 'sharp'),
    'TARP': os.path.join(PROCESSED_DATA_DIR, 'smarp'),
}
DATA_DIRS = {
    'HARP': '/data2/SHARP/image/',
    'TARP': '/data2/SMARP/image/',
}
SERIES = {
    'HARP': 'hmi.sharp_cea_720s',
    'TARP': 'su_mbobra.smarp_cea_96m',
}
HEADER_DIRS = {
    'TARP': '/data2/SMARP/header/',
    'HARP': '/data2/SHARP/header_los/',
}
HEADER_TIME_FMT = '%Y.%m.%d_%H:%M:%S_TAI' # 2006.05.21_14:24:00_TAI
CADENCE = timedelta(minutes=96)


class ActiveRegionDataset(Dataset):
    """Active Region dataset.

    Args:
        df_sample: Sample data frame.
        features: List of features to use. If None, use magnetogram.
        num_frames: Number of frames before t_end to use.
        transforms (callable): Transform to apply to samples.
    """
    def __init__(self, df_sample, features=None, num_frames=16, transform=None):
        self.df_sample = df_sample
        self.df_sample['flares'].fillna('', inplace=True)

        features = features or ['MAGNETOGRAM']
        if 'MAGNETOGRAM' in features and len(features) > 1:
            raise ValueError('combining image with parameter not allowed')
        self.features = features
        assert 1 <= num_frames <= 16, 'num_frames not in [1,16]'
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.df_sample)

    def __getitem__(self, idx):
        sample = self.df_sample.iloc[idx]

        # data
        if 'MAGNETOGRAM' in self.features: # image
            data = self.load_video(sample['prefix'], sample['arpnum'], sample['t_end'])
        else: # parameters
            data = self.load_parameters(sample['prefix'], sample['arpnum'], sample['t_end'])

        # label
        label = int(sample['label'])

        # meta
        t_end = datetime.strptime(sample['t_end'], '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')
        largest_flare = max(sample['flares'].split('|')) #WARNING: X10+
        meta = f'{sample["prefix"]}{sample["arpnum"]:06d}_{t_end}_H0_W0_{largest_flare}.npy'
        return data, label, meta

    def load_video(self, prefix, arpnum, t_end):
        t_end = drms.to_datetime(t_end)
        t_start = t_end - timedelta(minutes=96) * (self.num_frames - 1)
        t_steps = pd.date_range(t_start, t_end, freq='96min').strftime('%Y%m%d_%H%M%S_TAI')
        filenames = (f"{SERIES[prefix]}.{arpnum}.{t}.magnetogram.fits"
                     for t in t_steps)
        _filepaths = [os.path.join(DATA_DIRS[prefix], f'{arpnum:06d}', filename)
                     for filename in filenames]
        # Assumes (1) last frame exists (2) missing at most one filepath
        filepaths = [p if os.path.isfile(p) else _filepaths[i+1]
                     for i, p in enumerate(_filepaths)]
        video = query(filepaths)
        video = torch.from_numpy(video)
        video = torch.unsqueeze(video, 0) # C,T,H,W
        if self.transform:
            video = self.transform(video)
        return video

    @functools.lru_cache(maxsize=8)
    def load_header(self, prefix, arpnum):
        if prefix == 'HARP':
            dataset = 'sharp'
        elif prefix == 'TARP':
            dataset = 'smarp'
        else:
            raise
        header_df = read_header(dataset, arpnum)
        #header_file = os.path.join(HEADER_DIRS[prefix], f'{prefix}{arpnum:06d}_ATTRS.csv')
        #header_df = pd.read_csv(header_file)
        # df['T_REC'] = df['T_REC'].apply(drms.to_datetime) # time consuming
        return header_df

    def load_parameters(self, prefix, arpnum, t_end):
        t_end = drms.to_datetime(t_end)
        t_start = t_end - timedelta(minutes=96) * (self.num_frames - 1)
        t_recs = pd.date_range(t_start, t_end, freq='96min').strftime('%Y.%m.%d_%H:%M:%S_TAI')
        df = query_parameters(prefix, arpnum, t_recs, self.features)
        df = df.fillna(method='bfill')
        if df.isna().any(axis=None):
            print(df)
            breakpoint()
        #if self.transform:
        #    df = self.transform(df)
        df = self.standardize(df, prefix)
        sequence = torch.tensor(df.to_numpy(), dtype=torch.float32) # float16 causes error, lstm is 32bit
        return sequence
        #sequence = standardize(prefix, sequence).astype(np.float32)

    def standardize(self, df, prefix):
        if prefix == 'HARP':
            dataset = 'SHARP'
        elif prefix == 'TARP':
            dataset = 'SMARP'
        else:
            raise
        mean = {k: v for k, v in CONSTANTS[dataset + '_MEAN'].items() if k in self.features}
        std = {k: v for k, v in CONSTANTS[dataset + '_STD'].items() if k in self.features}
        df = (df - mean) / std
        return df


class ActiveRegionDataModule(pl.LightningDataModule):
    """Active region DataModule.

    Handles cfg. Load and organize dataframes.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.construct_transforms()
        self.construct_datasets()
        self.testmode = 'test'

    def construct_transforms(self):
        transforms = [get_transform(name, self.cfg)
                      for name in self.cfg.DATA.TRANSFORMS]
        self.transform = Compose(transforms)

    def construct_datasets(self):
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
        #self.df_vis = sharp_test.iloc[:4]
        self.df_vis = sharp_train.loc[sharp_train['arpnum'] == 377].iloc[0:8:2]

    def train_dataloader(self):
        dataset = ActiveRegionDataset(self.df_train,
                                      features=self.cfg.DATA.FEATURES,
                                      num_frames=self.cfg.DATA.NUM_FRAMES,
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
        dataset = ActiveRegionDataset(self.df_val,
                                      features=self.cfg.DATA.FEATURES,
                                      num_frames=self.cfg.DATA.NUM_FRAMES,
                                      transform=self.transform)
        loader = DataLoader(dataset,
                            batch_size=self.cfg.DATA.BATCH_SIZE,
                            num_workers=self.cfg.DATA.NUM_WORKERS,
                            pin_memory=True)
        return loader

    def test_dataloader(self):
        if self.testmode == 'test':
            dataset = ActiveRegionDataset(self.df_test,
                                          features=self.cfg.DATA.FEATURES,
                                          num_frames=self.cfg.DATA.NUM_FRAMES,
                                          transform=self.transform)
            loader = DataLoader(dataset,
                                batch_size=self.cfg.DATA.BATCH_SIZE,
                                num_workers=self.cfg.DATA.NUM_WORKERS,
                                pin_memory=True)
        elif self.testmode == 'visualize_predictions':
            dataset = ActiveRegionDataset(self.df_vis,
                                          features=self.cfg.DATA.FEATURES,
                                          num_frames=self.cfg.DATA.NUM_FRAMES,
                                          transform=self.transform)
            loader = DataLoader(dataset,
                                batch_size=self.cfg.DATA.BATCH_SIZE,
                                num_workers=0,
                                pin_memory=False)
        elif self.testmode == 'visualize_features':
            dataset = ActiveRegionDataset(self.df_vis,
                                          features=self.cfg.DATA.FEATURES,
                                          num_frames=self.cfg.DATA.NUM_FRAMES,
                                          transform=self.transform)
            loader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=0,
                                pin_memory=False)
        else:
            raise
        return loader


