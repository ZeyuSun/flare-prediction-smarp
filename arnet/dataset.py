import os
import functools
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import drms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import pytorch_lightning as pl

from arnet.fusion import get_datasets
from arnet.transforms import get_transform
from arnet.utils import query, query_parameters, read_header
from arnet.constants import CONSTANTS


DATA_DIRS = {
    'HARP': '/data2/SHARP/image/',
    'TARP': '/data2/SMARP/image/',
}
SERIES = {
    'HARP': 'hmi.sharp_cea_720s',
    'TARP': 'su_mbobra.smarp_cea_96m',
}


def imputed_indices(invalid, length, method='bfill'):
    """Imputation indices assuming the last element is valid.

    Args:
        invalid: List of negative indices of invalid entries.
        length: Length of the list to impute.
        method: Imputation methods. Default 'bfill' implements backward-fill
            (use next valid element to fill the gap).

    Returns:
        indices: indices that give an imputed array.

    Example:
        bad_map = [0, 1, 1, 0, 0, 1, 0]
        invalid = [-2, -5, -6], length = 7
        indices = [0, 3, 3, 3, 4, 6, 6]
    """
    assert method == 'bfill'
    indices = list(range(length))
    for i in range(-2, -length-1, -1):
        if i in invalid:
            indices[i] = indices[i+1]
    return indices


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
        s = self.df_sample.iloc[idx]

        # data
        if 'MAGNETOGRAM' in self.features: # image
            data = self.load_video(s['prefix'], s['arpnum'], s['t_end'], s['bad_img_idx'])
        else: # parameters
            data = self.load_parameters(s['prefix'], s['arpnum'], s['t_end'])

        # label
        label = int(s['label'])

        # meta
        t_end = datetime.strptime(s['t_end'], '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')
        largest_flare = max(s['flares'].split('|')) #WARNING: X10+
        meta = f'{s["prefix"]}{s["arpnum"]:06d}_{t_end}_H0_W0_{largest_flare}.npy'
        return data, label, meta

    def load_video(self, prefix, arpnum, t_end, bad_img_idx):
        t_end = drms.to_datetime(t_end)
        t_start = t_end - timedelta(minutes=96) * (self.num_frames - 1)
        t_steps = pd.date_range(t_start, t_end, freq='96min').strftime('%Y%m%d_%H%M%S_TAI')
        filenames = [f"{SERIES[prefix]}.{arpnum}.{t}.magnetogram.fits"
                     for t in t_steps]
        indices = imputed_indices(bad_img_idx, len(filenames))
        filepaths = [os.path.join(DATA_DIRS[prefix], f'{arpnum:06d}', filenames[k])
                     for k in indices]
        video = query(filepaths)
        video = torch.from_numpy(video)
        video = torch.unsqueeze(video, 0) # C,T,H,W
        if self.transform:
            video = self.transform(video)
        return video

    def load_parameters(self, prefix, arpnum, t_end):
        t_end = datetime.strptime(t_end, '%Y-%m-%d %H:%M:%S') #2013-07-03 01:36:00
        t_start = t_end - timedelta(minutes=96) * (self.num_frames - 1)
        t_recs = pd.date_range(t_start, t_end, freq='96min').strftime('%Y.%m.%d_%H:%M:%S_TAI')
        df = query_parameters(prefix, arpnum, t_recs, self.features)
        df = df.fillna(method='bfill')

        # Check na is time consuming. If na, loss will be nan
        #if df.isna().any(axis=None):
        #    print(df)
        #    breakpoint()

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
        mean = [v for k, v in CONSTANTS[dataset + '_MEAN'].items() if k in self.features]
        std = [v for k, v in CONSTANTS[dataset + '_STD'].items() if k in self.features]
        df[self.features] = (df.values - mean) / std  # runtime: arr - arr < df - arr
        return df


class ActiveRegionDataModule(pl.LightningDataModule):
    """Active region DataModule.

    Handles cfg. Load and organize dataframes.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._construct_transforms()
        self._construct_datasets(balanced=cfg.DATA.BALANCED)
        self.testmode = 'test'

    def _construct_transforms(self):
        transforms = [get_transform(name, self.cfg)
                      for name in self.cfg.DATA.TRANSFORMS]
        self.transform = Compose(transforms)

    def _construct_datasets(self, balanced=True):
        if self.cfg.DATA.BALANCED:
            sizes = 'balanced'
        else:
            sizes = None

        self.df_train, self.df_val, self.df_test = get_datasets(
            self.cfg.DATA.DATABASE,
            self.cfg.DATA.DATASET,
            self.cfg.DATA.AUXDATA,
            sizes=sizes,
            validation=True,
            seed=self.cfg.DATA.SEED)

        # # Refit on train and validation
        # self.df_train, self.df_test = get_datasets(
        #     self.cfg.DATA.DATABASE,
        #     self.cfg.DATA.DATASET,
        #     self.cfg.DATA.AUXDATA,
        #     sizes=sizes,
        #     validation=False,
        #     seed=self.cfg.DATA.SEED)
        # self.df_val = self.df_test

        self.df_vis = self.df_test.iloc[:4] #sharp_train.loc[sharp_train['arpnum'] == 377].iloc[0:8:2]

    def set_class_weight(self, cfg):
        p = self.df_train['label'].mean()
        cfg.DATA.CLASS_WEIGHT = [1-p, p]
        return cfg

    def train_dataloader(self):
        dataset = ActiveRegionDataset(self.df_train,
                                      features=self.cfg.DATA.FEATURES,
                                      num_frames=self.cfg.DATA.NUM_FRAMES,
                                      transform=self.transform)
        #sampler = RandomSampler(dataset, len(dataset) // 2)
        loader = DataLoader(dataset,
                            batch_size=self.cfg.DATA.BATCH_SIZE,
                            shuffle=True,
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


if __name__ == '__main__':
    df_sample = pd.read_csv('datasets/preprocessed/MX_Q_6hr/smarp_train.csv')
    dataset = ActiveRegionDataset(df_sample, features=['MAGNETOGRAM'], num_frames=16, transform=None)
    for idx, (sample, _, m) in enumerate(dataset):
        if not sample.isnan().any():
            continue
        nanmap = sample.isnan().detach().cpu().numpy()
        print(m)
        print(nanmap)