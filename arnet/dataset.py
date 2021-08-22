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
from arnet.utils import query_images, query_parameters, read_header
from arnet.constants import CONSTANTS


DATA_DIRS = {
    'HARP': '/data2/SHARP/image/',
    'TARP': '/data2/SMARP/image/',
}
SERIES = {
    'HARP': 'hmi.sharp_cea_720s',
    'TARP': 'mdi.smarp_cea_96m',
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
        # Default values and assertions
        features = features or ['MAGNETOGRAM']
        assert 1 <= num_frames <= 16, 'num_frames not in [1,16]'

        self.df_sample = df_sample
        self.parameters = [f for f in features if f != 'MAGNETOGRAM']
        self.yield_video = 'MAGNETOGRAM' in features
        self.yield_parameters = len(self.parameters) > 0
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.df_sample)

    def __getitem__(self, idx):
        s = self.df_sample.iloc[idx]

        # data
        data_list = []
        if self.yield_video:
            video, size = self.load_video(s['prefix'], s['arpnum'], s['t_end'], s['bad_img_idx'])
            data_list.append(video)
            data_list.append(size)
        if self.yield_parameters:
            parameters = self.load_parameters(s['prefix'], s['arpnum'], s['t_end'])
            data_list.append(parameters)


        # label
        label = int(s['label'])

        # meta
        t_end = datetime.strptime(s['t_end'], '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')
        largest_flare = max(s['flares'].split('|')) #WARNING: X10+
        meta = f'{idx}_{s["prefix"]}{s["arpnum"]:06d}_{t_end}_H0_W0_{largest_flare}.npy'
        return *data_list, label, meta

    def load_video(self, prefix, arpnum, t_end, bad_img_idx):
        t_end = drms.to_datetime(t_end)
        t_start = t_end - timedelta(minutes=96) * (self.num_frames - 1)
        t_steps = pd.date_range(t_start, t_end, freq='96min').strftime('%Y%m%d_%H%M%S_TAI')
        filenames = [f"{SERIES[prefix]}.{arpnum}.{t}.magnetogram.fits"
                     for t in t_steps]
        indices = imputed_indices(bad_img_idx, len(filenames))
        filepaths = [os.path.join(DATA_DIRS[prefix], f'{arpnum:06d}', filenames[k])
                     for k in indices]
        video = query_images(filepaths)
        video = torch.from_numpy(video)
        video = torch.unsqueeze(video, 0) # C,T,H,W
        size = torch.tensor(video.shape[-2:], dtype=torch.float) # convert to tensor, otherwise batching gives list not tensor
        size -= torch.tensor([78, 157])
        size /= torch.tensor([38, 78])
        if self.transform:
            video = self.transform(video)
        return video, size

    def load_parameters(self, prefix, arpnum, t_end):
        t_end = datetime.strptime(t_end, '%Y-%m-%d %H:%M:%S') #2013-07-03 01:36:00
        t_start = t_end - timedelta(minutes=96) * (self.num_frames - 1)
        t_recs = pd.date_range(t_start, t_end, freq='96min').strftime('%Y.%m.%d_%H:%M:%S_TAI')
        df = query_parameters(prefix, arpnum, t_recs, self.parameters)
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
        mean = [v for k, v in CONSTANTS[dataset + '_MEAN'].items() if k in self.parameters]
        std = [v for k, v in CONSTANTS[dataset + '_STD'].items() if k in self.parameters]
        df[self.parameters] = (df.values - mean) / std  # runtime: arr - arr < df - arr
        return df


class ActiveRegionDataModule(pl.LightningDataModule):
    """Active region DataModule.

    Handles cfg. Load and organize dataframes.
    """
    def __init__(self, cfg, second_stage=False):
        super().__init__()
        self.cfg = cfg
        self.second_stage = second_stage
        self._construct_transforms()
        self._construct_datasets(balanced=cfg.DATA.BALANCED)
        self.testmode = 'test'

    def _construct_transforms(self):
        transforms = [get_transform(name, self.cfg)
                      for name in self.cfg.DATA.TRANSFORMS]
        self.transform = Compose(transforms)

    def _construct_datasets(self, balanced=True):
        from pathlib import Path
        from arnet.fusion import load_csv_dataset, group_split_data, rus

        # Short aliases
        database = self.cfg.DATA.DATABASE
        dataset = self.cfg.DATA.DATASET
        seed = self.cfg.DATA.SEED
        balanced = self.cfg.DATA.BALANCED

        # Load
        df_smarp = load_csv_dataset(Path(database) / 'smarp.csv')
        df_sharp = load_csv_dataset(Path(database) / 'sharp.csv')

        # Split
        if dataset in ['sharp', 'fused_sharp']:
            df_train, df_test = group_split_data(df_sharp, seed=seed)
            df_train, df_val = group_split_data(df_train, seed=seed)
            df_vals = [df_val, df_test]
            # first stage fused_sharp is the same as sharp
            if dataset == 'fused_sharp' and self.second_stage:
                df_vals.append(df_train)
                df_train = df_smarp
        elif dataset in ['smarp', 'fused_smarp']:
            df_train, df_test = group_split_data(df_smarp, seed=seed)
            df_train, df_val = group_split_data(df_train, seed=seed)
            df_vals = [df_val, df_test]
            if dataset == 'fused_smarp' and self.second_stage:
                df_vals.append(df_train)
                df_train = df_sharp

        # RUS
        if balanced:
            df_train = rus(df_train, sizes='balanced', seed=seed)
            df_vals= [rus(df, sizes='balanced', seed=seed) for df in df_vals]
            df_test = rus(df_test, sizes='balanced', seed=seed)

        # Set attributes
        self.df_train = df_train
        self.df_vals = df_vals
        self.df_test = df_test
        self.df_vis = self.df_test.iloc[0:64:4] #sharp_train.loc[sharp_train['arpnum'] == 377].iloc[0:8:2]

        # Prediction results
        self.val_history = {
            f'validation{i}': df.copy()
            for i, df in enumerate(self.df_vals)
        }
        self.val_history['test'] = self.df_test.copy() # learner.py: fill_prob in test_epoch_end()

    def set_class_weight(self, cfg):
        p = self.df_train['label'].mean()
        cfg.DATA.CLASS_WEIGHT = [1-p, p]
        return cfg

    def fill_prob(self, tag, global_step, probs):
        import numpy as np
        df = self.val_history[tag]
        probs = [probs[i] if i < len(probs) else np.nan for i in range(len(df))]
        df[f'step-{global_step}'] = probs

    def get_dataloader(self, df_sample, shuffle=False, drop_last=False):
        dataset = ActiveRegionDataset(df_sample,
                                      features=self.cfg.DATA.FEATURES,
                                      num_frames=self.cfg.DATA.NUM_FRAMES,
                                      transform=self.transform)
        dataloader = DataLoader(dataset,
                                batch_size=self.cfg.DATA.BATCH_SIZE,
                                shuffle=shuffle,
                                drop_last=drop_last,
                                num_workers=self.cfg.DATA.NUM_WORKERS,
                                pin_memory=True)
        return dataloader

    def train_dataloader(self):
        loader = self.get_dataloader(self.df_train, shuffle=True, drop_last=True)
        return loader

    def val_dataloader(self):
        loaders = [self.get_dataloader(df) for df in self.df_vals]
        return loaders

    def test_dataloader(self):
        if self.testmode == 'test':
            loader = self.get_dataloader(self.df_test)
        elif self.testmode == 'visualize_predictions':
            loader = self.get_dataloader(self.df_vis)
        elif self.testmode == 'visualize_features':
            loader = self.get_dataloader(self.df_vis)
        else:
            raise
        return loader

    def predict_dataloader(self):
        return self.test_dataloader()


if __name__ == '__main__':
    df = pd.read_csv('datasets/M_Q_6hr/sharp.csv')
    df.loc[:, 'flares'] = df['flares'].fillna('')
    df.loc[:, 'bad_img_idx'] = df['bad_img_idx'].apply(
        lambda s: [int(x) for x in s.strip('[]').split()])
    dataset = ActiveRegionDataset(df, features=['MAGNETOGRAM', 'AREA'], num_frames=16, transform=None)
    for idx, (videos, params, _, m) in enumerate(dataset):
        if not videos.isnan().any():
            continue
        nanmap = videos.isnan().detach().cpu().numpy()
        print(m)
        print(nanmap)
