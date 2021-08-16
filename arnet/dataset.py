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
        if self.transform:
            data_list = self.transform(data_list)

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

        sequence = torch.tensor(df.to_numpy(), dtype=torch.float32) # float16 causes error, lstm is 32bit
        return sequence


class ActiveRegionDataModule(pl.LightningDataModule):
    """Active region DataModule.

    Handles cfg. Load and organize dataframes.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._construct_datasets(balanced=cfg.DATA.BALANCED)
        self._construct_transforms()
        self.testmode = 'test'

    def _construct_transforms(self):
        self.transform = get_transform(self.cfg, self.df_train)

    def _construct_datasets(self, balanced=True):
        self.df_train, self.df_val, self.df_test = get_datasets(
            self.cfg.DATA.DATABASE,
            self.cfg.DATA.DATASET,
            self.cfg.DATA.AUXDATA,
            sizes='balanced' if self.cfg.DATA.BALANCED else None,
            validation=True,
            seed=self.cfg.DATA.SEED)

        # # Refit on train and validation
        # self.df_train, self.df_test = get_datasets(
        #     self.cfg.DATA.DATABASE,
        #     self.cfg.DATA.DATASET,
        #     self.cfg.DATA.AUXDATA,
        #     sizes='balanced' if self.cfg.DATA.BALANCED else None,
        #     validation=False,
        #     seed=self.cfg.DATA.SEED)
        # self.df_val = self.df_test

        self.df_vis = self.df_test.iloc[0:64:4] #sharp_train.loc[sharp_train['arpnum'] == 377].iloc[0:8:2]

        self.df_val_pred = self.df_val.copy() # to be logged as artifacts
        self.df_test_pred = self.df_test.copy()

    def set_class_weight(self, cfg):
        p = self.df_train['label'].mean()
        cfg.DATA.CLASS_WEIGHT = [1-p, p]
        return cfg

    def fill_prob(self, tag, global_step, probs):
        if tag == 'validation':
            self.df_val_pred[f'step-{global_step}'] = probs
        elif tag == 'test':
            self.df_test_pred[f'step-{global_step}'] = probs

    def get_dataloader(self, df_sample, shuffle=False, drop_last=False):
        dataset = ActiveRegionDataset(df_sample,
                                      features=self.cfg.DATA.FEATURES,
                                      num_frames=self.cfg.DATA.NUM_FRAMES,
                                      transform=self.transform)
        #g = torch.Generator()
        #g.manual_seed(self.cfg.DATA.SEED)
        dataloader = DataLoader(dataset,
                                batch_size=self.cfg.DATA.BATCH_SIZE,
                                shuffle=shuffle,
                                drop_last=drop_last,
                                num_workers=self.cfg.DATA.NUM_WORKERS,
                                pin_memory=True)
                                # Must not be set if use seed_everything(workers=True)
                                #worker_init_fn=pl.utilities.seed.pl_worker_init_function,
                                #generator=g)
        return dataloader

    def train_dataloader(self):
        loader = self.get_dataloader(self.df_train, shuffle=True, drop_last=True)
        return loader

    def val_dataloader(self):
        loader = self.get_dataloader(self.df_val)
        return loader

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
