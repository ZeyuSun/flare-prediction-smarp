import os
import logging
from multiprocessing import Pool
from collections import defaultdict
from functools import partial
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import drms
from tqdm import tqdm, trange

from data import read_header, query
from utils import get_flare_index


######### Change these ##########
DATA_DIR = '/data2'
#################################


GOES_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S.000'
KEYWORDS = ['AREA', 'USFLUX', 'MEANGBZ', 'R_VALUE']
goes = pd.read_csv(os.path.join(DATA_DIR, 'GOES/goes.csv'))
goes['goes_class'] = goes['goes_class'].fillna('')
logging.basicConfig(filename='log_preprocess.txt',
                    filemode='a',
                    format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


def get_prefix(dataset):
    if dataset == 'sharp':
        prefix = 'HARP'
    elif dataset == 'smarp':
        prefix = 'TARP'
    else:
        raise
    return prefix


def get_image_filepath(dataset, arpnum, t_rec):
    t_rec = t_rec.strftime('%Y%m%d_%H%M%S_TAI')
    if dataset == 'sharp':
        return os.path.join(DATA_DIR, f'SHARP/image/{arpnum:06d}/hmi.sharp_cea_720s.{arpnum}.{t_rec}.magnetogram.fits')
    elif dataset == 'smarp':
        return os.path.join(DATA_DIR, f'SMARP/image/{arpnum:06d}/su_mbobra.smarp_cea_96m.{arpnum}.{t_rec}.magnetogram.fits')
    else:
        raise


def split(dataset, split_num, seed=None):
    if dataset == 'sharp':
        header_dir = os.path.join(DATA_DIR, 'SHARP/header_vec')
    elif dataset == 'smarp':
        header_dir = os.path.join(DATA_DIR, 'SMARP/header')
    else:
        raise
    header_files = sorted(os.listdir(header_dir))
    np.random.seed(seed)
    numbers = np.random.permutation([int(f[4:10]) for f in header_files])

    train_size = (len(numbers) // split_num) * (split_num - 1)
    splits = numbers[:train_size].reshape(split_num-1, -1).tolist()
    splits += [numbers[train_size:].tolist()]
    return splits


def select(dataset, arpnums):
    # Non-parallel
    #samples = map(partial(select_per_arp, dataset), arpnums)

    # Parallel
    with Pool(24) as pool:
        samples = pool.map(partial(select_per_arp, dataset), arpnums)

    samples = [s for s in samples if s is not None]
    samples = [i for s in samples for i in s]  # concatenate
    sample_df = pd.DataFrame(samples)
    return sample_df


#@profile
def select_per_arp(dataset, arpnum):
    """
    Args:
        dataset (str): 'smarp' or 'sharp'
        arpnum (int): active region patch number

    Returns:
        samples (list): a list of samples, each represented by a dictionary
    """
    df = read_header(dataset, arpnum)
    if df is None: # No matched los header for SHARP
        return None

    # Only keep observations near central meridian
    LON_MIN, LON_MAX = -70, 70
    df = df[(df['LON_MIN'] >= LON_MIN) & (df['LON_MAX'] <= LON_MAX)]
    if len(df) == 0:
        return None

    # Get relevant GOES event records
    df['NOAA_ARS'] = df['NOAA_ARS'].astype(str) # cast to str if all entries are int
    noaa_ars = df['NOAA_ARS'].unique() # Series.unique returns numpy.ndarray
    assert len(noaa_ars) == 1
    noaa_ars = [int(ar) for ar in noaa_ars[0].split(',')]
    goes_ar = goes[goes['noaa_active_region'].isin(noaa_ars)]

    # For SHARP, only keep observations between 2010.10.29 and 2020.12.01
    df['T_REC'] = df['T_REC'].apply(drms.to_datetime)
    T_REC_MIN = datetime(year=2010, month=10, day=29)
    T_REC_MAX = datetime(year=2020, month=12, day=1)
    if dataset == 'sharp':
        df = df[(df['T_REC'] >= T_REC_MIN) &
                (df['T_REC'] <= T_REC_MAX)]
        if len(df) == 0:
            return None

    # 1st scan: read images and mark if there is nan
    df['image_nan'] = None
    for idx in df.index:
        image_file = get_image_filepath(dataset, arpnum, df.loc[idx, 'T_REC'])
        image_data = query(image_file)
        df.loc[idx, 'image_nan'] = np.any(np.isnan(image_data))

    # 2nd scan: generate sequences
    OBS_TIME = timedelta(days=1)  # observation time
    VAL_TIME = timedelta(days=1)  # prediction validity period
    samples = []
    for idx in df.index:
        t_start = df.loc[idx, 'T_REC']  # sequence start
        t_end = t_start + OBS_TIME  # sequence end; flare issuance time
        mask = (df['T_REC'] >= t_start) & (df['T_REC'] <= t_end)
        if len(df.loc[mask]) <= 14:
            # Allow for at most 2 missing frames
            # There are 16 frames (1+24*60/96) if no frame is missing
            continue

        if (df.loc[mask, KEYWORDS].isna().sum(axis=0) > 2).any():
            # Allow for at most 2 missing entries for each feature column
            #print(df.loc[mask, KEYWORDS].isna())
            continue

        if df.loc[mask, KEYWORDS].iloc[-1,:].isna().any():
            # The last record is used in snapshot datasets so it can't be nan
            continue

        t_after = t_end + VAL_TIME
        t_before = t_end - OBS_TIME
        flares_after = goes_ar.loc[(goes_ar['start_time'] >= t_end.strftime(GOES_TIME_FORMAT)) &
                                   (goes_ar['start_time'] <= t_after.strftime(GOES_TIME_FORMAT)),
                                   'goes_class'].tolist()
        flares_after = '|'.join(flares_after)
        label = 'M' in flares_after or 'X' in flares_after

        flares_before = goes_ar.loc[(goes_ar['start_time'] >= t_before.strftime(GOES_TIME_FORMAT)) &
                                    (goes_ar['start_time'] <= t_end.strftime(GOES_TIME_FORMAT)),
                                    'goes_class'].tolist()
        flare_index = get_flare_index(flares_before)
        flares_before = '|'.join(flares_before)
        if not label and (len(flares_before) > 0  or len(flares_after) > 0):
            # Only select queit samples for the negative class
            continue

        if df.loc[mask, 'image_nan'].sum() >= 1:
            continue

        if df.loc[mask, 'image_nan'].iloc[-1]:
            continue

        sample = {
            'prefix': get_prefix(dataset),
            'arpnum': arpnum,
            't_start': t_start,
            't_end': t_end,
            'label': label,
            'flares': flares_after,
            'FLARE_INDEX': flare_index,
        }
        sample.update({k: df.loc[mask, k].iloc[-1] for k in KEYWORDS})
        samples.append(sample)
    logging.info('{} {}: {}/{} sequences extracted'.format(get_prefix(dataset), arpnum, len(samples), len(df)))
    return samples


def main(dataset, split_num=5, output_dir=None):
    output_dir = output_dir or f'datasets_quiet/{dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    splits = split(dataset, split_num, seed=0)
    for key, arpnums in tqdm(enumerate(splits)):
        logger.info(f'Split {key} / {len(splits)-1}')
        df = select(dataset, arpnums)
        filepath = os.path.join(output_dir, f'{dataset}_{key}.csv')
        df.to_csv(filepath, index=False)

    dfs = [pd.read_csv(os.path.join(output_dir, f'{dataset}_{key}.csv'))
           for key in range(split_num)]
    train_df = pd.concat(dfs[:split_num-1]).reset_index(drop=True) #TODO: sort by dataset/arpnum
    train_df = rus(train_df, seed=0)
    test_df = dfs[-1]
    test_df = rus(test_df, seed=1)

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)


def rus(df, seed=None):
    np.random.seed(seed)
    pos_mask = df['label'].to_numpy()
    neg_mask = (~df['label']).to_numpy()
    drop_idx = np.random.choice(np.nonzero(neg_mask)[0],
                                size=neg_mask.sum() - pos_mask.sum(),
                                replace=False)
    neg_mask[drop_idx] = False
    df = df.iloc[pos_mask | neg_mask].reset_index(drop=True)
    return df


def test_seed():
    np.random.seed(0)
    a = np.random.randint(0, 65536, 10)
    assert np.all(a==[2732, 43567, 42613, 52416, 45891, 21243, 30403, 32103, 41993, 57043])


if __name__ == '__main__':
    test_seed()
    for dataset in ['smarp', 'sharp']:
        main(dataset, split_num=5, output_dir=None)
