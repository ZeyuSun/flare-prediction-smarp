import os
import argparse
import logging
from multiprocessing import Pool
from collections import defaultdict
from functools import partial
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from arnet.utils import read_header, query_images
from utils import get_flare_index


def get_prefix(dataset):
    if dataset == 'sharp':
        prefix = 'HARP'
    elif dataset == 'smarp':
        prefix = 'TARP'
    else:
        raise
    return prefix


def get_image_filepath(dataset, arpnum, t_rec):
    """t_rec(str) should be in format '%Y%m%d_%H%M%S_TAI'"""
    t_rec = datetime.strptime(t_rec, T_REC_FORMAT).strftime('%Y%m%d_%H%M%S_TAI')
    if dataset == 'sharp':
        return os.path.join(args.raw_data_dir, f'SHARP/image/{arpnum:06d}/hmi.sharp_cea_720s.{arpnum}.{t_rec}.magnetogram.fits')
    elif dataset == 'smarp':
        return os.path.join(args.raw_data_dir, f'SMARP/image/{arpnum:06d}/mdi.smarp_cea_96m.{arpnum}.{t_rec}.magnetogram.fits')
    else:
        raise


def get_label(flares_observed, flares_future, criterion='M_Q'):
    """Assign a label to a sample given observed and future flares.

    `flares_observed` and `flares_futures` falls into one of the following
    three categories, given threshold T:
        Q: quiet. S: small flares (<T). L: large flares (>=T)

    Their combinations can be represented by a 3x3 matrix:
                Future
         Obs    Q   S   L
          Q    ( ) ( ) ( )
          S    ( ) ( ) ( )
          L    ( ) ( ) ( )

    `criterion` is in the form of `threshold_negative` and is used to select
    positive (+) and negative (-) samples or discard samples ( ).
                'T_Q'           'T_QS'          'T_QSL'
                Future          Future          Future
         Obs    Q   S   L       Q   S   L       Q   S   L
          Q    (-) ( ) (+)     (-) (-) (+)     (-) (-) (+)
          S    ( ) ( ) (+)     (-) (-) (+)     (-) (-) (+)
          L    ( ) ( ) (+)     ( ) ( ) (+)     (-) (-) (+)

    Args:
        flares_observed: List of flares in the observation time
        flares_future: List of flares in the prediction window
        criterion: Classification criterion.

    Returns:
        label: True, False, or None
        evolution: Two letter string of evolution
    """
    # Prefer to return 2-letter type than binary label.
    THRESHOLDS = ['C', 'M', 'X']
    thresh, neg = criterion.split('_')
    assert thresh in THRESHOLDS
    large_class = [T for T in THRESHOLDS if T >= thresh]
    evolution_neg = {
        'Q': ['QQ'],
        'QS': ['QQ', 'QS', 'SQ', 'SS'],
        'QSL': ['QQ', 'QS', 'SQ', 'SS', 'LQ', 'LS'],
    }[neg]

    activities = [None, None]
    for i, flares in enumerate([flares_observed, flares_future]):
        if len(flares) == 0:
            activities[i] = 'Q'
        elif any([T in ''.join(flares) for T in large_class]):
            activities[i] = 'L'
        else:
            activities[i] = 'S'

    evolution = ''.join(activities)
    if evolution in ['QL', 'SL', 'LL']:
        label = True
    elif evolution in evolution_neg:
        label = False
    else:
        label = None

    return label, evolution


#@profile
def select_per_arp(dataset, arpnum,
                   val_time=None, criterion=None):
    """
    Args:
        dataset (str): 'smarp' or 'sharp'
        arpnum (int): active region patch number

    Returns:
        samples (list): a list of samples, each represented by a dictionary
    """
    df = read_header(dataset, arpnum, index_col='T_REC')
    if df is None: # No matched los header for SHARP
        return None

    assert df.index.is_monotonic_increasing

    # Only keep observations near central meridian
    # Side effect: records with nan LON_MIN and LON_MAX will be dropped
    df = df[(df['LON_MIN'] >= LON_MIN) & (df['LON_MAX'] <= LON_MAX)]
    if len(df) == 0:
        return None

    # Get relevant GOES event records
    df.loc[:, 'NOAA_ARS'] = df['NOAA_ARS'].astype(str) # cast to str if all entries are int
    noaa_ars = df['NOAA_ARS'].unique() # Series.unique returns numpy.ndarray
    assert len(noaa_ars) == 1 # expect all records to have the same NOAA_ARS
    noaa_ars = [int(ar) for ar in noaa_ars[0].split(',')]
    goes_ar = GOES[GOES['noaa_active_region'].isin(noaa_ars)]

    # For SHARP, only keep observations between 2010.10.29 and 2020.12.01
    if dataset == 'sharp':
        #TODO: searchsorted?
        df = df[(df.index >= T_REC_MIN) & (df.index <= T_REC_MAX)]
        if len(df) == 0:
            return None

    # 1st scan: read images and mark if there is nan
    df.loc[:, 'bad_img'] = None
    df.loc[:, 'HEIGHT'] = None
    df.loc[:, 'WIDTH'] = None
    df.loc[:, 'SUM'] = None
    df.loc[:, 'SUM_SQR'] = None
    # bad image: file missing, nan pixels, or inconsistent sizes.
    shapes = []
    for t_rec in df.index:
        image_file = get_image_filepath(dataset, arpnum, t_rec)
        image_data = query_images(image_file)
        df.loc[t_rec, 'bad_img'] = np.any(np.isnan(image_data))
        df.loc[t_rec, 'HEIGHT'] = image_data.shape[0]
        df.loc[t_rec, 'WIDTH'] = image_data.shape[1]
        df.loc[t_rec, 'SUM'] = np.sum(image_data)
        df.loc[t_rec, 'SUM_SQR'] = np.sum(image_data ** 2)
        shapes.append(image_data.shape)

    #TODO: check how size consistency are violated
    # Check image size consistency
    hs, ws = zip(*shapes)
    for ss in [hs, ws]:
        # Frames with either dim deviating more than 2 pix from the median
        df.loc[np.abs(ss - np.median(ss)) > 2, 'bad_img'] = True

    # 2nd scan: generate sequences
    samples = []
    counter = defaultdict(int)
    for t_rec in df.index:
        t_start = datetime.strptime(t_rec, T_REC_FORMAT)  # observation start
        t_end = t_start + OBS_TIME  # observation end; prediction time window start
        t_future = t_end + val_time  # prediction time window end

        t_steps = pd.date_range(t_start, t_end, freq='96min').strftime(T_REC_FORMAT)
        df_new = df.reindex(index=t_steps) # conformed df

        # missing records introduced by reindex are considered to have bad images
        df_new.loc[df_new['bad_img'].isna(), 'bad_img'] = True

        # (1) Drop the sample with too many nan/missing keywords:
        #   Allow for <=2 missing entries in all but last row for each feature column
        nan_arr = np.isnan(df_new[KEYWORDS].values)  # ~3x faster than df
        if ((nan_arr.sum(axis=0) > 2).any() or
            nan_arr[-1, :].any()):
            counter['nan_key'] += 1
            continue

        # (2) Drop the sample with too many nan/missing images:
        #   Allow for <=2 nan/missing images in all but last t_rec
        if (df_new['bad_img'].sum() > 2 or
            df_new['bad_img'].iloc[-1]):
            counter['bad_img'] += 1
            continue

        bad_img_idx = (np.where(df_new['bad_img'])[0] - 16)  # neg idx of bad images

        flares_future = goes_ar.loc[(goes_ar['start_time'] >= t_end.strftime(GOES_TIME_FORMAT)) &
                                    (goes_ar['start_time'] <= t_future.strftime(GOES_TIME_FORMAT)),
                                    'goes_class'].tolist()
        flares_observed = goes_ar.loc[(goes_ar['start_time'] >= t_start.strftime(GOES_TIME_FORMAT)) &
                                      (goes_ar['start_time'] <= t_end.strftime(GOES_TIME_FORMAT)),
                                      'goes_class'].tolist()
        flare_index = get_flare_index(flares_observed)

        # (3) Drop the negative sample with large observed flares
        label, evolution = get_label(flares_observed, flares_future, criterion)
        if label is None:
            counter['obs_pos'] += 1
            continue

        sample = {
            'prefix': get_prefix(dataset),
            'arpnum': arpnum,
            't_start': t_start,
            't_end': t_end,
            'label': label,
            'evolution': evolution,
            'noaa_ars': noaa_ars, # list[int]. Untested. Add NOAA AR so that we can look for flares given the sample
            'flares': '|'.join(flares_future),
            'bad_img_idx': bad_img_idx,
            'HEIGHT': df_new['HEIGHT'].iloc[-1],
            'WIDTH': df_new['WIDTH'].iloc[-1],
            'SUM': df_new['SUM'].iloc[-1],
            'SUM_SQR': df_new['SUM_SQR'].iloc[-1],
            'FLARE_INDEX': flare_index,
        }
        sample.update({k: df_new[k].iloc[-1] for k in KEYWORDS})
        samples.append(sample)
    logger.info('{} {}: {}/{} sequences extracted. {}'.format(
        get_prefix(dataset), arpnum, len(samples), len(df), dict(counter)))
    return samples


def select(dataset, arpnums, val_time, criterion):
    closure = partial(select_per_arp, dataset,
                      val_time=val_time, criterion=criterion)

    # # Non-parallel
    # samples = map(closure, arpnums)

    # Parallel
    with Pool(24) as pool:
        samples = pool.map(closure, arpnums)

    samples = [s for s in samples if s is not None]
    samples = [i for s in samples for i in s]  # concatenate
    sample_df = pd.DataFrame(samples)
    return sample_df


def get_arpnums(dataset):
    if dataset == 'sharp':
        header_dir = os.path.join(args.raw_data_dir, 'SHARP/header')
    elif dataset == 'smarp':
        header_dir = os.path.join(args.raw_data_dir, 'SMARP/header')
    else:
        raise
    header_files = sorted(os.listdir(header_dir))
    arpnums = [int(f[4:10]) for f in header_files]

    return arpnums


def main(split_num, output_dir, val_time, criterion):
    output_dir = os.path.join(args.processed_data_dir, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dataset in ['smarp', 'sharp']:
        logger.info(dataset)
        arpnums = get_arpnums(dataset)
        df = select(dataset, arpnums, val_time, criterion)
        df.to_csv(os.path.join(output_dir, f'{dataset}.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', default='/data2')
    parser.add_argument('--processed_data_dir', default='datasets')
    parser.add_argument('--seed', default=0)
    args = parser.parse_args()

    # global variables
    LON_MIN, LON_MAX = -70, 70
    T_REC_FORMAT = '%Y.%m.%d_%H:%M:%S_TAI'
    T_REC_MIN = datetime(year=2010, month=10, day=29).strftime(T_REC_FORMAT)
    T_REC_MAX = datetime(year=2020, month=12, day=1).strftime(T_REC_FORMAT)
    OBS_TIME = timedelta(days=1)  # observation time
    KEYWORDS = ['AREA', 'USFLUXL', 'MEANGBL', 'R_VALUE']

    GOES_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S.000'
    GOES = pd.read_csv(os.path.join(args.raw_data_dir, 'GOES/goes.csv'))
    GOES = GOES.dropna(subset=['goes_class'])
    if not os.path.exists(args.processed_data_dir):
        os.makedirs(args.processed_data_dir)
    logging.basicConfig(filename=os.path.join(args.processed_data_dir, 'log_preprocess.txt'),
                        filemode='a',
                        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()

    # debug
    # samples = select_per_arp('smarp', 3697, val_time=timedelta(hours=6), criterion='MX_Q')
    # breakpoint()
    # raise

    # begin preprocessing
    for criterion in ['M_Q', 'M_QS', 'M_QSL']:
        for val_hours in [24]:
            output_dir = f'{criterion}_{val_hours}hr'
            logger.info(output_dir)
            print(output_dir)
            main(split_num=5,
                 output_dir=output_dir,
                 val_time=timedelta(hours=val_hours),
                 criterion=criterion)
