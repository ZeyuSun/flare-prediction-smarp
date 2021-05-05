import os
from multiprocessing import Pool
from datetime import datetime, timedelta
import drms
import pandas as pd
from tqdm import tqdm, trange


######### Change these ##########
EMAIL = 'szymails@gmail.com'
DATA_DIR = '/data2'
#################################


c = drms.Client(debug=False, verbose=False, email=EMAIL)
SHARP_LOS_HEADER_DIR = os.path.join(DATA_DIR, 'SHARP/header_los')
SHARP_VEC_HEADER_DIR = os.path.join(DATA_DIR, 'SHARP/header_vec')
SHARP_IMAGE_DIR = os.path.join(DATA_DIR, 'SHARP/image')
SMARP_HEADER_DIR = os.path.join(DATA_DIR, 'SMARP/header')
SMARP_IMAGE_DIR = os.path.join(DATA_DIR, 'SMARP/image')
for folder in [SHARP_LOS_HEADER_DIR, SHARP_VEC_HEADER_DIR, SHARP_IMAGE_DIR,
               SMARP_HEADER_DIR, SMARP_IMAGE_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)


def download_sharp_headers(harpnum):
    #### Download sharp_vec headers
    filename = os.path.join(SHARP_VEC_HEADER_DIR, f'HARP{harpnum:06d}_ATTRS.csv')
    if os.path.exists(filename):
        # Header exists
        return 1

    keys = c.query(f'hmi.sharp_cea_720s[{harpnum}][][? (QUALITY<65536) ?][? (NOAA_NUM>=1) ?]', 
                   key=drms.const.all)
    if len(keys) == 0:
        # Header has no record subject to the constraints
        return 2

    t_rec = drms.to_datetime(keys['T_REC'])
    d = t_rec[0].date()
    start_date = datetime(d.year, d.month, d.day)
    delta = timedelta(minutes=96)
    mod = (t_rec - start_date) % delta
    keys = keys[mod.dt.seconds == 0]
    if len(keys) == 0:
        # Header has no record aligned with MDI recording times
        return 3

    keys.to_csv(filename, index=None)

    #### Download sharp_los headers
    filename = os.path.join(SHARP_LOS_HEADER_DIR, f'HARP{harpnum:06d}_ATTRS.csv')
    if os.path.exists(filename):
        # Header exists
        return 4

    vec_header = os.path.join(SHARP_VEC_HEADER_DIR, f'HARP{harpnum:06d}_ATTRS.csv')
    if not os.path.exists(vec_header):
        # No corresponding SHARP_VEC header
        return 5

    keys = c.query(f'su_mbobra.sharp_loskeystest_720s[{harpnum}][][? (QUALITY<65536) ?]',
                   key=drms.const.all)  # NOAA_NUM >= 1 satisfied by corresp. SHARP_VEC header
    if len(keys) == 0:
        # Header has no record subject to the constraints
        return 6

    t_rec = drms.to_datetime(keys['T_REC'])
    d = t_rec[0].date()
    start_date = datetime(d.year, d.month, d.day)
    delta = timedelta(minutes=96)
    mod = (t_rec - start_date) % delta
    keys = keys[mod.dt.seconds == 0]
    if len(keys) == 0:
        # Header has no record aligned with MDI recording times
        return 7

    keys.to_csv(filename, index=None)
    return 0


def download_smarp_headers(tarpnum):
    filename = os.path.join(SMARP_HEADER_DIR, f'TARP{tarpnum:06d}_ATTRS.csv')
    if os.path.exists(filename):
        return 1

    keys = c.query(f'su_mbobra.smarp_cea_96m[{tarpnum}][][? (QUALITY<262144) ?][? (NOAA_NUM>=1) ?]',
                   key=drms.const.all)
    if len(keys) == 0:
        return 2

    keys.to_csv(filename, index=None)
    return 0


def download_sharp_images(harpnum):
    header = os.path.join(SHARP_VEC_HEADER_DIR, f'HARP{harpnum:06d}_ATTRS.csv')
    if not os.path.exists(header):
        return 1

    image_dir = os.path.join(SHARP_IMAGE_DIR, f'{harpnum:06d}')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    df = pd.read_csv(header)
    t1, t2 = df['T_REC'].iloc[0], df['T_REC'].iloc[-1]
    request = c.export(f'hmi.sharp_cea_720s[{harpnum}][{t1}-{t2}@96m][? (QUALITY<65536) ?]{{magnetogram}}')
    if len(request.data) == 0:
        return 2

    request.download(image_dir)
    return 0

    
def download_smarp_images(tarpnum):
    header = os.path.join(SMARP_HEADER_DIR, f'TARP{tarpnum:06d}_ATTRS.csv')
    if not os.path.exists(header):
        return 1

    image_dir = os.path.join(SMARP_IMAGE_DIR, f'{tarpnum:06d}')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    df = pd.read_csv(header)
    t1, t2 = df['T_REC'].iloc[0], df['T_REC'].iloc[-1]
    request = c.export(f'su_mbobra.smarp_cea_96m[{tarpnum}][{t1}-{t2}@96m][? (QUALITY<262144) ?]{{magnetogram}}')
    if len(request.data) == 0:
        return 2

    request.download(image_dir)
    return 0


def batch_run(func, total_num, batch_size, num_workers=8):
    """Parallelize func by batch with progress bar"""
    pbar = trange(0, total_num, batch_size)
    results = []
    for start in pbar:
        stop = start + batch_size
        pbar.set_description('Batch [%d, %d) / %d' % (start, stop, total_num))
        batch = range(start, stop)
        with Pool(num_workers) as p:
            results.extend(p.map(func, batch))
    return results


def analyze(results, tag):
    df = pd.DataFrame(results, columns=['result'])
    df.to_csv(tag+'.csv')
    print(tag)
    print(df['result'].value_counts())


if __name__ == '__main__':
    results = batch_run(download_sharp_headers, 8000, 100)
    analyze(results, 'sharp_headers')

    results = batch_run(download_smarp_headers, 14000, 1000)
    analyze(results, 'smarp_headers')

    results = batch_run(download_sharp_images, 8000, 100)
    analyze(results, 'sharp_images')

    results = batch_run(download_smarp_images, 14000, 1000)
    analyze(results, 'smarp_images')
