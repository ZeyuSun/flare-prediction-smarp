import re
import os
from glob import glob
from multiprocessing import Pool
import multiprocessing
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm, trange
import drms
from sunpy.time import TimeRange
from sunpy.instr.goes import get_goes_event_list


######### Change these ##########
EMAIL = 'szymails@gmail.com'
DATA_DIR = '/data2'
#################################


c = drms.Client(debug=False, verbose=False, email=EMAIL)
SHARP_LOS_HEADER_DIR = os.path.join(DATA_DIR, 'SHARP_720s/header_los')
SHARP_VEC_HEADER_DIR = os.path.join(DATA_DIR, 'SHARP_720s/header_vec')
SHARP_IMAGE_DIR = os.path.join(DATA_DIR, 'SHARP_720s/image')
for folder in [SHARP_LOS_HEADER_DIR, SHARP_VEC_HEADER_DIR, SHARP_IMAGE_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)
_re_export_recset = re.compile(r'^\s*([\w\.]+)\s*(\[.*\])?\s*(?:\{([\w\s\.,]*)\})?\s*$')
_re_export_recset_pkeys = re.compile(r'\[([^\[^\]]*)\]')
_re_export_recset_slist = re.compile(r'[\s,]+')


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

    keys.to_csv(filename, index=None)
    return 0


def _filename_from_export_record(rs):
    m = _re_export_recset.match(rs)
    sname, pkeys, segs = m.groups()
    pkeys = _re_export_recset_pkeys.findall(pkeys)
    segs = _re_export_recset_slist.split(segs)
    pkeys[1] = pkeys[1].replace('.', "").replace(':', "").replace('-', "")
    str_list = [sname] + pkeys + segs + ['fits']
    fname = '.'.join(str_list)
    return fname


def download_sharp_images(harpnum):
    header = os.path.join(SHARP_VEC_HEADER_DIR, f'HARP{harpnum:06d}_ATTRS.csv')
    if not os.path.exists(header):
        return 1

    image_dir = os.path.join(SHARP_IMAGE_DIR, f'{harpnum:06d}')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    df = pd.read_csv(header)
    t1, t2 = df['T_REC'].iloc[0], df['T_REC'].iloc[-1]
    request = c.export(f'hmi.sharp_cea_720s[{harpnum}][{t1}-{t2}][? (QUALITY<65536) ?]{{magnetogram}}') #,
                       #method='url', protocol='fits') # can't be pickled by in parallel pool
    if len(request.data) == 0:
        return 2

    downloaded = sorted([os.path.basename(f) for f in glob(os.path.join(image_dir, 'hmi*'))])
    filenames = request.data['record'].apply(_filename_from_export_record)
    idx = request.data[~filenames.isin(downloaded)].index
    if len(idx) == 0:
        return 3

    request.download(image_dir, index=idx)
    return request.data.loc[idx]

    
def batch_run(func, max_num, batch_size, num_workers=8):
    """Parallelize func by batch with progress bar"""
    pbar = trange(0, max_num+1, batch_size)
    results = []
    for start in pbar:
        stop = min(start + batch_size, max_num+1)
        pbar.set_description('Batch [%d, %d) / %d' % (start, stop, max_num+1))
        batch = range(start, stop)
        with Pool(num_workers) as p:
            results.extend(p.map(func, batch))
    return results


def analyze(results, tag, images=False):
    if images:
        dfs = [i for i in results if not isinstance(i, int)]
        if len(dfs) > 0:
            pd.concat(dfs).reset_index(drop=True).to_csv('log_add_'+tag+'.csv')
        results = [i if isinstance(i, int) else 0 for i in results]

    df = pd.DataFrame(results, columns=['result'])
    df.to_csv('log_download_'+tag+'.csv')
    print(tag)
    print(df['result'].value_counts())


if __name__ == '__main__':
    # Last Record = su_mbobra.sharp_loskeystest_720s[7545][2021.02.17_19:36:00_TAI]
    results = batch_run(download_sharp_headers, 7545, 100)
    analyze(results, 'sharp_headers')

    results = batch_run(download_sharp_images, 7545, 100)
    analyze(results, 'sharp_images', images=True)
