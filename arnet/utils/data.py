import os
import json
import logging
import numpy as np
import pandas as pd
import redis
from astropy.io import fits


DATA_DIR = '/data2'
r_header = redis.Redis(db=3)
r_image = redis.Redis(db=13)


def read_header(dataset, arpnum):
    """
    Returns:
        header (dataframe): The keywords dataframe. Returns None if no sharp_los found.
    """
    KEYWORDS_LOS = ['USFLUX', 'MEANGBZ', 'R_VALUE']

    if dataset == 'sharp':
        header = pd.read_csv(os.path.join(DATA_DIR, f'SHARP/header_vec/HARP{arpnum:06d}_ATTRS.csv'),
                             index_col='T_REC')
        header[KEYWORDS_LOS] = np.nan
        header_los_path = os.path.join(DATA_DIR, f'SHARP/header_los/HARP{arpnum:06d}_ATTRS.csv')
        if not os.path.exists(header_los_path):
            logging.warning('%s: file not found. Return None.' % header_los_path)
            return None
        header_los = pd.read_csv(header_los_path, index_col='T_REC')
        if not header.index.equals(header_los.index):
            logging.warning('Header t_recs mismatch: header_los and header of HARP %d' % arpnum)
        header.update(header_los[KEYWORDS_LOS])
        header.reset_index(inplace=True)
    elif dataset == 'smarp':
        header = pd.read_csv(os.path.join(DATA_DIR, f'SMARP/header/TARP{arpnum:06d}_ATTRS.csv'))
    else:
        raise
    return header


def toRedis(arr: np.array) -> bytes:
    arr_dtype = bytearray(str(arr.dtype), 'utf-8')
    arr_shape = bytearray(','.join([str(a) for a in arr.shape]), 'utf-8')
    sep = bytearray('|', 'utf-8')
    arr_bytes = arr.ravel().tobytes()
    to_return = arr_dtype + sep + arr_shape + sep + arr_bytes
    return bytes(to_return)


def fromRedis(serialized_arr: bytes) -> np.array:
    sep = '|'.encode('utf-8')
    i_0 = serialized_arr.find(sep)
    i_1 = serialized_arr.find(sep, i_0 + 1)
    arr_dtype = serialized_arr[:i_0].decode('utf-8')
    arr_shape = tuple([int(a) for a in serialized_arr[i_0 + 1:i_1].decode('utf-8').split(',')])
    arr_str = serialized_arr[i_1 + 1:]
    arr = np.frombuffer(arr_str, dtype = arr_dtype).reshape(arr_shape)
    return arr


def fits_open(filepath):
    """
    A wrapper around fits.open to transform SHARP magnetogram to SMARP

    Resolution: SHARP: CDELT1 = 0.03 (deg), CROTA2 = 0
                SMARP: CDELT1 = 0.12 (deg), CROTA2 = 0
    """
    data = fits.open(filepath)[1].data
    if 'sharp' in filepath:
        data = data[::4, ::4]
        k = 1.098679
        b = -0.462782
        data = k * data + b
    return data


def query(filepaths, redis=True):
    """Query FITS image file(s) using filepath(s).

    If `redis` is True, a Redis server should be running.
    """
    single_file = isinstance(filepaths, str)
    if single_file:
        filepaths = [filepaths]

    if redis:
        buff = r_image.mget(filepaths)
        indices = [i for i, b in enumerate(buff) if b is None]
        if len(indices) > 0:
            keys = [filepaths[i] for i in indices]
            values = [toRedis(fits_open(k).astype(np.float16)) for k in keys]
            r_image.mset(dict(zip(keys, values)))
            for j, i in enumerate(indices):
                buff[i] = values[j]
        data_arrays = [fromRedis(b).astype(np.float32) for b in buff]
    else:
        data_arrays = [fits_open(k) for k in filepaths]

    data = np.stack(data_arrays)
    if single_file:
        data = data[0]
    return data


def query_parameters(prefix, arpnum, t_recs, keywords, redis=True):
    """Query keyword sequence from a header file.

    Alternative design: replace prefix and arpnum with filepath.
    """
    KEYWORDS = ['T_REC', 'AREA', 'USFLUX', 'MEANGBZ', 'R_VALUE']

    if redis:
        id = f'{prefix}{arpnum:06d}' # header file identifier
        if r_header.exists(id) == 0:
            dataset = 'sharp' if prefix == 'HARP' else 'smarp'
            header = read_header(dataset, arpnum)
            header = header[KEYWORDS]
            header = header.set_index('T_REC')
            mapping = {t_rec: header.loc[t_rec].to_json() for t_rec in header.index}
            r_header.hmset(id, mapping)
        buff = r_header.hmget(id, t_recs)
        # series = [pd.read_json(b, typ='series') if b else None for b in buff]
        # if any([s is None for s in series]):
        #     print(series)
        records = [json.loads(b) if b else {} for b in buff]
        df = pd.DataFrame(records, index=t_recs)
    else:
        raise
    return df


if __name__ == '__main__':
    from glob import glob
    from tqdm import tqdm

    print("=========== TEST =============")

    filepaths = sorted(glob(os.path.join(DATA_DIR, 'SHARP/image/000001/*')))
    filepaths = np.random.permutation(filepaths)[20:]
    for filepath in tqdm(filepaths):
        data = query(filepath)

