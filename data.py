import os
import logging
import numpy as np
import pandas as pd
import redis
from astropy.io import fits


r = redis.Redis(db=13)


def read_header(dataset, arpnum):
    """
    Returns:
        header dataframe or None if no sharp_los found
    """
    FEATURES_LOS = ['USFLUX', 'MEANGBZ', 'R_VALUE']
    FEATURES = ['AREA', 'USFLUX', 'MEANGBZ', 'R_VALUE']

    if dataset == 'sharp':
        header = pd.read_csv(f'/data2/SHARP/header/HARP{arpnum:06d}_ATTRS.csv', index_col='T_REC')
        header[FEATURES_LOS] = np.nan
        header_los_path = f'/data2/SHARP/header_los/HARP{arpnum:06d}_ATTRS.csv'
        if not os.path.exists(header_los_path):
            logging.warning('%s: file not found. Return None.' % header_los_path)
            return None
        header_los = pd.read_csv(header_los_path, index_col='T_REC')
        if not header.index.equals(header_los.index):
            logging.warning('Header t_recs mismatch: header_los and header of HARP %d' % arpnum)
        header.update(header_los[FEATURES_LOS])
        header.reset_index(inplace=True)
    elif dataset == 'smarp':
        header = pd.read_csv(f'/data2/SMARP/header/TARP{arpnum:06d}_ATTRS.csv')
    else:
        raise
    return header


def read_goes():
    GOES_HMI = '/home/zeyusun/SOLSTICE/goes/GOES_HMI.csv'
    GOES_MDI = '/home/zeyusun/SOLSTICE/goes/GOES_MDI.csv'

    goes = pd.concat((
        pd.read_csv(GOES_MDI, index_col=0),
        pd.read_csv(GOES_HMI, index_col=0)
    ))
    goes['goes_class'].fillna('', inplace=True)
    goes = goes.drop_duplicates(ignore_index=True)
    goes = goes.reset_index()
    return goes


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
        buff = r.mget(filepaths)
        indices = [i for i, b in enumerate(buff) if b is None]
        if len(indices) > 0:
            keys = [filepaths[i] for i in indices]
            values = [toRedis(fits_open(k).astype(np.float16)) for k in keys]
            r.mset(dict(zip(keys, values)))
            buff = [values[i] if i in indices else b for i, b in enumerate(buff)]
        data_arrays = [fromRedis(b).astype(np.float32) for b in buff]
    else:
        data_arrays = [fits_open(k) for k in keys]

    data = np.stack(data_arrays)
    if single_file:
        data = data[0]
    return data


if __name__ == '__main__':
    import os
    from glob import glob
    from tqdm import tqdm
    import numpy as np

    filepaths = sorted(glob('/data2/SHARP/image/000001/*'))
    filepaths = filepaths[:20]
    #filepaths += ['/home/data/magnetogram_data/image/HARP004225.npy']
    #filepaths += ['/home/data/magnetogram_data/image/HARP003784.npy']
    #filepaths += ['/home/data/magnetogram_data/image/HARP001126.npy']
    #print([[f, '{:.2f}MB'.format(os.stat(f).st_size / 1024 / 1024)] for f in filepaths])

    #with r.pipeline() as pipe:
    #    for filepath in tqdm(filepaths):
    #        r_load(r, filepath)
    #    pipe.execute()

    print("=========== TEST =============")
    filepaths = np.random.permutation(filepaths)
    for filepath in tqdm(filepaths):
        data = r_query(filepath)

