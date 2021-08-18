def file_scanning(paths):
    """
    Args:
        paths (str/list/tuple): a directory path or
            a (potentially nested) list/tuple of directory paths
    Returns:
        A list of all files under `paths`
    """
    import os
    import typing
    if isinstance(paths, typing.List) or isinstance(paths, typing.Tuple):
        filepaths = []
        for d in paths:
            filepaths.extend(file_scanning(d))
        return filepaths
    elif isinstance(paths, str):
        return [os.path.join(root, fn)
                for root, _, filenames in os.walk(paths)
                for fn in filenames]
    else:
        raise TypeError("Argument type not accepted: {}".format(paths))


def array_to_uint8(a, low=0, high=100):
    """
    a: array-like object
    low: lower percentile
    high: higher percentile
    """
    import numpy as np
    vmin = np.percentile(a, low)
    vmax = np.percentile(a, high)
    a = np.clip(a, a_min=vmin, a_max=vmax)
    a = ((a - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return a


def array_to_float_video(a, low=None, high=None, perc=True):
    """Truncate and rescale an array to float values in [0,1] for visualization
    a: array-like object
    low: lower percentile, unless `high=None`, in which case this parameter is assigned to high and 0 is assigned to low.
    high: higher percentile. (See above for behavior if `high=None`)
    TODO: change doc
    """
    import numpy as np
    if perc:
        low = low or 0
        high = high or 100
        vmin = np.percentile(a, low)
        vmax = np.percentile(a, high)
    else:
        vmin = low or np.min(a)
        vmax = high or np.max(a)
    a = np.clip(a, a_min=vmin, a_max=vmax)
    a = (a - vmin) / (vmax - vmin)
    return a


def generate_batch_info_classification(videos, meta, y_true, y_prob):
    """
    Arguments: numpy arrays or list
    Returns: Dataframe
    """
    import os
    import numpy as np
    import pandas as pd

    axes = (1,2,3,4)
    y_prob_round = [round(p, 2) for p in y_prob]
    d = {
        'min': np.min(videos, axis=axes), # if videos is tensor, axes is also expected to be
        '98-perc': np.percentile(videos, 98, axis=axes), # tried to convert arg to numpy ndarray
        'max': np.min(videos, axis=axes),
        'idx': [],
        'harp_num': [],
        'start_time': [],
        'h': [],
        'w': [],
        'flare': [],
        'y_true': y_true,
        'y_prob': y_prob_round,
    }
    for m in meta:
        info = os.path.basename(m).replace(".npy", "").split("_")
        d['idx'].append(int(info[0]))
        d['harp_num'].append(int(info[1][4:]))
        d['start_time'].append(info[2])
        d['h'].append(int(info[3][1:]))
        d['w'].append(int(info[4][1:]))
        d['flare'].append(info[5])
    df = pd.DataFrame(d)
    return df


def generate_batch_info_regression(videos, meta, i_true, i_hat, q_prob):
    """
    Arguments: numpy arrays or list
    Returns: Dataframe
    """
    import os
    import numpy as np
    import pandas as pd

    import arnet.utils as utils

    axes = (1,2,3,4)
    c_true = [utils.get_flare_class(i) for i in i_true]
    c_hat = [utils.get_flare_class(i) for i in i_hat]
    q_prob = [round(q, 2) for q in q_prob]
    c_pred = ['Q' if q > 0.5 else c for q, c in zip(q_prob, c_hat)]
    d = {
        'min': np.min(videos, axis=axes), # if videos is tensor, axes is also expected to be
        '25-perc': np.percentile(videos, 25, axis=axes),  # tried to convert arg to numpy ndarray
        '75-perc': np.percentile(videos, 75, axis=axes),  # tried to convert arg to numpy ndarray
        '98-perc': np.percentile(videos, 98, axis=axes), # tried to convert arg to numpy ndarray
        'max': np.max(videos, axis=axes),
        'idx': [],
        'harp_num': [],
        'start_time': [],
        'h': [],
        'w': [],
        'flare': [],
        'c_true': c_true,
        'c_hat': c_hat,
        'c_pred': c_pred,
    }
    for m in meta:
        info = os.path.basename(m).replace(".npy", "").split("_")
        d['idx'].append(int(info[0]))
        d['harp_num'].append(int(info[1][4:]))
        d['start_time'].append(info[2])
        d['h'].append(int(info[3][1:]))
        d['w'].append(int(info[4][1:]))
        d['flare'].append(info[5])
    df = pd.DataFrame(d)
    return df


if __name__ == '__main__':
    import numpy as np
    n = 8
    videos = np.random.randn(n,1,12,32,32)
    dates = ['this date'] * n
    harps = np.random.randint(0,16,n)
    heights = np.random.randint(0,16,n)
    widths = np.random.randint(0,16,n)
    flares = ['A' for i in range(n)]
    y_true = np.arange(n)
    y_prob = np.arange(n)
    # Deprecated test
    print(generate_batch_video_info(videos, dates, harps, heights, widths, flares=flares, y_true=y_true, y_prob=y_prob).to_markdown())
