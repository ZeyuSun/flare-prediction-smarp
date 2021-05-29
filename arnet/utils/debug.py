def plot(data, *args, **kwargs):
    import numpy as np
    import matplotlib; matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import torch
    if isinstance(data, torch.Tensor):
        data = data.double().detach().cpu().numpy() #float16 doesn't work
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

    if data.ndim == 1:
        args = list(args)
        if isinstance(args[0], torch.Tensor):
            args[0] = args[0].detach().cpu().numpy()
        plt.plot(data, *args, **kwargs)
        plt.legend()
    elif data.ndim == 2:
        plt.imshow(data, *args, **kwargs)
        plt.colorbar()
    elif data.ndim == 3:
        if data.shape[2] not in {1, 3}:
            print("last dimension should be 1 or 3")
            raise
        plt.imshow(data)
        plt.colorbar()
    elif data.ndim == 4:
        print('N, C, T, H, W')
    plt.show()


def imshow(prefix, arpnum, t_rec):
    import os
    from datetime import datetime
    import matplotlib; matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from arnet.utils import fits_open

    DATA_DIRS = {
        'HARP': '/data2/SHARP/image/',
        'TARP': '/data2/SMARP/image/',
    }
    SERIES = {
        'HARP': 'hmi.sharp_cea_720s',
        'TARP': 'mdi.smarp_cea_96m',
    }
    T_REC_FORMAT = '%Y.%m.%d_%H:%M:%S_TAI'

    t = datetime.strptime(t_rec, T_REC_FORMAT).strftime('%Y%m%d_%H%M%S_TAI')
    filename = f'{SERIES[prefix]}.{arpnum}.{t}.magnetogram.fits'
    filepath = os.path.join(DATA_DIRS[prefix], f'{arpnum:06d}', filename)
    data = fits_open(filepath)
    plt.imshow(data)
    plt.show()


def check(tensor):
    """check nan and inf"""
    import numpy as np
    array = tensor.detach().cpu().numpy() # torch.any can only reduce one dim
    nans = np.isnan(array)
    infs = np.isinf(array)
    if nans.any():
        for i in range(nans.ndim):
            axis = tuple([j for j in range(nans.ndim) if j != i]) # has to be a tuple, list won't work
            print(i, nans.any(axis=axis))

    if infs.any():
        for i in range(infs.ndim):
            axis = tuple([j for j in range(infs.ndim) if j != i]) # has to be a tuple, list won't work
            print(i, infs.any(axis=axis))

    if not (nans.any() or infs.any()):
        print('Good tensor!')