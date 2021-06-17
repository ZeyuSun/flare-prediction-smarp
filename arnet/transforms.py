import math
import torch
import torch.nn.functional as F
from fvcore.common.registry import Registry

from arnet.constants import CONSTANTS # run in the flare-


TRANSFORM_REGISTRY = Registry("TRANSFORM")
TRANSFORM_REGISTRY.__doc__ = """Registry for input transforms."""


@TRANSFORM_REGISTRY.register()
class CenterCropPad():
    """Center cropping and zero padding a 3D tensor to a target size.

    If both `crop` and `pad` are true, then the output size is exactly `target_size`.

    Arguments:
        target_size: 3-tuple (D, H, W). A None value outputs the dimension as-is.
        crop: If true, the output size is no larger than `target_size`.
        pad: If true, the output size is no smaller than `target_size`.
    """
    def __init__(self, target_size, crop=True, pad=True, value=0):
        self.target_size = target_size
        self.crop = crop
        self.pad = pad
        self.value = value

    def __call__(self, video):
        C, D, H, W = video.shape
        d, h, w = self.target_size
        d = d or D
        h = h or H
        w = w or W
        offsets = torch.tensor([(D - d)/2, (H - h)/2, (W - w)/2])

        if self.crop:
            off = F.relu(offsets).int()
            video = video[:,
                          off[0]:off[0]+d,
                          off[1]:off[1]+h,
                          off[2]:off[2]+w]
        if self.pad:
            off = F.relu(-offsets)
            padding = [
                math.floor(off[-1]), math.ceil(off[-1]),
                math.floor(off[-2]), math.ceil(off[-2]),
                math.floor(off[-3]), math.ceil(off[-3]),
                0, 0
            ]
            video = F.pad(video, padding, value=self.value)
        return video


@TRANSFORM_REGISTRY.register()
class Resize():
    """Take a video of C x T x H x W and resize spatial dimension.

    Args:
        target_size: (H, W)
    """
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, video):
        C, T, H, W = video.shape
        dt = torch.linspace(-1, 1, T)
        dh = torch.linspace(-1, 1, self.target_size[0])
        dw = torch.linspace(-1, 1, self.target_size[1])
        meshz, meshy, meshx = torch.meshgrid([dt, dh, dw])
        grid = torch.stack((meshx, meshy, meshz), 3)
        resized_video = F.grid_sample(
            video.unsqueeze(0),
            grid.unsqueeze(0),
            align_corners=True, # default False # omitting gives warnings
            mode="bilinear")[0]
        #interpolate is upsampling, we need down sampling
        return resized_video


@TRANSFORM_REGISTRY.register()
class ValueTransform():
    def __init__(self, shrinkage='1/2', thresh=236):
        """
        Args:
            shrinkage (str): shrinkage method for large values
            thresh (numerical): default value 236 from http://jsoc.stanford.edu/data/hmi/HMI_M.ColorTable.pdf
        """
        self.shrinkage = shrinkage
        self.thresh = thresh

    def __call__(self, tensor):
        if self.shrinkage == '1/2':
            shrink = lambda a: torch.sign(a) * self.thresh * torch.abs(a / self.thresh) ** (1 / 2)
        elif self.shrinkage == '1/3':
            shrink = lambda a: torch.sign(a) * self.thresh * torch.abs(a / self.thresh) ** (1 / 3)
        elif self.shrinkage == 'log':
            c = math.log(1 + self.thresh) / self.thresh
            shrink = lambda a: torch.sign(a) * torch.log(1 + torch.abs(a)) / c
        else:
            raise

        tensor = torch.where(
            torch.abs(tensor) <= self.thresh,
            tensor,
            shrink(tensor)
        )
        return tensor


@TRANSFORM_REGISTRY.register()
class Standardize():
    def __init__(self, mean=0, std=50):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / self.std
        return tensor


@TRANSFORM_REGISTRY.register()
class Reverse():
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor[::-1]


def calc_stats(hist, bins, func=None):
    import numpy as np
    mids = 0.5 * (bins[1:] + bins[:-1])
    if func is not None:
        mids = func(torch.tensor(mids)).numpy()
    mean = np.average(mids, weights=hist)
    var = np.average((mids - mean) ** 2, weights=hist)
    std = np.sqrt(var)
    return mean, std


def get_transform_kwargs(name, cfg):
    import numpy as np

    if name == 'CenterCropPad':
        kwargs = {'target_size': (None, cfg.DATA.HEIGHT, cfg.DATA.WIDTH)}
    elif name == 'Resize':
        kwargs = {'target_size': (cfg.DATA.HEIGHT, cfg.DATA.WIDTH)}
    elif name == 'Standardize':
        if 'MAGNETOGRAM' in cfg.DATA.FEATURES:
            hist = np.load('datasets/sharp_hist.npy', allow_pickle=True).item() # TODO: config
            func = get_transform('ValueTransform', cfg) if 'ValueTransform' in cfg.DATA.TRANSFORMS else None
            mean, std = calc_stats(hist['hist'], hist['bins'], func=func)
            cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD = float(mean), float(std)
            kwargs = {'mean': mean, 'std': std}
        else:
            kwargs = {
                'mean': {k: CONSTANTS['SMARP_MEAN'][k] for k in cfg.DATA.FEATURES},
                'std': {k: CONSTANTS['SMARP_STD'][k] for k in cfg.DATA.FEATURES},
            }
    elif name == 'ValueTransform':
        kwargs = {'shrinkage': cfg.DATA.SHRINKAGE, 'thresh': cfg.DATA.THRESH}
    else:
        kwargs = {}
    return kwargs


def get_transform(name, cfg):
    kwargs = get_transform_kwargs(name, cfg)
    transform = TRANSFORM_REGISTRY.get(name)(**kwargs)
    return transform


def test_ValueTransform():
    import matplotlib.pyplot as plt
    from itertools import product

    threshes = [50, 150]
    xx = torch.linspace(-1500, 1500, 100)
    shrinks = ['log', '1/2']
    for s, t in product(shrinks, threshes):
        label = f'{s} @ {t:d}'
        plt.plot(xx, ValueTransform(shrinkage=s, thresh=t)(xx), label=label)
    plt.plot([-t, t], [-t, t], 'o')
    plt.legend()
    # plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    # size = (3,3)
    # ccp = CenterCropPad(size)
    # for h in range(1,5):
    #     for w in range(1,5):
    #         X = torch.ones((2,2,h,w))
    #         output_size = ccp(X).shape[-2:]
    #         assert output_size[0] == size[0]
    #         assert output_size[1] == size[1]
    #         output_size = ccp(X, crop=False)

    test_ValueTransform()
