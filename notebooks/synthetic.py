from itertools import product
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from cotrain_helper import get_learner_by_query


def unipole(points, B0, r, x, y):
    numerator = np.sum((points - [x, y]) ** 2)
    B = B0 * np.exp(- numerator / (2 * r ** 2))
    return B


def bipole_gaussian(points, B0, rho, gamma, sigma):
    """ Two gaussians with opposite sign
    Args:
        points: points with coordinates in the last dimension.
        B0: maximum magnitude of each gaussian. (May cancel if rho is small)
        rho: polarity separation.
        gamma: tilt angle of the positive pole in radian.
        sigma: standard deviation of gaussian.

    Return:
        Z: ndarray of shape points.shape[:-1], i.e., the last dimension of `points` is reduced while other dimensions preserved.
    """
    c = np.array([0, 0])
    c1 = c + rho / 2 * np.array([np.cos(gamma), np.sin(gamma)])
    c2 = -c1 #c2 = c - rho / 2 * np.array([np.cos(gamma), np.sin(gamma)])
    Z1 = np.exp(-np.sum((points - c1) ** 2 / (2 * sigma ** 2), axis=-1))
    Z2 = np.exp(-np.sum((points - c2) ** 2 / (2 * sigma ** 2), axis=-1))
    Z = B0 * (Z1 - Z2)
    return Z


def bipole_yeates2020(points, B0, rho, gamma, a=0.56):
    """
    Args:
        points: points with coordinates in the last dimension.
        B0: amplitude. (NB: not max abs value)
        rho: polarity separation in radian.
        gamma: tilt angle wrt the equator in radian.
        a: size of BMR relative to rho
        
    Note: Yeates 2020, eq (5)
    """
    R = np.array([
        [np.cos(gamma), np.sin(gamma)],
        [-np.sin(gamma), np.cos(gamma)],
    ]) # rotation matrix (-gamma)
    _points = np.einsum('ij,klj->kli', R, points)
    A = - (B0 / rho) * _points[:,:,0]  # factor before exp
    numerator = (_points ** 2).dot([1,2])  # x^2 + 2 * y^2
    B = A * np.exp(- numerator / (a * rho)**2)
    return B


def sweep_constant_images(learner, lim=10):
    assert lim > 0
    with torch.no_grad():
        input = (torch.ones(1, 1, 128, 128)
                 .to(learner.device))
        const = np.linspace(-lim, lim, 51)
        dataloader = DataLoader(
            [input * c for c in const],
            batch_size=16,
            #num_workers=8,
            #pin_memory=True,
        )
        probs = []
        for batch in dataloader:
            output = learner(batch)
            prob = F.softmax(output)[:, 1].tolist()
            #prob = (output[0, 1] - output[0, 0]).item()
            probs.extend(prob)
        probs = probs
    return const, probs


def sweep_Z_list(learner, Z_list):
    dataset = [(torch.tensor(np.expand_dims(Z, axis=(0,1)))
                .to(learner.device)
                .to(torch.float32))
               for Z in Z_list]

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        #num_workers=4,
        #pin_memory=True,
        # Setting num_worker > 0 causes problem
        # Change fork to spawn: torch.multiprocessing.set_start_method('spawn')
        # It works but loading is very slow
    )

    probs = []
    for Z_batch in tqdm(dataloader):
        probs.extend(F.softmax(learner(Z_batch), dim=-1)[:, 1].tolist())
    probs = np.array(probs)
    return probs


def sweep_learner_and_Z_list(function, XY, params, names):
    """
    Args:
        function: unipole, bipole_gaussian, or bipole_yeates2020
        XY: first argument of function. Coordinate systems
        params: other args of function. A list of parameters.
        names: names of the parameteres. Used to name columns in df
    """
    df = pd.DataFrame(data=params, columns=names)
    Z_list = [function(XY, *args) for args in params]

    dataset = 'sharp'
    for val_split, test_split in tqdm(list(product(range(5), range(5)))):
        query = f'cv/base/{dataset}/0/{val_split}/{test_split}/CNN'
        learner = get_learner_by_query(query, eval_mode=True, device='cuda:1')
        probs = sweep_Z_list(learner, Z_list)
        df[f'prob_{val_split}_{test_split}'] = probs
    df['prob'] = df[[c for c in df.columns if c[:5] == 'prob_']].mean(axis=1)

    #(df[[c for c in df.columns if c[:5] == 'prob_']]
    # .describe()
    # .style
    # .background_gradient(axis=1)
    #)
    #df['prob'].hist(bins=20)

    #df_pos = df[df['prob'] > 0.5]
    #Z_list_pos = [Z_list[i] for i in df_pos.index]
    #print(len(df_pos))
    return df, Z_list