import os
import json
from functools import lru_cache
import numpy as np
import pandas as pd


FEATURES = ['AREA', 'USFLUX', 'MEANGBZ', 'R_VALUE', 'FLARE_INDEX']

@lru_cache
def get_constants():
    CONSTANTS = {}
    for dataset in ['sharp', 'smarp']:
        filepath = os.path.join('datasets', dataset, 'train.csv')
        df = pd.read_csv(filepath)

        CONSTANTS[dataset.upper() + '_MEAN'] = df[FEATURES].mean().to_dict()
        CONSTANTS[dataset.upper() + '_STD'] = df[FEATURES].std().to_dict()
    return CONSTANTS
CONSTANTS = get_constants()
#print(json.dumps(CONSTANTS, indent=2))

#sharp2smarp = np.load('datasets/sharp2smarp.npy', allow_pickle=True).item()