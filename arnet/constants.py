import os
import json
from functools import lru_cache
import numpy as np
import pandas as pd


FEATURES = ['AREA', 'USFLUXL', 'MEANGBL', 'R_VALUE', 'FLARE_INDEX']
#TODO: constants calculated on training set
# relative address: code may be run from other directory, e.g., notebooks/
# absolute address: different machine
# Should use a env variable
PROCESSED_DATA_DIR = '/home/zeyusun/work/flare-prediction-smarp/datasets/M_Q_24hr/'

@lru_cache(8)
def get_constants():
    CONSTANTS = {}
    for dataset in ['sharp', 'smarp']:
        filepath = os.path.join(PROCESSED_DATA_DIR, f'{dataset}.csv')
        df = pd.read_csv(filepath)

        CONSTANTS[dataset.upper() + '_MEAN'] = df[FEATURES].mean().to_dict()
        CONSTANTS[dataset.upper() + '_STD'] = df[FEATURES].std().to_dict()
    return CONSTANTS
CONSTANTS = get_constants()
#print(json.dumps(CONSTANTS, indent=2))

#sharp2smarp = np.load('datasets/sharp2smarp.npy', allow_pickle=True).item()
