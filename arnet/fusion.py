from pathlib import Path
import numpy as np
import pandas as pd


def load_csv_dataset(csv_path, **kwargs):
    df = pd.read_csv(csv_path, **kwargs)
    df.loc[:, 'flares'] = df['flares'].fillna('')
    df.loc[:, 'bad_img_idx'] = df['bad_img_idx'].apply(
        lambda s: [int(x) for x in s.strip('[]').split()])
    return df


def load_fusion_dataset(auxdata):
    d = np.load(auxdata, allow_pickle=True).item()
    return d


def fuse_sharp_to_smarp(df, fuse_dict):
    for k, v in fuse_dict.items():
        if k in df.columns:
            df[k] = df[k] * v['coef'] + v['intercept']
    return df


def group_split_data(df, seed=None):
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df['arpnum']))
    return df.iloc[train_idx], df.iloc[test_idx]


def group_split_data_cv(df, cv=5, split=0):
    """
    Args:
        cv: number of cv folds
        split: index of the cv fold to return
    Note that GroupKFold is not random
    """
    from sklearn.model_selection import GroupKFold
    splitter = GroupKFold(n_splits=cv)
    split_generator = splitter.split(df, groups=df['arpnum'])
    for k, (train_idx, test_idx) in enumerate(split_generator):
        if k == split:
            return df.iloc[train_idx], df.iloc[test_idx]


def rus(df, sizes='balanced', seed=False):
    """Random Undersampling

    Args:
        sizes: dict of class sizes or 'balanced'
        seed: default is False, do not reset seed. Pass int/None to reset a
            seed particularly/randomly.
    """
    import numpy as np
    if seed is None or isinstance(seed, int):
        np.random.seed(seed)

    neg = np.where(~df['label'])[0]
    pos = np.where(df['label'])[0]
    if sizes == 'balanced':
        sizes = {0: len(pos), 1:len(pos)}
    elif isinstance(sizes, float):
        # sizes = p/n after rus
        #   ranges 0.03 (no rus) to 1 (balanced)
        # compute rus rate (the proportion of neg samples to keep)
        #   rus rate: 1 to 0.03
        rus_rate = (len(pos) / sizes) / len(neg)
        rus_rate = min(sizes, 1) # prevent raise in np.random.choice
        sizes = {0: int(rus_rate * len(neg)), 1: len(pos)}
    idx = np.concatenate((
        np.random.choice(neg, size=sizes[0], replace=False),
        np.random.choice(pos, size=sizes[1], replace=False),
    ))
    idx = np.sort(idx)
    df = df.iloc[idx].reset_index(drop=True)
    return df


def get_datasets(database, dataset, auxdata,
                 sizes=None, validation=False, shuffle=False, seed=None,
                 val_split=0, test_split=0,
                 rus_test=True,
    ):
    """
    Args:
        sizes: Dict of desired class sizes. None: no rus. 'balanced': balanced rus.
    """
    df_smarp = load_csv_dataset(Path(database) / 'smarp.csv')
    df_sharp = load_csv_dataset(Path(database) / 'sharp.csv', low_memory=False)
    # Warning: Col(7) in sharp.csv (maybe noaa_ars) has mixed type. SMARP doesn't have this warning.
    # low_memory=False is as fast as low_memory=True (perhaps the same implementation)
    # low_memory=False is 5x faster than engine=python
    fuse_dict = load_fusion_dataset(Path(auxdata))
    # Two keys are outdated. fuse_dict =
    #{'MEANGBZ': {'coef': 1.9920261748674042, 'intercept': 8.342889969768606},
    #  'USFLUX': {'coef': 1.216160520290385, 'intercept': -3.8777994451166115e+20},
    #  'R_VALUE': {'coef': 0.8327836641793915, 'intercept': -0.0945601961295528}}
    df_sharp = fuse_sharp_to_smarp(df_sharp, fuse_dict)

    # Cross validation split. No randomness.
    if val_split is None and test_split is None:
        # This is how I split before cv was implemented
        if dataset in ['sharp', 'fused_sharp']:
            df_train, df_test = group_split_data(df_sharp, seed=seed)
            if validation:
                df_train, df_val = group_split_data(df_train, seed=seed)
            if dataset == 'fused_sharp':
                df_train = pd.concat((df_train, df_smarp)).reset_index(drop=True)
        elif dataset in ['smarp', 'fused_smarp']:
            df_train, df_test = group_split_data(df_smarp, seed=seed)
            if validation:
                df_train, df_val = group_split_data(df_train, seed=seed)
            if dataset == 'fused_smarp':
                df_train = pd.concat((df_train, df_sharp)).reset_index(drop=True)
    else:
        # If either of them is not None, then this is after cv was implemented
        # We initialize None with 0
        val_split = val_split or 0
        test_split = test_split or 0
        if dataset in ['sharp', 'fused_sharp']:
            df_train, df_test = group_split_data_cv(df_sharp, cv=5, split=test_split)
            if validation:
                df_train, df_val = group_split_data_cv(df_train, cv=5, split=val_split)
            if dataset == 'fused_sharp':
                df_train = pd.concat((df_train, df_smarp)).reset_index(drop=True)
        elif dataset in ['smarp', 'fused_smarp']:
            df_train, df_test = group_split_data_cv(df_smarp, cv=5, split=test_split)
            if validation:
                df_train, df_val = group_split_data_cv(df_train, cv=5, split=val_split)
            if dataset == 'fused_smarp':
                df_train = pd.concat((df_train, df_sharp)).reset_index(drop=True)

    if sizes: # Why rus after split? Strict ratio; Option to rus only train
        df_train = rus(df_train, sizes=sizes, seed=seed)
        if validation: # validation controls both existence of balance
            df_val = rus(df_val, sizes=sizes, seed=seed)
        if rus_test: # rus_test only controls existence
            df_test = rus(df_test, sizes=rus_test, seed=seed)

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)

    if validation:
        return df_train, df_val, df_test
    else:
        return df_train, df_test
