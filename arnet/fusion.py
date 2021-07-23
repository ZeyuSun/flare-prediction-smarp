from pathlib import Path
import numpy as np
import pandas as pd


def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
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


def group_split_data_cv(df, cv=5):
    # GroupKFold is not random
    from sklearn.model_selection import GroupKFold
    splitter = GroupKFold(n_splits=cv)
    for train_idx, test_idx in splitter.split(df, groups=df['arpnum']):
        yield df.iloc[train_idx], df.iloc[test_idx]


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
    idx = np.concatenate((
        np.random.choice(neg, size=sizes[0], replace=False),
        np.random.choice(pos, size=sizes[1], replace=False),
    ))
    idx = np.sort(idx)
    df = df.iloc[idx].reset_index(drop=True)
    #TODO: why bother reset index when split and shuffle don't
    return df


def get_datasets(database, dataset, auxdata,
                 sizes=None, validation=0, shuffle=False, seed=None):
    """
    Args:
        sizes: Dict of desired class sizes. None: no rus. 'balanced': balanced rus.
        validation (int): 0 means no validation, 1 means a hold-out validation.
            >=2 means cross-validation.
        shuffle: only meaningful for train split.
    Note:
        index is not to be trusted. RUS resets but others ops don't.
    """
    df_smarp = load_csv_dataset(Path(database) / 'smarp.csv')
    df_sharp = load_csv_dataset(Path(database) / 'sharp.csv')
    fuse_dict = load_fusion_dataset(Path(auxdata))
    df_sharp = fuse_sharp_to_smarp(df_sharp, fuse_dict)

    if dataset == 'fused_sharp':
        df_sharp_train, df_sharp_test = group_split_data(df_sharp, seed=seed)
        df_train = pd.concat((df_smarp, df_sharp_train)).reset_index(drop=True)
        df_test = df_sharp_test
    elif dataset == 'fused_smarp':
        df_smarp_train, df_smarp_test = group_split_data(df_smarp, seed=seed)
        df_train = pd.concat((df_sharp, df_smarp_train)).reset_index(drop=True)
        df_test = df_smarp_test
    elif dataset == 'sharp':
        df_train, df_test = group_split_data(df_sharp, seed=seed)
    elif dataset == 'smarp':
        df_train, df_test = group_split_data(df_smarp, seed=seed)

    if validation == 1:
        df_train, df_val = group_split_data(df_train, seed=seed)
    elif validation > 1:
        cv_splits = group_split_data_cv(df_train, cv=validation)

    if sizes:
        # Why rus after split? Strict ratio; Option to rus only train;
        # O.w., GroupKFold starts with starts longest AR after rus. But why would that be a problem?
        df_test = rus(df_test, sizes=sizes, seed=seed)
        if validation == 0:
            df_train = rus(df_train, sizes=sizes, seed=seed)
        elif validation == 1:
            df_train = rus(df_train, sizes=sizes, seed=seed)
            df_val = rus(df_val, sizes=sizes, seed=seed)
        else:
            cv_splits = (
                (
                    rus(df_train, sizes=sizes, seed=seed),
                    rus(df_val, sizes=sizes, seed=seed),
                )
                for  df_train, df_val in cv_splits
            )

    if shuffle:
        if validation <= 1:
            df_train = df_train.sample(frac=1, random_state=seed)
        else:
            cv_splits = (
                (
                    df_train.sample(frac=1, random_state=seed),
                    df_val,
                )
                for  df_train, df_val in cv_splits
            )


    if validation == 0:
        return df_train, df_test
    elif validation == 1:
        #TODO: maybe should return cv_splits and use next() if only one is needed
        return df_train, df_val, df_test
    else:
        return cv_splits, df_test

