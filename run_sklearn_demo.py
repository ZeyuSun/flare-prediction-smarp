# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix


cfg = {
    'features': ['AREA', 'USFLUXL', 'MEANGBL', 'R_VALUE'],
}


def group_split_data(df):
    split_generator = GroupKFold(n_splits=5).split(df, groups=df['arpnum'])
    train_idx, test_idx = next(split_generator)
    return df.iloc[train_idx], df.iloc[test_idx]


def get_dataset_from_df(df):
    X = df[cfg['features']].to_numpy()
    y = df['label'].to_numpy()
    groups = (df['prefix'] + df['arpnum'].apply(str)).to_numpy()
    # groups = ['HARP248', 'HARP248', 'HARP7487', ...]
    return X, y, groups


def standardize_data(X_train, X_test):
    X_mean = X_train.mean(0)
    X_std = X_train.std(0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    return X_train, X_test


def get_scores_from_cm(cm):
    [[TN, FP], [FN, TP]] = cm
    N = TN + FP
    P = TP + FN
    precisions = np.diagonal(cm) / np.sum(cm, 0)
    recalls = np.diagonal(cm) / np.sum(cm, 1)
    f1 = 2 * precisions[1] * recalls[1] / (precisions[1] + recalls[1])
    pod = recalls[1]
    far = 1 - recalls[0]
    tss = pod - far
    hss1 = (TP + TN -N) / P
    hss2 = 2 * (TP * TN - FN * FP) / (P * (FN+TN) + (TP+FP) * N)
    scores = {
        'precision': precisions[1],
        'recall': recalls[1],
        'accuracy': (TP + TN) / (N + P),
        'f1': f1,
        'tss': tss,
        'hss1': hss1,
        'hss2': hss2,
    }
    return scores


# %% Get dataset
df_sharp = pd.read_csv('datasets/M_Q_24hr/sharp.csv', low_memory=False)
df_sharp = df_sharp.sample(n=10000, replace=False)
df_sharp.loc[:, 'flares'] = df_sharp['flares'].fillna('')
df_train, df_test = group_split_data(df_sharp)  # grouped by HARP number

X_train, y_train, g_train = get_dataset_from_df(df_train)
X_test, y_test, g_test = get_dataset_from_df(df_test)

X_train, X_test = standardize_data(X_train, X_test)

# %% Train model
Model = SGDClassifier
parameters = {
    'loss': 'log_loss',
    'alpha': 0.0001,
    'class_weight': 'balanced',
}
model = Model(**parameters)
model.fit(X_train, y_train)

# %% Evaluate model
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)

cm = confusion_matrix(y_test, y_pred)
scores = get_scores_from_cm(cm)
print(scores)