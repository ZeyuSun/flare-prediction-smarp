def get_flare_index(flares):
    """Daily SXR flare index (Abramenko 2005)"""
    weights = {
        'X': 100,
        'M': 10,
        'C': 1,
        'B': 0.1,
        'A': 0,
    }
    flare_index = 0
    for f in flares:
        if f == '':
            continue
        if f == 'C':
            continue
        flare_index += weights[f[0]] * float(f[1:])
    flare_index = round(flare_index, 1) # prevent numerical error
    return flare_index


def get_log_intensity(class_str):
    import numpy as np

    if class_str == '' or class_str == 'Q':
        return -9. # torch.floor not implemented for Long

    class_map = {
        'A': -8.,
        'B': -7.,
        'C': -6.,
        'M': -5.,
        'X': -4.,
    }
    a = class_map[class_str[0]]
    b = np.log10(float(class_str[1:]))
    return a + b


def get_output(model, X):
    if hasattr(model, 'decision_function'):
        y_score = model.decision_function(X)
    elif hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X)[:,1]
    else:
        raise
    return y_score


def draw_pairplot(Xs, labels, keys):
    """
    Returns:
        pair_grid (PairGrid): has attribution `figure` and `layout`
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    # If drawing density, should use stratefied sampling
    # Why so complicated?
    N = 2000
    Xs = [X[np.random.choice(len(X), size=N, replace=False)] if len(X) > N else X
          for X in Xs]

    df = pd.DataFrame(np.concatenate(Xs), columns=keys)
    labels = np.concatenate([[label] * len(X) for X, label in zip(Xs, labels)])
    df = df.assign(label=labels)

    #plt.figure(figsize=(3,3))
    # Changing figure size also proportionally changes the font size
    plt.rcParams["font.size"] = 14 # It works! But the ticks are too coarse

    #sns.set(font_scale=1.2)
    #matplotlib.rc_file_defaults() # sns.set changes the style
    #sns.set_style("ticks") # another style

    pair_grid = sns.pairplot(df, hue='label', corner=True,
                 kind='kde', # both diag and off-diag
                 #diag_kind = 'kde',
                 diag_kws = {'common_norm': False}, # kdeplot(univariate)
                 plot_kws = {'common_norm': False, 'levels': 5}, # kdeplot(bivariate)
    )
    #plt.savefig('outputs/pairplot.png')
    #plt.show()
    return pair_grid
