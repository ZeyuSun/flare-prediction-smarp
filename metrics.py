import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, make_scorer

def get_scores_from_cm(cm):
    """a numpy version instead of torch
    """
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

def roc_auc_score(y_true, y_score):
    """
    y_score doesn't have to be probabilities.
    """
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_score)

def optimal_tss(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    return max(tpr - fpr)

def draw_ssp(y_true, y_score):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:] # remove added point
    P = y_true.sum() #np.sum(y_true) don't use np.sum. Won't take sum() received an invalid combination of arguments - got (out=NoneType, axis=NoneType, ), but expected one of:
    N = len(y_true) - P
    FP, TP = N * fpr, P * tpr
    TN, FN = N - FP, P - TP
    TSS = tpr - fpr
    HSS2 = 2 * (TP * TN - FN * FP) / (P * (FN+TN) + (TP+FP) * N)
    idx = np.argmax(TSS)
    xm, ym = thresholds[idx], TSS[idx]

    fig, ax = plt.subplots(figsize=(3.75,3))
    ax.plot(thresholds, HSS2, label='HSS2')
    ax.plot(thresholds, TSS, label='TSS')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([xm], [ym], 'o', color='C7')
    ax.hlines(ym, xlim[0], xm, linestyle="dashed", color='C7')
    ax.text(xlim[0], ym+0.02, f'{ym:.3f}', color='C7')
    ax.set_xlim(xlim) # hline extends xlim
    ax.set_ylim([ylim[0], ylim[1]+0.1]) # text in the plot
    ax.set_xlabel('Threshold')
    ax.legend(loc='lower center')
    return fig

def gini(y_true, y_prob):
    """Gini Coefficient
    """
    return 2 * roc_auc_score(y_true, y_prob) - 1

def tss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred) # rows are true classes, cols are predicted classes
    [[TN, FP], [FN, TP]] = cm
    pod = TP / (FN + TP)
    far = FP / (TN + FP)
    tss = pod - far
    return tss

def hss2(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred) # rows are true classes, cols are predicted classes
    [[TN, FP], [FN, TP]] = cm
    N = TN + FP
    P = TP + FN
    hss2 = 2 * (TP * TN - FN * FP) / (P * (FN+TN) + (TP+FP) * N)
    return hss2

