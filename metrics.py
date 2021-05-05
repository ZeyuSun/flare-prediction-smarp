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

