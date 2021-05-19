from typing import Optional
import torch
import logging


def confusion_matrix(
        pred: torch.Tensor,
        target: torch.Tensor,
        normalize: bool = False,
        num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    A modification to pytorch-lightning's confusion_matrix.
    Computes the confusion matrix C where each entry C_{i,j} is the number of
    observations in group i that were predicted in group j.

    Args:
        pred: estimated targets
        target: ground truth labels
        normalize: normalizes confusion matrix
        num_classes: number of classes

    Return:
        Tensor, confusion matrix C [num_classes, num_classes ]

    Example:

        >>> x = torch.tensor([1, 2, 3])
        >>> y = torch.tensor([0, 2, 3])
        >>> confusion_matrix(x, y)
        tensor([[0., 1., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]])
    """
    from pytorch_lightning.metrics.utils import get_num_classes
    num_classes = get_num_classes(pred, target, num_classes)

    unique_labels = target.view(-1) * num_classes + pred.view(-1)

    bins = torch.bincount(unique_labels, minlength=num_classes ** 2)
    cm = bins.reshape(num_classes, num_classes).squeeze().float()

    if normalize:
        cm = cm / cm.sum(-1)

    return cm


def get_thresh(y_true, y_prob, criterion=None):
    if criterion is None:
        return 0.5

    from pytorch_lightning.metrics.functional import roc
    if y_true.sum() == 0:
        logging.warning('Return thresh 0.5, because no positive samples in targets, true positive value should be meaningless')
        return 0.5 # ValueError: No positive samples in targets, true positive value should be meaningless
    fpr, tpr, thresholds = roc(y_prob, y_true, num_classes=2)
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]  # remove added point
    if criterion == 'tss':
        TSS = tpr - fpr
        return thresholds[TSS.argmax()]

    P = y_true.sum()
    N = len(y_true) - P
    FP, TP = N * fpr, P * tpr
    TN, FN = N - FP, P - TP
    if criterion == 'hss2':
        HSS2 = 2 * (TP * TN - FN * FP) / (P * (FN + TN) + (TP + FP) * N)
        return thresholds[HSS2.argmax()]


def get_scores_from_cm(cm):
    [[TN, FP], [FN, TP]] = cm
    N = TN + FP
    P = TP + FN
    precisions = torch.diagonal(cm) / torch.sum(cm, 0)
    recalls = torch.diagonal(cm) / torch.sum(cm, 1)
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


def get_metrics_probabilistic(y_true, y_prob, criterion='hss2'):
    """
    =====================================================
        \ Pred  0               1               |
    True \                                      |
    --------------------------------------------+--------
    0           TN (rejection)  FP (false alarm)|   N
    1           FN (miss)       TP (detection)  |   P
    =====================================================
    """
    import torch
    from sklearn.metrics import roc_auc_score

    thresh = get_thresh(y_true, y_prob, criterion=criterion)
    y_pred = y_prob > thresh
    cm = confusion_matrix(y_pred, y_true, num_classes=2)
    scores = get_scores_from_cm(cm)

    y_clim = torch.mean(y_true.double())
    bss = 1 - torch.mean((y_prob - y_true)**2) / torch.mean((y_prob - y_clim)**2)
    auc = roc_auc_score(y_true.detach().cpu().numpy(),
                        y_prob.detach().cpu().numpy())
    scores.update({
        'auc': auc,
        'bss': bss,
    })

    return scores, cm


def get_metrics_multiclass(i_true, i_pred):
    import torch
    y_true = i_true.floor().to(torch.int32) + 9
    y_pred = i_pred.floor().to(torch.int32).clamp(-9,-4) + 9  # clamp -3 to -4
    cm6 = confusion_matrix(y_pred, y_true, num_classes=6)
    cm2 = confusion_matrix(y_pred>=4, y_true>=4, num_classes=2)
    cmq = confusion_matrix(y_pred>=1, y_true>=1, num_classes=2)
    scores = get_scores_from_cm(cm2)

    return scores, cm6, cm2, cmq


if __name__ == '__main__':
    # Test draw_reliability_plot
    y_true = torch.tensor([0, 0, 1, 0, 1])
    y_prob = torch.tensor([0., 0.1, 0.2, 0.8, 0.9])
    scores, cm = get_metrics_probabilistic(y_true, y_prob)
    print(scores)
    print(cm)