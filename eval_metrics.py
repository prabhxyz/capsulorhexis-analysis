import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import ttest_rel

def compute_iou(pred, target, num_classes):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

def compute_dice(pred, target, num_classes):
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        if denom == 0:
            dices.append(float('nan'))
        else:
            dices.append(2.0*intersection/denom)
    return dices

def get_confusion_matrix(pred, target, num_classes):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    return confusion_matrix(target_flat, pred_flat, labels=list(range(num_classes)))

def confidence_interval(metric_values, confidence=0.95):
    from scipy.stats import t
    arr = np.array(metric_values)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    mean_val = np.mean(arr)
    if n > 1:
        sem = np.std(arr, ddof=1)/np.sqrt(n)
        t_val = t.ppf((1+confidence)/2.0, n-1)
        margin = sem*t_val
        return mean_val, mean_val - margin, mean_val + margin
    else:
        return mean_val, mean_val, mean_val

def significance_test(values_a, values_b):
    a = np.array(values_a)
    b = np.array(values_b)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return None, None
    t_stat, p_val = ttest_rel(a, b)
    return t_stat, p_val
