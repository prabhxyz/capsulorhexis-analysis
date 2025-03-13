import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from math import sqrt
from scipy.stats import ttest_rel

def compute_iou(pred, target, num_classes):
    """
    Compute IoU for each class and return as a list [IoU_class_0, IoU_class_1, ...].
    `pred` and `target` are torch Tensors or numpy arrays of shape (H,W) with
    integer class labels. 
    """
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
            ious.append(float('nan'))  # or 0.0, depending on how you want to treat empty
        else:
            ious.append(intersection / union)
    return ious

def compute_dice(pred, target, num_classes):
    """
    Compute Dice for each class.
    """
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        if denom == 0:
            dices.append(float('nan'))
        else:
            dices.append(2.0 * intersection / denom)
    return dices

def get_confusion_matrix(pred, target, num_classes):
    """
    Return a confusion matrix for all classes combined.
    Flatten the arrays and compute with sklearn.
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    cm = confusion_matrix(target_flat, pred_flat, labels=list(range(num_classes)))
    return cm

def confidence_interval(metric_values, confidence=0.95):
    """
    Simple confidence interval (CI) using t-distribution if we have
    multiple samples (e.g. metrics from multiple images).
    """
    arr = np.array(metric_values)
    arr = arr[~np.isnan(arr)]  # remove NaNs
    n = len(arr)
    mean_val = np.mean(arr)
    if n > 1:
        # t-distribution approach
        from scipy.stats import t
        sem = np.std(arr, ddof=1)/np.sqrt(n)
        t_val = t.ppf((1 + confidence) / 2.0, n - 1)
        margin = sem * t_val
        lower = mean_val - margin
        upper = mean_val + margin
    else:
        # not enough samples for meaningful CI
        lower = mean_val
        upper = mean_val
    return mean_val, lower, upper

def significance_test(values_a, values_b):
    """
    Example paired t-test between two sets of metric values 
    (e.g. dice for class=1 vs. class=2 if relevant).
    This can also be used to compare two models, etc.
    """
    a = np.array(values_a)
    b = np.array(values_b)
    # remove NaN
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return None, None
    t_stat, p_val = ttest_rel(a, b)
    return t_stat, p_val
