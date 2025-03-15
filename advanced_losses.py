import torch
import torch.nn.functional as F

def weighted_focal_dice_loss(logits, targets, alpha=0.5, gamma=2.0, smooth=1.0):
    """
    Weighted Focal + Dice Loss for multi-class segmentation.
    logits: (B, C, H, W)
    targets: (B, H, W) integer class labels
    alpha: weighting factor (for foreground vs. background).
    gamma: focusing param for focal
    smooth: for dice
    """
    probs = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    with torch.no_grad():
        t_onehot = F.one_hot(targets.long(), num_classes=num_classes).permute(0,3,1,2).float()

    # Focal
    focal_term = - ((1 - probs)**gamma) * torch.log(probs + 1e-12)
    # apply alpha equally to all classes or adapt if you want class-based alpha
    focal_loss = alpha * (t_onehot * focal_term).sum(dim=1).mean()  # average over pixels

    # Dice
    dice_sum = 0.0
    for c in range(num_classes):
        p = probs[:,c,:,:]
        t = t_onehot[:,c,:,:]
        intersection = (p*t).sum()
        union = p.sum() + t.sum()
        dice_score = (2.0*intersection + smooth)/(union + smooth)
        dice_sum += (1.0 - dice_score)
    dice_loss = dice_sum/num_classes

    return focal_loss + dice_loss
