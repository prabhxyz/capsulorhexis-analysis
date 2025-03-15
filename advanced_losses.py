import torch
import torch.nn.functional as F

def dice_ce_loss(logits, targets, smooth=1.0):
    """
    Combined Dice + CrossEntropy. 
    logits: (batch, C, H, W)
    targets: (batch, H, W) integer class labels
    """
    # 1) CE part
    ce_loss = F.cross_entropy(logits, targets.long())

    # 2) Dice part (we treat each class's channel vs. one-hot target)
    probs = torch.softmax(logits, dim=1)
    num_classes = probs.shape[1]
    targets_1hot = F.one_hot(targets.long(), num_classes=num_classes)
    targets_1hot = targets_1hot.permute(0,3,1,2).float()  # (B, C, H, W)

    dice_sum = 0.0
    for c in range(num_classes):
        p = probs[:,c,:,:]
        t = targets_1hot[:,c,:,:]
        intersection = (p*t).sum()
        union = p.sum() + t.sum()
        dice_score = (2.0*intersection + smooth)/(union + smooth)
        # we do a "dice loss" = 1 - dice_score for that class
        dice_sum += (1.0 - dice_score)
    dice_loss_val = dice_sum / num_classes

    return ce_loss + dice_loss_val
