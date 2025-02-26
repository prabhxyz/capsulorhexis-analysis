import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate
from segmentation.dataset_seg import InvalidSegSampleError

def collate_fn(batch):
    """Handles normal case where items are (image, mask)."""
    return tuple(zip(*batch))

def collate_seg_skip_invalid(batch):
    """Skip any samples that raised InvalidSegSampleError."""
    clean_data = []
    for item in batch:
        if isinstance(item, Exception):
            continue
        else:
            clean_data.append(item)
    if len(clean_data) == 0:
        # all invalid in this batch
        return None, None
    # Now we can do the normal "pair zip"
    return tuple(zip(*clean_data))
    
def create_target_from_mask(masks, device):
    """
    Convert a (B,H,W) mask of class IDs into a detection format for Mask R-CNN:
    Each sample -> Dict with "boxes", "labels", "masks"
    We'll treat each distinct class (besides background=0) as a separate instance.
    """
    targets = []
    for mask in masks:
        # mask shape: (H,W)
        unique_classes = torch.unique(mask)
        boxes, labels, ms = [], [], []
        for c in unique_classes:
            if c.item() == 0:
                continue
            # c is class ID
            class_mask = (mask == c)
            if class_mask.sum() < 10:
                # skip tiny areas
                continue
            pos = class_mask.nonzero()
            ymin = pos[:,0].min()
            ymax = pos[:,0].max()
            xmin = pos[:,1].min()
            xmax = pos[:,1].max()
            boxes.append(torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32))
            labels.append(c.item())
            ms.append(class_mask.unsqueeze(0))  # (1,H,W)
        if len(labels) == 0:
            # no instruments found
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            ms = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.stack(boxes, dim=0)
            labels = torch.tensor(labels, dtype=torch.int64)
            ms = torch.cat(ms, dim=0).type(torch.uint8)

        sample_target = {
            "boxes": boxes.to(device),
            "labels": labels.to(device),
            "masks": ms.to(device)
        }
        targets.append(sample_target)
    return targets

def save_training_plot_seg(train_losses, val_losses, out_path="seg_training_loss.png"):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Segmentation Training Loss')
    plt.savefig(out_path)
    plt.close()