import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate

def save_training_plot(train_losses, val_losses, out_path="phase_training_loss.png"):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Phase Recognition Training Loss')
    plt.savefig(out_path)
    plt.close()

def accuracy_score(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().sum()
    return correct / labels.shape[0]

class InvalidSubclipError(Exception):
    pass

def collate_phase_skip_invalid(batch):
    """Skip samples that raised InvalidSubclipError."""
    clean_data = []
    for item in batch:
        if isinstance(item, Exception):
            # skip
            continue
        else:
            clean_data.append(item)
    if len(clean_data) == 0:
        # all subclips were invalid
        return None, None
    return default_collate(clean_data)