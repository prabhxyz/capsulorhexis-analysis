import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Suppose we have a cataract_seg_dataset.py with 'CataractSegDataset' + 'NUM_SEG_CLASSES'
from cataract_seg_dataset import CataractSegDataset, NUM_SEG_CLASSES

# Suppose your segmentation model is in models/segmentation_model.py
from models.segmentation_model import LightweightSegModel

def get_seg_transforms():
    return A.Compose([
        A.Resize(512,512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def compute_iou_dice(pred_mask, true_mask, ignore_background=True):
    """
    Computes mean IoU and mean Dice for multi-class segmentation.
    pred_mask, true_mask: shape (B,H,W) [already argmaxed for pred]
    If ignore_background=True, skip class_id=0 in averaging.

    Returns: (mean_iou, mean_dice) as floats
    """
    device = pred_mask.device
    num_classes = torch.max(true_mask).item()  # or known from dataset (NUM_SEG_CLASSES-1)
    # But safer: use a for-loop up to 'NUM_SEG_CLASSES' if you want to fix it.

    # We'll assume 'NUM_SEG_CLASSES' is known, or dynamically find it:
    # E.g. from the dataset, or a fixed constant
    # But if pred_mask has none of the highest classes, we can do:
    num_classes = NUM_SEG_CLASSES  # from import
    class_ids = range(num_classes)
    if ignore_background:
        class_ids = range(1, num_classes)  # skip 0

    iou_vals = []
    dice_vals = []

    for c in class_ids:
        # pred_c, true_c are bool masks
        pred_c = (pred_mask == c)
        true_c = (true_mask == c)
        intersection = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        pred_area = pred_c.sum().float()
        true_area = true_c.sum().float()

        if union == 0:
            # If there's no ground-truth or prediction for this class in the batch,
            # skip or treat as iou=1, dice=1. We'll skip here.
            continue

        iou = intersection / union
        # dice denominator: pred_area + true_area
        if (pred_area + true_area) == 0:
            dice = torch.tensor(1.0, device=device)
        else:
            dice = 2*intersection / (pred_area + true_area)

        iou_vals.append(iou.item())
        dice_vals.append(dice.item())

    if len(iou_vals) == 0:
        mean_iou = 1.0
        mean_dice = 1.0
    else:
        mean_iou = sum(iou_vals)/len(iou_vals)
        mean_dice = sum(dice_vals)/len(dice_vals)

    return mean_iou, mean_dice

def validate_segmentation(model, loader, device, criterion):
    model.eval()
    losses = []
    ious = []
    dices = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            output = model(imgs)["out"]  # (B, num_classes, H, W)
            loss = criterion(output, masks)
            losses.append(loss.item())

            pred = torch.argmax(output, dim=1)  # shape (B,H,W)
            iou, dice = compute_iou_dice(pred, masks, ignore_background=True)
            ious.append(iou)
            dices.append(dice)

    val_loss = sum(losses)/len(losses) if len(losses)>0 else 0
    val_iou = sum(ious)/len(ious) if len(ious)>0 else 1.0
    val_dice = sum(dices)/len(dices) if len(dices)>0 else 1.0

    return val_loss, val_iou, val_dice

def train_segmentation(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_ious,   val_ious   = [], []
    train_dices,  val_dices  = [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_iou  = 0.0
        running_dice = 0.0
        count_batches = 0

        pbar = tqdm(train_loader, desc=f"Seg Epoch {epoch}/{epochs}")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            output = model(imgs)["out"]
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # compute IoU/Dice for train batch
            pred = torch.argmax(output, dim=1)
            iou, dice = compute_iou_dice(pred, masks, ignore_background=True)
            running_iou  += iou
            running_dice += dice

            count_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if count_batches > 0:
            train_loss = running_loss / count_batches
            train_iou  = running_iou  / count_batches
            train_dice = running_dice / count_batches
        else:
            train_loss, train_iou, train_dice = 0,1,1

        val_loss, val_iou, val_dice = validate_segmentation(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        train_ious.append(train_iou)
        train_dices.append(train_dice)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)

        print(f"Epoch {epoch}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_iou={train_iou:.4f}, val_iou={val_iou:.4f}, "
              f"train_dice={train_dice:.4f}, val_dice={val_dice:.4f}")

    # Plot final curves
    plt.figure(figsize=(10,6))

    # 1) Loss
    plt.subplot(2,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 2) IoU
    plt.subplot(2,2,2)
    plt.plot(train_ious, label='Train IoU')
    plt.plot(val_ious, label='Val IoU')
    plt.title('Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    # 3) Dice
    plt.subplot(2,2,3)
    plt.plot(train_dices, label='Train Dice')
    plt.plot(val_dices, label='Val Dice')
    plt.title('Mean Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()

    plt.tight_layout()
    plt.savefig("seg_training_metrics.png")
    plt.close()

    return train_losses, val_losses, train_ious, val_ious, train_dices, val_dices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_data_root", type=str, default="Cataract-1k-Seg")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    cudnn.benchmark = True

    seg_transform = get_seg_transforms()
    seg_dataset = CataractSegDataset(root_dir=args.seg_data_root, transform=seg_transform)

    total_len = len(seg_dataset)
    print(f"Seg dataset length = {total_len}")
    if total_len == 0:
        print("No segmentation data found. Exiting.")
        return

    val_size = int(0.2 * total_len)
    train_size = total_len - val_size
    seg_train, seg_val = random_split(seg_dataset, [train_size, val_size])

    train_loader = DataLoader(seg_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(seg_val,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = LightweightSegModel(num_classes=NUM_SEG_CLASSES, use_pretrained=True, aux_loss=True).to(device)

    print("Starting segmentation training with IoU & Dice metrics...")
    train_segmentation(model, train_loader, val_loader, device,
                       epochs=args.epochs, lr=args.lr)

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "lightweight_seg.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    print("Training curves saved to 'seg_training_metrics.png'.")

if __name__ == "__main__":
    main()
