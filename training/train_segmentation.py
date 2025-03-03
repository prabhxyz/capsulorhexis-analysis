import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from datasets.dataset_seg import CataractSegDataset, NUM_SEG_CLASSES
from models.segmentation_model import AdvancedSegModel

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

def get_seg_transforms():
    """
    Returns advanced data augmentation for segmentation.
    """
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1,
                           rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def compute_iou_and_dice(pred, target, num_classes):
    """
    Computes mean IoU and mean Dice for a batch of predictions.
    pred, target: (N, H, W) integer class maps
    """
    ious = []
    dices = []

    for c in range(1, num_classes):  # ignoring background=0
        pred_c = (pred == c)
        tgt_c  = (target == c)
        intersection = (pred_c & tgt_c).sum().item()
        union = (pred_c | tgt_c).sum().item()
        iou = intersection / union if union > 0 else 1.0
        ious.append(iou)

        dice_den = pred_c.sum().item() + tgt_c.sum().item()
        dice = 2.0 * intersection / dice_den if dice_den > 0 else 1.0
        dices.append(dice)

    mean_iou  = np.mean(ious) if ious else 1.0
    mean_dice = np.mean(dices) if dices else 1.0
    return mean_iou, mean_dice

def evaluate_segmentation(model, loader, device, num_classes):
    """
    Computes average loss, IoU, Dice over the dataset.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    iou_list = []
    dice_list = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outs = model(imgs)["out"]
            loss = criterion(outs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outs, dim=1)
            iou_val, dice_val = compute_iou_and_dice(preds.cpu(), masks.cpu(), num_classes)
            iou_list.append(iou_val)
            dice_list.append(dice_val)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    avg_iou  = np.mean(iou_list) if iou_list else 1.0
    avg_dice = np.mean(dice_list) if dice_list else 1.0
    return avg_loss, avg_iou, avg_dice

def main_seg_train(root_dir, epochs=10, batch_size=4, lr=1e-4):
    """
    Trains the segmentation model, logs cross-entropy + IoU/Dice, saves each epoch's model
    to 'lightweight_seg.pth'. Overwrites for simplicity.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SegTraining] Using device={device}")

    # Dataset
    transform = get_seg_transforms()
    dataset = CataractSegDataset(root_dir=root_dir, transform=transform)
    seg_len = len(dataset)
    print(f"[SegTraining] Found {seg_len} samples in '{root_dir}'")
    if seg_len == 0:
        print("[SegTraining] No data => skipping.")
        return

    val_size = int(0.2 * seg_len)
    train_size = seg_len - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    seg_model = AdvancedSegModel(num_classes=NUM_SEG_CLASSES, use_pretrained=True, aux_loss=True).to(device)

    optimizer = optim.AdamW(seg_model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type=="cuda"))

    train_loss_list = []
    val_loss_list   = []
    train_iou_list  = []
    val_iou_list    = []
    train_dice_list = []
    val_dice_list   = []

    for epoch in range(1, epochs + 1):
        seg_model.train()
        running_loss = 0.0
        running_iou  = []
        running_dice = []

        pbar = tqdm(train_loader, desc=f"Seg Epoch {epoch}/{epochs}")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=(device.type=="cuda")):
                outs = seg_model(imgs)["out"]
                loss = criterion(outs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            # Compute IoU/Dice for the batch:
            preds = torch.argmax(outs, dim=1)
            iou_val, dice_val = compute_iou_and_dice(preds.cpu(), masks.cpu(), NUM_SEG_CLASSES)
            running_iou.append(iou_val)
            running_dice.append(dice_val)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_loader)
        train_iou  = np.mean(running_iou) if running_iou else 1.0
        train_dice = np.mean(running_dice) if running_dice else 1.0
        train_loss_list.append(train_loss)
        train_iou_list.append(train_iou)
        train_dice_list.append(train_dice)

        # Evaluate on val:
        val_loss, val_iou, val_dice = evaluate_segmentation(seg_model, val_loader, device, NUM_SEG_CLASSES)
        val_loss_list.append(val_loss)
        val_iou_list.append(val_iou)
        val_dice_list.append(val_dice)

        print(f"[SegTraining] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_iou={train_iou:.4f}, val_iou={val_iou:.4f}, "
              f"train_dice={train_dice:.4f}, val_dice={val_dice:.4f}")

        # Overwrite model each epoch
        torch.save(seg_model.state_dict(), "lightweight_seg.pth")

    # Plot
    ep_range = range(1, epochs + 1)
    plt.figure(figsize=(12,6))

    plt.subplot(1,3,1)
    plt.plot(ep_range, train_loss_list, label='Train Loss')
    plt.plot(ep_range, val_loss_list,   label='Val Loss')
    plt.title("Seg Loss")
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(ep_range, train_iou_list, label='Train IoU')
    plt.plot(ep_range, val_iou_list,   label='Val IoU')
    plt.title("Seg IoU")
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(ep_range, train_dice_list, label='Train Dice')
    plt.plot(ep_range, val_dice_list,   label='Val Dice')
    plt.title("Seg Dice")
    plt.legend()

    plt.tight_layout()
    plt.savefig("seg_training_curve.png")
    plt.close()

    print("[SegTraining] Done. Model => 'lightweight_seg.pth', plot => 'seg_training_curve.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="Cataract-1k-Seg")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    main_seg_train(
        root_dir=args.root_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )