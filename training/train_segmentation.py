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

from models.segmentation_model import AdvancedSegModel
from datasets.dataset_seg import CataractSegDataset

NUM_SEG_CLASSES = 13  # Adjust as needed

def get_seg_transforms():
    return A.Compose([
        A.Resize(512,512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def dice_score(pred, target, num_classes):
    """
    Computes mean Dice (for segmentation) across classes (excluding background).
    pred, target are [B,H,W] integer class maps.
    """
    dice_per_class = []
    for c in range(1, num_classes):  # skip background=0
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_per_class.append(dice.item())
    return sum(dice_per_class)/len(dice_per_class) if dice_per_class else 0.0

def iou_score(pred, target, num_classes):
    """
    Computes mean IoU across classes (excluding background).
    """
    ious = []
    for c in range(1, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = (pred_c + target_c).clamp(0,1).sum()
        iou = (intersection + 1e-5) / (union + 1e-5)
        ious.append(iou.item())
    return sum(ious)/len(ious) if ious else 0.0

def validate_segmentation(model, loader, device, criterion, num_classes):
    model.eval()
    running_loss = 0.0
    running_iou  = 0.0
    running_dice = 0.0
    count = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device, memory_format=torch.channels_last)
            masks = masks.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                out_dict = model(imgs)
                logits = out_dict["out"]
                loss = criterion(logits, masks)

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            batch_iou  = iou_score(preds, masks, num_classes)
            batch_dice = dice_score(preds, masks, num_classes)
            running_iou  += batch_iou
            running_dice += batch_dice
            count += 1

    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    avg_iou  = running_iou / count if count > 0 else 0
    avg_dice = running_dice / count if count > 0 else 0
    return avg_loss, avg_iou, avg_dice

def train_segmentation(model, train_loader, val_loader, device, epochs=10, lr=1e-4, num_classes=13):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    train_losses, val_losses = [], []
    val_ious, val_dices = [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Seg Epoch {epoch}/{epochs}")
        for imgs, masks in pbar:
            imgs  = imgs.to(device, memory_format=torch.channels_last)
            masks = masks.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                out_dict = model(imgs)
                logits = out_dict["out"]
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        val_loss, val_iou, val_dice = validate_segmentation(model, val_loader, device, criterion, num_classes)
        scheduler.step(epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_iou={val_iou:.4f}, val_dice={val_dice:.4f}")

        # Overwrite checkpoint each epoch
        torch.save(model.state_dict(), "lightweight_seg_current.pth")

    # Plot training curves
    plt.figure(figsize=(12,5))

    plt.subplot(1,3,1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Segmentation Loss')
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(range(1, epochs+1), val_ious, label='Val IoU')
    plt.xlabel('Epoch'); plt.ylabel('IoU')
    plt.title('Validation Mean IoU')
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(range(1, epochs+1), val_dices, label='Val Dice')
    plt.xlabel('Epoch'); plt.ylabel('Dice')
    plt.title('Validation Mean Dice')
    plt.legend()

    plt.tight_layout()
    plt.savefig("seg_training_curves.png")
    print("Saved seg_training_curves.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of the segmentation dataset.")
    parser.add_argument("--seg_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    cudnn.benchmark = True

    transform = get_seg_transforms()
    dataset = CataractSegDataset(root_dir=args.root_dir, transform=transform)
    ds_len = len(dataset)
    print(f"Seg dataset length = {ds_len}")

    if ds_len == 0 or args.seg_epochs == 0:
        print("No data or 0 epochs => skipping segmentation.")
        return

    val_size = int(0.2 * ds_len)
    train_size = ds_len - val_size
    seg_train, seg_val = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(seg_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(seg_val,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = AdvancedSegModel(num_classes=NUM_SEG_CLASSES).to(device, memory_format=torch.channels_last)

    print("Beginning Segmentation Model Training...")
    train_segmentation(model, train_loader, val_loader, device,
                       epochs=args.seg_epochs, lr=args.lr,
                       num_classes=NUM_SEG_CLASSES)

    torch.save(model.state_dict(), "lightweight_seg_final.pth")
    print("Saved lightweight_seg_final.pth")

if __name__ == "__main__":
    main()