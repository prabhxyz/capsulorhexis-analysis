import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CataractSuperviselyAllDataset, get_train_transforms, get_val_transforms
from model import get_segmentation_model
from eval_metrics import compute_iou, compute_dice, confidence_interval
from advanced_losses import weighted_focal_dice_loss

import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="k-fold cross validation. Default=5.")
    parser.add_argument("--model_type", type=str, default="deeplab", help="'unet' or 'deeplab'.")
    parser.add_argument("--encoder_name", type=str, default="resnet101", help="Backbone.")
    parser.add_argument("--num_classes", type=int, default=10, help="Max classes across dataset, 0=bg => total=10.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for Weighted Focal Loss portion.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma for focal focusing.")
    args = parser.parse_args()

    data_dir = "Cataract-1k-Seg"
    sup_dir = os.path.join(data_dir, "Annotations", "Images-and-Supervisely-Annotations")
    all_cases = [
        d for d in os.listdir(sup_dir)
        if d.startswith("case_") and os.path.isdir(os.path.join(sup_dir, d))
    ]
    all_cases.sort()
    n_cases = len(all_cases)
    print(f"Found {n_cases} total cases => using k={args.k} fold cross validation.")

    random.seed(42)
    random.shuffle(all_cases)

    fold_size = int(np.ceil(n_cases / args.k))
    folds = []
    for i in range(args.k):
        start = i*fold_size
        end = start + fold_size
        folds.append(all_cases[start:end])

    fold_results = []

    for fold_idx in range(args.k):
        print(f"\n=== FOLD {fold_idx+1}/{args.k} ===")

        val_cases = folds[fold_idx]
        train_cases = []
        for i in range(args.k):
            if i != fold_idx:
                train_cases.extend(folds[i])

        train_dataset = CataractSuperviselyAllDataset(
            root_dir=data_dir,
            case_ids=train_cases,
            transforms=get_train_transforms()
        )
        val_dataset = CataractSuperviselyAllDataset(
            root_dir=data_dir,
            case_ids=val_cases,
            transforms=get_val_transforms()
        )

        print(f"Fold {fold_idx+1}: train={len(train_dataset)} images, val={len(val_dataset)} images.")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_segmentation_model(args.model_type, args.num_classes, args.encoder_name)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

        fold_dir = f"fold_{fold_idx+1}"
        os.makedirs(fold_dir, exist_ok=True)

        train_loss_history = []
        val_iou_history = []
        val_dice_history = []

        best_val_iou = 0.0

        for epoch in range(args.epochs):
            # ---------------- TRAIN ----------------
            model.train()
            running_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"[Fold {fold_idx+1} E{epoch+1}/{args.epochs} train]"):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = weighted_focal_dice_loss(
                    logits, masks,
                    alpha=args.alpha,
                    gamma=args.gamma
                )
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            train_loss_history.append(epoch_loss)

            scheduler.step()

            # ---------------- VAL ----------------
            model.eval()
            iou_scores = []
            dice_scores = []
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"[Fold {fold_idx+1} E{epoch+1}/{args.epochs} val]"):
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    preds = torch.argmax(logits, dim=1)
                    for b in range(images.size(0)):
                        pred_b = preds[b].cpu().numpy()
                        mask_b = masks[b].cpu().numpy()

                        iou_list = compute_iou(pred_b, mask_b, args.num_classes)
                        dice_list = compute_dice(pred_b, mask_b, args.num_classes)
                        # skip background=0 => measure classes [1..num_classes-1]
                        valid = list(range(1, args.num_classes))
                        iou_vals = [iou_list[c] for c in valid]
                        dice_vals = [dice_list[c] for c in valid]
                        iou_avg = np.nanmean(iou_vals) if (iou_vals and not all(np.isnan(iou_vals))) else 0.0
                        dice_avg = np.nanmean(dice_vals) if (dice_vals and not all(np.isnan(dice_vals))) else 0.0
                        iou_scores.append(iou_avg)
                        dice_scores.append(dice_avg)

            mean_iou = np.nanmean(iou_scores) if len(iou_scores)>0 else 0.0
            mean_dice = np.nanmean(dice_scores) if len(dice_scores)>0 else 0.0
            val_iou_history.append(mean_iou)
            val_dice_history.append(mean_dice)

            print(f"[Fold {fold_idx+1}, Epoch {epoch+1}] train_loss={epoch_loss:.4f}, val_mIoU={mean_iou:.4f}, val_mDice={mean_dice:.4f}")

            # Save best
            if mean_iou > best_val_iou:
                best_val_iou = mean_iou
                torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
                print(f"   [BEST] Saved best model so far for fold {fold_idx+1}")

            # ----------- SAVE GRAPHS EVERY EPOCH -----------
            # 1) train_loss
            plt.figure()
            plt.plot(range(1,len(train_loss_history)+1), train_loss_history, marker='o')
            plt.title(f"Fold {fold_idx+1} - Train Loss (Epoch {epoch+1})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(os.path.join(fold_dir, f"train_loss_epoch_{epoch+1}.png"), dpi=150)
            plt.close()

            # 2) val_iou
            plt.figure()
            plt.plot(range(1,len(val_iou_history)+1), val_iou_history, marker='o', color='orange')
            plt.title(f"Fold {fold_idx+1} - Val IoU (Epoch {epoch+1})")
            plt.xlabel("Epoch")
            plt.ylabel("Mean IoU")
            plt.grid(True)
            plt.savefig(os.path.join(fold_dir, f"val_iou_epoch_{epoch+1}.png"), dpi=150)
            plt.close()

            # 3) val_dice
            plt.figure()
            plt.plot(range(1,len(val_dice_history)+1), val_dice_history, marker='o', color='green')
            plt.title(f"Fold {fold_idx+1} - Val Dice (Epoch {epoch+1})")
            plt.xlabel("Epoch")
            plt.ylabel("Mean Dice")
            plt.grid(True)
            plt.savefig(os.path.join(fold_dir, f"val_dice_epoch_{epoch+1}.png"), dpi=150)
            plt.close()

        print(f"FOLD {fold_idx+1} best val IoU = {best_val_iou:.4f}")
        fold_results.append(best_val_iou)

    # Summarize
    print("\n=== Cross Validation Summary ===")
    print("Fold IoUs:", fold_results)
    avg_iou = np.mean(fold_results)
    std_iou = np.std(fold_results, ddof=1) if len(fold_results)>1 else 0
    print(f"k={args.k} folds => IoU= {avg_iou:.4f} ± {std_iou:.4f}")

if __name__ == "__main__":
    main()
