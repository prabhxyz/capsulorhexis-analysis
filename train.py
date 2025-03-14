import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import argparse

from dataset import CataractSuperviselyDataset, get_train_transforms, get_val_transforms
from model import get_segmentation_model
from eval_metrics import compute_iou, compute_dice, get_confusion_matrix, confidence_interval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="unet",
                        help="'unet' or 'deeplab'.")
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b0",
                        help="Encoder/backbone name, e.g. 'resnet101'.")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classes: background(0) + relevant(2) => total=3.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # We assume the top-level folder is "Cataract-1k-Seg/"
    data_dir = "Cataract-1k-Seg"
    sup_anno_dir = os.path.join(data_dir, "Annotations", "Images-and-Supervisely-Annotations")

    # 1) Find all "case_XXXX" folders.
    all_cases = [
        d for d in os.listdir(sup_anno_dir)
        if d.startswith("case_") and os.path.isdir(os.path.join(sup_anno_dir, d))
    ]
    all_cases.sort()

    # 2) 80/20 split by case.
    random.seed(42)
    random.shuffle(all_cases)
    split_idx = int(0.8 * len(all_cases))
    train_cases = all_cases[:split_idx]
    val_cases = all_cases[split_idx:]
    print(f"Found {len(all_cases)} cases => {len(train_cases)} train, {len(val_cases)} val.")

    # 3) Build Datasets
    train_dataset = CataractSuperviselyDataset(
        root_dir=data_dir,
        case_ids=train_cases,
        transforms=get_train_transforms(),
        relevant_class_names=["Capsulorhexis Forceps", "Capsulorhexis Cystotome"]
    )
    val_dataset = CataractSuperviselyDataset(
        root_dir=data_dir,
        case_ids=val_cases,
        transforms=get_val_transforms(),
        relevant_class_names=["Capsulorhexis Forceps", "Capsulorhexis Cystotome"]
    )

    print(f"Train dataset size = {len(train_dataset)}")
    print(f"Val dataset size   = {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 4) Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_segmentation_model(args.model_type, args.num_classes, args.encoder_name)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    # Make a checkpoints folder
    checkpoints_folder = "checkpoints"
    os.makedirs(checkpoints_folder, exist_ok=True)

    best_val_iou = 0.0
    train_loss_history = []
    val_iou_history = []
    val_dice_history = []

    for epoch in range(args.epochs):
        # ---------------------------
        # TRAIN PHASE
        # ---------------------------
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)

        # ---------------------------
        # VALIDATION PHASE
        # ---------------------------
        model.eval()
        iou_scores = []
        dice_scores = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)  # (batch, H, W)

                for b in range(images.size(0)):
                    # Move to CPU-based NumPy
                    pred_b = preds[b].detach().cpu().numpy()
                    mask_b = masks[b].detach().cpu().numpy()

                    iou = compute_iou(pred_b, mask_b, args.num_classes)
                    dice = compute_dice(pred_b, mask_b, args.num_classes)

                    valid_classes = list(range(1, args.num_classes))  # skip background=0
                    iou_vals = [iou[c] for c in valid_classes]
                    dice_vals = [dice[c] for c in valid_classes]

                    if not iou_vals or all(np.isnan(iou_vals)):
                        iou_avg = 0.0
                    else:
                        iou_avg = np.nanmean(iou_vals)

                    if not dice_vals or all(np.isnan(dice_vals)):
                        dice_avg = 0.0
                    else:
                        dice_avg = np.nanmean(dice_vals)

                    iou_scores.append(iou_avg)
                    dice_scores.append(dice_avg)

        mean_iou = np.nanmean(iou_scores) if len(iou_scores) > 0 else 0.0
        mean_dice = np.nanmean(dice_scores) if len(dice_scores) > 0 else 0.0
        val_iou_history.append(mean_iou)
        val_dice_history.append(mean_dice)

        print(f"[Epoch {epoch+1}] train_loss={epoch_loss:.4f}, val_mIoU={mean_iou:.4f}, val_mDice={mean_dice:.4f}")

        # Save checkpoint for this epoch into "checkpoints/epoch_{n}.pth"
        checkpoint_path = os.path.join(checkpoints_folder, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # If best IoU so far, save model in the **root** as best_segmentation_model.pth
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            torch.save(model.state_dict(), "best_segmentation_model.pth")
            print("   [BEST] Saved best model so far.")

    # 5) Final Stats
    m, ci_low, ci_high = confidence_interval(val_iou_history)
    print(f"Val IoU across epochs => mean={m:.4f} [95%CI: ({ci_low:.4f}, {ci_high:.4f})]")

    # Optionally do a confusion matrix with best model
    model.load_state_dict(torch.load("best_segmentation_model.pth", map_location=device))
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    from eval_metrics import get_confusion_matrix
    cm = get_confusion_matrix(all_preds, all_targets, args.num_classes)
    print("Confusion Matrix:\n", cm)

    # ---------------------------
    # PLOTS
    # ---------------------------

    # (1) Train Loss
    plt.figure()
    plt.plot(range(1, args.epochs+1), train_loss_history, marker='o')
    plt.title("Train Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("train_loss_curve.png", dpi=150)
    plt.close()

    # (2) Validation IoU
    plt.figure()
    plt.plot(range(1, args.epochs+1), val_iou_history, marker='o', color='orange')
    plt.title("Validation Mean IoU per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.grid(True)
    plt.savefig("val_iou_curve.png", dpi=150)
    plt.close()

    # (3) Validation Dice
    plt.figure()
    plt.plot(range(1, args.epochs+1), val_dice_history, marker='o', color='green')
    plt.title("Validation Mean Dice per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice")
    plt.grid(True)
    plt.savefig("val_dice_curve.png", dpi=150)
    plt.close()

    print("Training complete.")
    print("All epoch checkpoints saved under 'checkpoints/'.")
    print("Best model => 'best_segmentation_model.pth' in project root.")

if __name__ == "__main__":
    main()