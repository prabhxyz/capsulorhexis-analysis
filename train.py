import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from dataset import CataractCocoDataset, get_train_transforms, get_val_transforms
from model import get_segmentation_model
from eval_metrics import compute_iou, compute_dice, get_confusion_matrix, confidence_interval

def train(
    train_image_dir,
    train_annotation_json,
    val_image_dir,
    val_annotation_json,
    num_classes=4,  # e.g. 0=background, 1=forceps, 2=cystotome, 3=capsulorhexis boundary
    epochs=10,
    batch_size=4,
    lr=1e-4,
):
    """
    Full training script for a U-Net style model with an EfficientNet encoder.
    Demonstrates advanced transforms, logging each epoch's training loss, 
    validating with Dice/IoU, and producing stats + plots.
    """

    # Prepare datasets
    train_dataset = CataractCocoDataset(
        image_dir=train_image_dir,
        annotation_json=train_annotation_json,
        transforms=get_train_transforms(),
        relevant_class_names=["capsulorhexis forceps", "capsulorhexis cystotome", "capsulorhexis boundary"]
    )
    val_dataset = CataractCocoDataset(
        image_dir=val_image_dir,
        annotation_json=val_annotation_json,
        transforms=get_val_transforms(),
        relevant_class_names=["capsulorhexis forceps", "capsulorhexis cystotome", "capsulorhexis boundary"]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, optimizer, loss
    model = get_segmentation_model(num_classes=num_classes, encoder_name="efficientnet-b0")
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    best_val_iou = 0.0

    train_loss_history = []
    val_iou_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - train"):
            images = images.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()
            logits = model(images)  # shape: (batch, num_classes, H, W)
            loss = criterion(logits, masks.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)

        # Validation loop
        model.eval()
        iou_scores = []
        dice_scores = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - val"):
                images = images.cuda()
                masks = masks.cuda()

                logits = model(images)
                # Predicted class is argmax over channels:
                preds = torch.argmax(logits, dim=1)

                # Evaluate IOU & Dice for each image, across all classes
                for b in range(images.size(0)):
                    iou = compute_iou(preds[b], masks[b], num_classes)
                    dice = compute_dice(preds[b], masks[b], num_classes)
                    # For a quick summary, we might average over classes (excluding background)
                    # but store them all for advanced stats
                    valid_classes = list(range(1, num_classes))  # skip background=0
                    iou_avg = np.nanmean([iou[c] for c in valid_classes])
                    dice_avg = np.nanmean([dice[c] for c in valid_classes])

                    iou_scores.append(iou_avg)
                    dice_scores.append(dice_avg)

        mean_iou = np.nanmean(iou_scores)
        mean_dice = np.nanmean(dice_scores)
        val_iou_history.append(mean_iou)

        print(f"[Epoch {epoch+1}] train_loss: {epoch_loss:.4f}, val_mIoU: {mean_iou:.4f}, val_mDice: {mean_dice:.4f}")

        # Simple checkpoint
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            torch.save(model.state_dict(), "best_segmentation_model.pth")
            print("Saved best model")

    # After training, do an in-depth statistical analysis on the collected val iou_scores if desired
    from eval_metrics import confidence_interval
    mean_iou, ci_lower, ci_upper = confidence_interval(val_iou_history)
    print(f"Validation IoU over epochs: mean={mean_iou:.4f} 95%CI=[{ci_lower:.4f}, {ci_upper:.4f}]")

    # Plot confusion matrix on the entire val set (example demonstration for one epoch)
    # We'll do a single pass again to gather predictions. 
    # For more robust analysis, gather from each epoch or best model
    model.load_state_dict(torch.load("best_segmentation_model.pth"))
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.cuda()
            masks = masks.cuda()
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(masks.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    from eval_metrics import get_confusion_matrix
    cm = get_confusion_matrix(all_preds, all_targets, num_classes)
    print("Confusion Matrix:\n", cm)

    # Plot IoU history
    plt.figure()
    plt.plot(range(1, epochs+1), val_iou_history, label="Val mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.title("Validation IoU per Epoch")
    plt.savefig("val_iou_curve.png", dpi=150)
    plt.close()

    # Return the best model path or the model instance
    return "best_segmentation_model.pth"
