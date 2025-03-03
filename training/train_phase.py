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
# Use the old style for PyTorch <2.0
from torch.cuda.amp import autocast, GradScaler

from datasets.dataset_phase import PhaseRecognitionDataset
from models.phase_recognition_model import PhaseRecognitionNet

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

def get_phase_transforms():
    """
    Returns advanced data augmentations for phase recognition.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.RandomBrightnessContrast(p=0.4),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def compute_accuracy(model, loader, device):
    """
    Computes accuracy on the given loader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

def main_phase_train(root_dir, epochs=10, batch_size=8, lr=1e-4):
    """
    Trains a single-frame phase recognition model (MobileNetV3-based).
    Saves a plot of loss & accuracy.
    Overwrites model each epoch for simplicity.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PhaseTraining] Using device={device}")

    # Load dataset
    transform = get_phase_transforms()
    dataset = PhaseRecognitionDataset(root_dir=root_dir, transform=transform)
    dataset_len = len(dataset)
    print(f"[PhaseTraining] Found {dataset_len} samples in '{root_dir}'")
    if dataset_len == 0:
        print("[PhaseTraining] No data => skipping.")
        return

    val_size = int(0.2 * dataset_len)
    train_size = dataset_len - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=1)

    # Initialize model
    num_phases = len(dataset.phase_label_map)  # dynamic phases
    model = PhaseRecognitionNet(num_phases=num_phases, use_pretrained=True).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    # Remove 'cuda' from constructor
    scaler = GradScaler(enabled=(device.type=="cuda"))

    # For plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Phase Epoch {epoch}/{epochs}")
        for frames, labels in pbar:
            frames = frames.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # Remove 'device_type' param for compatibility
            with autocast(enabled=(device.type=="cuda")):
                outputs = model(frames)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # End of epoch => compute metrics
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_acc = compute_accuracy(model, train_loader, device)
        train_accs.append(train_acc)

        val_running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                outs = model(frames)
                val_loss = criterion(outs, labels)
                val_running_loss += val_loss.item()

        val_loss_avg = val_running_loss / len(val_loader)
        val_losses.append(val_loss_avg)
        val_acc = compute_accuracy(model, val_loader, device)
        val_accs.append(val_acc)

        print(f"[PhaseTraining] Epoch {epoch}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss_avg:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        # Overwrite model each epoch
        torch.save(model.state_dict(), "phase_recognition.pth")

    # Save training curve
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses,   label='Val Loss')
    plt.title("Phase - Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs,   label='Val Acc')
    plt.title("Phase - Accuracy")
    plt.legend()

    plt.savefig("phase_training_curve.png")
    plt.close()

    print("[PhaseTraining] Done. Model => 'phase_recognition.pth', plot => 'phase_training_curve.png'")

if __name__ == "__main__":
    # Example usage if you run train_phase.py standalone:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="Cataract-1k-Phase")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    main_phase_train(
        root_dir=args.root_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )