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

from ..models.phase_recognition_model import PhaseRecognitionNet
from datasets.dataset_phase import PhaseRecognitionDataset

def get_phase_transforms():
    return A.Compose([
        A.Resize(224,224),
        A.RandomBrightnessContrast(p=0.4),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def validate_phase(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, memory_format=torch.channels_last)
            labels = labels.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                outputs = model(frames)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def train_phase(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Phase Epoch {epoch}/{epochs}")
        for frames, labels in pbar:
            frames = frames.to(device, memory_format=torch.channels_last)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                outputs = model(frames)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = validate_phase(model, val_loader, device, criterion)
        scheduler.step(epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # Overwrite checkpoint each epoch
        torch.save(model.state_dict(), "phase_recognition_current.pth")

    # Plot training curves
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Phase Recognition Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, epochs+1), train_accs, label='Train Acc')
    plt.plot(range(1, epochs+1), val_accs,   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Phase Recognition Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("phase_training_curves.png")
    print("Saved phase_training_curves.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory for the Phase dataset.")
    parser.add_argument("--phase_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    cudnn.benchmark = True

    transform = get_phase_transforms()
    dataset = PhaseRecognitionDataset(root_dir=args.root_dir, transform=transform)
    ds_len = len(dataset)
    print(f"Phase dataset length = {ds_len}")
    if ds_len == 0 or args.phase_epochs == 0:
        print("No data or 0 epochs => skipping phase training.")
        return

    val_size = int(0.2 * ds_len)
    train_size = ds_len - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Get number of phases from dataset's label map
    num_phases = len(dataset.phase_label_map)
    model = PhaseRecognitionNet(num_phases=num_phases, use_pretrained=True).to(device, memory_format=torch.channels_last)

    print("Beginning Phase Model Training...")
    train_phase(model, train_loader, val_loader, device, epochs=args.phase_epochs, lr=args.lr)

    torch.save(model.state_dict(), "phase_recognition_final.pth")
    print("Saved phase_recognition_final.pth")

if __name__ == "__main__":
    main()