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

# Suppose you have a dataset file 'cataract_seg_dataset.py' with CataractSegDataset
from cataract_seg_dataset import CataractSegDataset, NUM_SEG_CLASSES

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

def validate_segmentation(model, val_loader, device, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            output = model(imgs)["out"]
            loss = criterion(output, masks)
            losses.append(loss.item())
    return sum(losses) / len(losses)

def train_segmentation(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_loader)
        val_loss = validate_segmentation(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Plot training curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Segmentation Training Loss')
    plt.legend()
    plt.savefig('seg_training_loss.png')
    plt.close()

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
    seg_len = len(seg_dataset)
    if seg_len == 0:
        print("No segmentation data found. Exiting.")
        return

    val_size = int(0.2 * seg_len)
    train_size = seg_len - val_size
    seg_train, seg_val = random_split(seg_dataset, [train_size, val_size])

    train_loader = DataLoader(seg_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(seg_val,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = LightweightSegModel(num_classes=NUM_SEG_CLASSES, use_pretrained=True, aux_loss=True).to(device)

    print("Training Segmentation Model...")
    train_segmentation(model, train_loader, val_loader, device,
                       epochs=args.epochs, lr=args.lr)

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "lightweight_seg.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved segmentation model to {save_path}")
    print("Training curve saved to 'seg_training_loss.png'")

if __name__ == "__main__":
    main()