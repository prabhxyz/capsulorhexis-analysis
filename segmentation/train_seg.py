import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from segmentation.dataset_seg import CataractSegDataset, RELEVANT_CLASSES
from segmentation.model_seg import get_mask_rcnn_model
from segmentation.utils_seg import collate_fn, create_target_from_mask, save_training_plot_seg, collate_seg_skip_invalid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory for the segmentation dataset (e.g. Cataract-1k-Seg).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = CataractSegDataset(root_dir=args.root_dir, transform=transform)

    val_ratio = 0.2
    val_size = int(len(dataset)*val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, collate_fn=collate_seg_skip_invalid)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, collate_fn=collate_seg_skip_invalid)

    # Number of relevant classes + 1 (background)
    # E.g. if we have 3 relevant instruments, then we have 4 total (including background)
    unique_cids = list(RELEVANT_CLASSES.values())
    num_classes = len(unique_cids) + 1  # background is 0
    model = get_mask_rcnn_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images = [img.to(device) for img in images]
            targets = create_target_from_mask(masks, device)
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            running_loss += losses.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images = [img.to(device) for img in images]
                targets = create_target_from_mask(masks, device)
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_running_loss += losses.item()

        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    os.makedirs("models/segmentation", exist_ok=True)
    torch.save(model.state_dict(), "models/segmentation/mask_rcnn_cataract.pth")

    save_training_plot_seg(train_losses, val_losses, out_path="seg_training_loss.png")

if __name__ == "__main__":
    main()
