import os
import glob
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# -------------------------
# Dataset Definition
# -------------------------
class PhaseRecognitionDataset(Dataset):
    """
    A dataset that loads short clips from cataract-surgery videos, along with phase labels.
    Expects structure:
      <root_dir>/videos/case_XXXX.mp4
      <root_dir>/annotations/case_XXXX/case_XXXX_annotations_phases.csv
    Each row of the CSV has "comment" (phase), "frame", and "endFrame".
    """

    def __init__(self, root_dir, clip_length=16, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform

        self.video_dir = os.path.join(root_dir, "videos")
        self.annotation_dir = os.path.join(root_dir, "annotations")

        self.samples = []

        # Gather all "case_XXXX.mp4"
        mp4_paths = glob.glob(os.path.join(self.video_dir, "case_*.mp4"))

        for mp4_path in mp4_paths:
            basename = os.path.basename(mp4_path)              # e.g. "case_0123.mp4"
            case_id = os.path.splitext(basename)[0]            # e.g. "case_0123"

            # CSV expected at: <root_dir>/annotations/case_0123/case_0123_annotations_phases.csv
            csv_folder = os.path.join(self.annotation_dir, case_id)
            csv_name = f"{case_id}_annotations_phases.csv"
            csv_path = os.path.join(csv_folder, csv_name)
            if not os.path.isfile(csv_path):
                continue

            # Load CSV
            df = pd.read_csv(csv_path)
            # Each row might have "comment" (phase), "frame", "endFrame"
            cap = cv2.VideoCapture(mp4_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for _, row in df.iterrows():
                phase_label = row["comment"]
                start_f = int(row["frame"])
                end_f = int(row["endFrame"])
                # If end_f beyond total_frames, clamp
                if start_f >= total_frames:
                    continue
                if end_f > total_frames:
                    end_f = total_frames

                # Generate subclips of length clip_length
                current_f = start_f
                while current_f + self.clip_length <= end_f:
                    self.samples.append({
                        "video_path": mp4_path,
                        "start_frame": current_f,
                        "end_frame": current_f + self.clip_length,
                        "phase_label": phase_label
                    })
                    current_f += self.clip_length

        # Build phase -> idx map
        all_phases = sorted(list(set([s["phase_label"] for s in self.samples])))
        self.phase_to_idx = {phase: idx for idx, phase in enumerate(all_phases)}

        print(f"Initialized PhaseRecognitionDataset from '{root_dir}'")
        print(f"Found {len(mp4_paths)} video(s) under 'videos/'")
        print(f"Total dataset samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        video_path = sample_info["video_path"]
        start_frame = sample_info["start_frame"]
        end_frame = sample_info["end_frame"]
        phase_label = sample_info["phase_label"]

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index >= end_frame:
                break
            if start_frame <= frame_index < end_frame:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_index += 1
        cap.release()

        # frames should have (end_frame - start_frame) frames
        if len(frames) == 0:
            # If for some reason the annotation is out of range, this might happen
            # You can raise an exception or skip. Here we'll just raise an exception:
            raise ValueError(f"No frames read from {video_path} for subclip [{start_frame},{end_frame})")

        # Stack into (T, H, W, 3)
        frames_np = np.stack(frames, axis=0)

        if self.transform is not None:
            transformed_list = []
            for frame_img in frames_np:
                out = self.transform(image=frame_img)["image"]  # shape: (C,H,W)
                transformed_list.append(out.unsqueeze(0))      # add batch dimension
            clip_tensor = torch.cat(transformed_list, dim=0)     # shape: (T,C,H,W)
        else:
            clip_tensor = torch.from_numpy(frames_np).permute(0,3,1,2).float()/255.

        label_idx = self.phase_to_idx[phase_label]
        label_tensor = torch.tensor(label_idx).long()
        return clip_tensor, label_tensor


# -------------------------
# Model Definition
# -------------------------
import torch.nn.functional as F
from torchvision.models.video import swin3d_t

class PhaseRecognitionModel(nn.Module):
    """
    A simple 3D Swin Transformer for phase classification.
    """
    def __init__(self, num_phases):
        super().__init__()
        self.backbone = swin3d_t(weights=None)
        in_feats = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_feats, num_phases)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        # But swin3d_t expects (B, C, T, H, W), so permute:
        x = x.permute(0, 2, 1, 3, 4)
        logits = self.backbone(x)
        return logits


# -------------------------
# Utility Functions
# -------------------------
def save_training_plot(train_losses, val_losses, out_path="phase_training_loss.png"):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Phase Recognition Loss')
    plt.legend()
    plt.savefig(out_path)
    plt.close()

def accuracy_score(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().sum()
    return correct / labels.shape[0]


# -------------------------
# Training Script (Main)
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory for the phase dataset, e.g. Cataract-1k-Phase.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    import random
    import torch
    from tqdm import tqdm

    # 1) Create transforms for each frame (resize, normalize, etc.)
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    # 2) Build dataset
    dataset = PhaseRecognitionDataset(root_dir=args.root_dir, clip_length=16, transform=transform)

    # 3) Train/Val split
    val_ratio = 0.2
    val_size = int(len(dataset)*val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # 4) DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 5) Model
    num_phases = len(dataset.phase_to_idx)
    model = PhaseRecognitionModel(num_phases=num_phases)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 6) Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 7) Training Loop
    train_losses, val_losses = [], []
    for epoch in range(args.epochs):
        # -------------------------
        # Train
        # -------------------------
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] [Train]", ncols=100)
        for clips, labels in train_pbar:
            clips = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy_score(logits, labels).item()

            total_train_loss += loss.item() * clips.size(0)
            total_train_acc  += acc * clips.size(0)

            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc":  f"{acc:.4f}"
            })

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_train_acc  = total_train_acc  / len(train_loader.dataset)

        # -------------------------
        # Validate
        # -------------------------
        model.eval()
        total_val_loss = 0.0
        total_val_acc  = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] [Val]  ", ncols=100)
        with torch.no_grad():
            for clips, labels in val_pbar:
                clips = clips.to(device)
                labels = labels.to(device)

                logits = model(clips)
                loss = criterion(logits, labels)

                acc = accuracy_score(logits, labels).item()

                total_val_loss += loss.item() * clips.size(0)
                total_val_acc  += acc * clips.size(0)

                val_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc":  f"{acc:.4f}"
                })

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_acc  = total_val_acc  / len(val_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"\n=== Epoch [{epoch+1}/{args.epochs}] Summary ===")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {avg_val_acc:.4f}\n")

    # 8) Save final model
    os.makedirs("models/phase_recognition", exist_ok=True)
    torch.save(model.state_dict(), "models/phase_recognition/phase_recognition_swin3d.pth")

    # 9) Plot training curves
    save_training_plot(train_losses, val_losses, out_path="phase_training_loss.png")
    print("Training complete. Model saved to 'models/phase_recognition/phase_recognition_swin3d.pth'")
    print("Training plot saved to 'phase_training_loss.png'.")


if __name__ == "__main__":
    main()
