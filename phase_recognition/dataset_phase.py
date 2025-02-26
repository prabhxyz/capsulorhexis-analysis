import os
import cv2
import glob
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

class InvalidSubclipError(Exception):
    """Custom exception for invalid subclip ranges."""
    pass

class PhaseRecognitionDataset(Dataset):
    """
    A dataset for loading short subclips from videos where
    the annotation CSV can have start_frame/end_frame that may be out of range.
    We skip those invalid subclips.
    """

    def __init__(self, root_dir, clip_length=16, transform=None):
        """
        root_dir: path to the dataset, e.g. "Cataract-1k-Phase"
        File structure:
          Cataract-1k-Phase/
            videos/
              case_XXXX.mp4
            annotations/
              case_XXXX/
                case_XXXX_annotations_phases.csv

        clip_length: number of frames per clip.
        transform: optional albumentations or other transforms for each frame.
        """
        super().__init__()
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform

        self.video_dir = os.path.join(root_dir, "videos")
        self.annotation_dir = os.path.join(root_dir, "annotations")

        self.samples = []
        self.video_frame_counts = {}  # memo: video_path -> frame_count

        # Collect all mp4s named "case_XXXX.mp4"
        mp4_paths = glob.glob(os.path.join(self.video_dir, "case_*.mp4"))

        for mp4_path in mp4_paths:
            case_basename = os.path.basename(mp4_path)   # e.g. "case_0123.mp4"
            case_id = os.path.splitext(case_basename)[0] # e.g. "case_0123"

            # CSV path => annotations/case_0123/case_0123_annotations_phases.csv
            csv_folder = os.path.join(self.annotation_dir, case_id)
            csv_name = f"{case_id}_annotations_phases.csv"
            csv_path = os.path.join(csv_folder, csv_name)
            if not os.path.isfile(csv_path):
                # If missing CSV, skip
                continue

            # Get total frames in the video
            total_frames = self._get_video_frame_count(mp4_path)

            df = pd.read_csv(csv_path)
            # For each row with "comment", "frame", "endFrame", generate subclips
            for _, row in df.iterrows():
                phase_label = row["comment"]
                start_f = int(row["frame"])
                end_f = int(row["endFrame"])
                # If out of range entirely, skip
                if start_f >= total_frames:
                    continue
                if end_f > total_frames:
                    end_f = total_frames

                # For each subclip (length=clip_length)
                current_f = start_f
                while current_f + self.clip_length <= end_f:
                    self.samples.append({
                        "video_path": mp4_path,
                        "start_frame": current_f,
                        "end_frame": current_f + self.clip_length,
                        "phase_label": phase_label
                    })
                    current_f += self.clip_length

        # Build label->index
        all_phases = sorted(list(set([s["phase_label"] for s in self.samples])))
        self.phase_to_idx = {phase: idx for idx, phase in enumerate(all_phases)}

        print(f"Initialized PhaseRecognitionDataset from '{root_dir}'")
        print(f"Found {len(mp4_paths)} video(s) under 'videos/'")
        print(f"Total dataset samples: {len(self.samples)}")

    def _get_video_frame_count(self, path):
        """Memoize the frame count for each video path."""
        if path not in self.video_frame_counts:
            cap = cv2.VideoCapture(path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.video_frame_counts[path] = length
        return self.video_frame_counts[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        video_path = sample_info["video_path"]
        start_frame = sample_info["start_frame"]
        end_frame = sample_info["end_frame"]
        phase_label = sample_info["phase_label"]

        # Read frames from video
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

        # Safety check
        if len(frames) < (end_frame - start_frame):
            # It's possible we read fewer frames than expected if the video is truncated
            raise InvalidSubclipError(
                f"Expected {end_frame - start_frame} frames but got {len(frames)}."
            )
        if len(frames) == 0:
            raise InvalidSubclipError("No frames read for this subclip.")

        # Stack frames into (T, H, W, 3)
        frames_np = np.stack(frames, axis=0)

        # Apply transform if available
        if self.transform is not None:
            import torch
            transformed_list = []
            for frame_img in frames_np:
                out = self.transform(image=frame_img)["image"]  # shape: (C,H,W)
                transformed_list.append(out.unsqueeze(0))      # add batch dim
            clip_tensor = torch.cat(transformed_list, dim=0)     # (T,C,H,W)
        else:
            import torch
            clip_tensor = torch.from_numpy(frames_np).permute(0,3,1,2).float()/255.

        label_idx = self.phase_to_idx[phase_label]
        label_tensor = torch.tensor(label_idx).long()
        return clip_tensor, label_tensor
