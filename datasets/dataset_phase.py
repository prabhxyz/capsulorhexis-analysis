import os
import cv2
import csv
import torch
import numpy as np
from torch.utils.data import Dataset

class PhaseRecognitionDataset(Dataset):
    """
    Dataset for phase recognition that parses CSV files with start/end frames for each phase.
    Expects:
      <root_dir>/videos/case_XXXX.mp4
      <root_dir>/annotations/case_XXXX/case_XXXX_annotations_phases.csv
    """
    def __init__(self, root_dir, transform=None, frame_skip=10):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.frame_skip = frame_skip

        self.samples = []  # (video_path, frame_idx, phase_name)
        self.phase_label_map = {}
        self._scan_data()

    def _scan_data(self):
        # for each "case_XXXX" in videos, find matching CSV in annotations
        videos_dir = os.path.join(self.root_dir, "videos")
        annotations_dir = os.path.join(self.root_dir, "annotations")

        all_videos = [v for v in os.listdir(videos_dir) if v.endswith(".mp4")]
        # build a set of "case_XXXX" from these
        cases = []
        for vid in all_videos:
            base = vid.replace(".mp4", "")  # e.g. case_4687
            if base.startswith("case_"):
                cases.append(base)

        # read each CSV
        all_phase_names = set()
        for case_name in cases:
            csv_path = os.path.join(annotations_dir, case_name, case_name + "_annotations_phases.csv")
            vid_path = os.path.join(videos_dir, case_name + ".mp4")
            if not os.path.isfile(csv_path) or not os.path.isfile(vid_path):
                continue

            # open video to get total frames
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                cap.release()
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    phase_name = row["comment"]
                    start_f = int(row["frame"])
                    end_f   = int(row["endFrame"])
                    # store
                    all_phase_names.add(phase_name)
                    # create samples
                    # skip frames by self.frame_skip
                    for fr in range(start_f, min(end_f+1, total_frames), self.frame_skip):
                        self.samples.append((vid_path, fr, phase_name))

        # create phase_label_map from sorted list
        sorted_phases = sorted(list(all_phase_names))
        self.phase_label_map = {ph: i for i, ph in enumerate(sorted_phases)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_idx, phase_name = self.samples[idx]
        phase_label = self.phase_label_map[phase_name]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret or frame_bgr is None:
            # fallback: blank
            frame_bgr = np.zeros((224,224,3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.transform:
            sample = self.transform(image=frame_rgb)
            frame_tensor = sample["image"]
        else:
            frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float() / 255.0

        return frame_tensor, phase_label
