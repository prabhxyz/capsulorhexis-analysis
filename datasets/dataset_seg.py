import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CataractSegDataset(Dataset):
    """
    Minimal example for segmentation data. Expects:
      <root_dir>/videos/case_XXXX.mp4
      <root_dir>/Annotations/Coco-Annotations/case_XXXX/annotations/instances.json
    This example does not handle the entire COCO structure in detail. 
    Typically you'd have frames extracted or direct images. 
    Adjust as appropriate for your actual dataset structure.
    """
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # store (video_path, list_of_annotation_data)

        videos_dir = os.path.join(self.root_dir, "videos")
        ann_dir = os.path.join(self.root_dir, "Annotations", "Coco-Annotations")
        # We'll look for case_XXXX.mp4 and matching JSON
        all_videos = [v for v in os.listdir(videos_dir) if v.endswith(".mp4")]
        for vidname in all_videos:
            base = vidname.replace(".mp4", "")
            if not base.startswith("case_"):
                continue
            case_id = base
            json_dir = os.path.join(ann_dir, case_id, "annotations")
            instances_json = os.path.join(json_dir, "instances.json")
            if not os.path.isfile(instances_json):
                continue

            self.samples.append((os.path.join(videos_dir, vidname), instances_json))

        # In a real scenario, you'd parse the COCO annotations for each frame or 
        # you'd have actual extracted frames. For demonstration, we only store paths.

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        This is a STUB. Typically you'd have actual frame extraction and 
        produce (image_tensor, mask_tensor). We'll just produce dummy data
        for demonstration, as if we had a single frame per sample.
        """
        video_path, json_path = self.samples[idx]

        # read the first frame as a placeholder
        cap = cv2.VideoCapture(video_path)
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret or frame_bgr is None:
            frame_bgr = np.zeros((512,512,3), dtype=np.uint8)

        # create a dummy mask
        # In reality, you'd parse 'instances.json' to build a segmentation mask
        mask = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)

        # transform
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.transform:
            sample = self.transform(image=frame_rgb, mask=mask)
            image_tensor = sample["image"]
            mask_tensor  = sample["mask"].long()
        else:
            image_tensor = torch.from_numpy(frame_rgb.transpose(2,0,1)).float()/255.0
            mask_tensor  = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor
