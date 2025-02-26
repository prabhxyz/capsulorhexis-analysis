import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

INSTRUMENT_CLASSES = {
    "Background": 0,
    "Iris": 1,
    "Pupil": 2,
    "IntraocularLens": 3,
    "SlitKnife": 4,
    "Gauge": 5,
    "Spatula": 6,
    "CapsulorhexisCystotome": 7,
    "PhacoTip": 8,
    "IrrigationAspiration": 9,
    "LensInjector": 10,
    "CapsulorhexisForceps": 11,
    "KatanaForceps": 12
}

RELEVANT_CLASSES = {
    "CapsulorhexisCystotome": 7,
    "CapsulorhexisForceps": 11,
    "KatanaForceps": 12
}

class InvalidSegSampleError(Exception):
    """Custom exception for invalid segmentation samples."""
    pass

def create_seg_mask(height, width, annotations):
    """
    Converts polygon or bounding box annotations into an HxW mask of class IDs.
    Skip or ignore objects not in RELEVANT_CLASSES.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        class_name = ann["classTitle"]
        if class_name in RELEVANT_CLASSES:
            cid = RELEVANT_CLASSES[class_name]
            pts = ann["points"]["exterior"]
            if not pts:
                # skip empty polygons
                continue
            poly = np.array(pts, dtype=np.int32)
            if poly.shape[0] < 3:
                # not a valid polygon
                continue
            cv2.fillPoly(mask, [poly], color=cid)
    return mask

class CataractSegDataset(Dataset):
    """
    Dataset for segmentation. We skip samples if the image is missing or annotation is invalid.
    """

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        ann_dir = os.path.join(root_dir, "Annotations", "Coco-Annotations")
        video_dirs = glob.glob(os.path.join(ann_dir, "case_*"))
        for vdir in video_dirs:
            ann_file = os.path.join(vdir, "annotations", "instances.json")
            if not os.path.isfile(ann_file):
                continue
            with open(ann_file, "r") as f:
                data = json.load(f)

            images = data["images"]       # list of { "id", "file_name", "height", "width" }
            annotations = data["annotations"]   # COCO list
            categories = data["categories"]     # list of { "id", "name" }

            # Build cat-id -> name map
            cat_map = {}
            for c in categories:
                cat_map[c["id"]] = c["name"]

            # Build image_id -> list of annotation objects (converted to our format)
            imgid_to_anns = {}
            for ann in annotations:
                img_id = ann["image_id"]
                # In standard COCO, ann["segmentation"] might be polygons. 
                # We can parse them or just store bounding boxes.
                # For simplicity, pretend ann["bbox"] is a bounding box we convert to a polygon.
                x, y, w, h = ann["bbox"]
                poly_pts = [
                    [x, y],
                    [x+w, y],
                    [x+w, y+h],
                    [x, y+h]
                ]
                class_id = ann["category_id"]
                class_name = cat_map[class_id]

                obj_dict = {
                    "classTitle": class_name,
                    "points": {"exterior": poly_pts}
                }
                if img_id not in imgid_to_anns:
                    imgid_to_anns[img_id] = []
                imgid_to_anns[img_id].append(obj_dict)

            # Build the final sample list
            for img in images:
                img_id = img["id"]
                file_name = img["file_name"]
                h = img["height"]
                w = img["width"]
                frame_path = os.path.join(self.root_dir, "videos", file_name)
                # If there's no corresponding video frame, skip
                if not os.path.isfile(frame_path):
                    continue
                ann_list = imgid_to_anns.get(img_id, [])
                self.samples.append({
                    "img_path": frame_path,
                    "height": h,
                    "width": w,
                    "objects": ann_list
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        h = sample["height"]
        w = sample["width"]
        objects = sample["objects"]

        # Attempt to load image
        if not os.path.isfile(img_path):
            raise InvalidSegSampleError(f"Image file not found: {img_path}")
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil, dtype=np.uint8)
        if image_np.shape[0] != h or image_np.shape[1] != w:
            # Dimension mismatch
            raise InvalidSegSampleError(f"Image dims do not match annotation: {image_np.shape[:2]} vs {(h,w)}")

        # Create mask
        mask_np = create_seg_mask(h, w, objects)
        if mask_np is None or mask_np.size == 0:
            raise InvalidSegSampleError("Could not create mask")

        # Transform
        if self.transform:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].long()
        else:
            image_tensor = torch.from_numpy(image_np.transpose(2,0,1)).float()/255.
            mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor
