import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CataractSuperviselyDataset(Dataset):
    """
    Reads data from:
      Cataract-1k-Seg/Annotations/Images-and-Supervisely-Annotations/case_XXXX/
        ├─ img/
        │   ├─ caseXXXX_01.png
        │   └─ ...
        └─ ann/
            ├─ caseXXXX_01.png.json
            └─ ...
    
    Each JSON file (e.g. 'caseXXXX_01.png.json') describes polygons in
    data["objects"], each with:
        "classTitle": <str>,
        "points": {"exterior": [[x1,y1],[x2,y2],...], "interior":[]}

    We'll fill a segmentation mask for the relevant classes only
    (e.g. "Capsulorhexis Forceps", "Capsulorhexis Cystotome", etc.),
    assigning each relevant class a unique integer label (1..N).
    0 => background.
    """

    def __init__(
        self,
        root_dir,           # path to top-level "Cataract-1k-Seg/"
        case_ids,           # which cases (e.g. ["case_5013","case_1234"]) to include
        transforms=None,
        relevant_class_names=None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.case_ids = case_ids
        self.transforms = transforms

        if relevant_class_names is None:
            # Example default
            self.relevant_class_names = [
                "capsulorhexis forceps",
                "capsulorhexis cystotome"
            ]
        else:
            self.relevant_class_names = [x.lower() for x in relevant_class_names]

        self.data_items = []  # will store (img_path, ann_path)

        sup_anno_dir = os.path.join(self.root_dir, "Annotations", "Images-and-Supervisely-Annotations")

        for case_name in self.case_ids:
            case_folder = os.path.join(sup_anno_dir, case_name)
            img_dir = os.path.join(case_folder, "img")
            ann_dir = os.path.join(case_folder, "ann")
            if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
                continue

            # For each .png in 'img', look for the corresponding .png.json in 'ann'
            image_files = sorted(
                f for f in os.listdir(img_dir)
                if f.lower().endswith(".png")
            )
            for img_file in image_files:
                ann_file = img_file + ".json"  # e.g. "case5013_01.png.json"
                img_path = os.path.join(img_dir, img_file)
                ann_path = os.path.join(ann_dir, ann_file)
                if os.path.isfile(img_path) and os.path.isfile(ann_path):
                    self.data_items.append((img_path, ann_path))

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        img_path, ann_path = self.data_items[idx]

        # Read image
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Read JSON annotation
        with open(ann_path, "r") as f:
            data = json.load(f)

        height = data["size"]["height"]
        width = data["size"]["width"]

        # If the actual image shape differs from (height,width) declared in JSON,
        # we might want to resize or just trust the actual image. We'll assume they match.
        # Prepare the segmentation mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Build a dict: classTitle => label index
        # In principle, we can dynamically do this for all classes in the JSON, 
        # but let's only keep relevant ones:
        # We'll keep them in alphabetical order or just in the order we see them:
        # For example, if we have 2 relevant classes, 1 => first class, 2 => second, etc.
        # Actually, let's do a simpler approach:
        #    cat2label = {} 
        #    next_label = 1
        #    for each relevant object => if classTitle not in cat2label => cat2label[classTitle] = next_label
        # but the user might want a stable mapping. We'll do this dynamically:
        cat2label = {}
        next_label = 1

        objects = data.get("objects", [])
        for obj in objects:
            classTitle = obj["classTitle"].lower()
            if classTitle not in self.relevant_class_names:
                continue

            if classTitle not in cat2label:
                cat2label[classTitle] = next_label
                next_label += 1

        # Now fill polygons
        for obj in objects:
            classTitle = obj["classTitle"].lower()
            if classTitle in cat2label:
                label_val = cat2label[classTitle]
                polygon_points = obj["points"]["exterior"]  # list of [x,y] coords
                # Fill polygon onto mask
                # We must convert to a numpy int array shaped (num_points,1,2) for cv2.fillPoly
                pts = np.array(polygon_points, dtype=np.int32).reshape((-1,1,2))
                cv2.fillPoly(mask, [pts], color=label_val)

        # Apply transforms
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(
            size=(512, 512),  # If older Albumentations, you can do "height=512,width=512" if that works
            scale=(0.8,1.0),
            ratio=(0.9,1.1),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.GaussianBlur(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(),
        ToTensorV2()
    ])
