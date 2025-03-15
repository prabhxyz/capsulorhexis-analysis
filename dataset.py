import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CataractSuperviselyAllDataset(Dataset):
    """
    Reads Supervisely JSON for all classes found in the annotation,
    assigning each unique classTitle an integer label >0 (0=background).
    
    Directory:
      Cataract-1k-Seg/
        Annotations/
          Images-and-Supervisely-Annotations/
            case_XXXX/
              ├─ img/
              └─ ann/
    """

    def __init__(self, root_dir, case_ids, transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.case_ids = case_ids
        self.transforms = transforms
        self.data_items = []

        sup_dir = os.path.join(self.root_dir, "Annotations", "Images-and-Supervisely-Annotations")

        for case_name in self.case_ids:
            case_folder = os.path.join(sup_dir, case_name)
            img_dir = os.path.join(case_folder, "img")
            ann_dir = os.path.join(case_folder, "ann")
            if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
                continue

            image_files = sorted(
                f for f in os.listdir(img_dir)
                if f.lower().endswith(".png")
            )
            for img_file in image_files:
                ann_file = img_file + ".json"
                img_path = os.path.join(img_dir, img_file)
                ann_path = os.path.join(ann_dir, ann_file)
                if os.path.isfile(img_path) and os.path.isfile(ann_path):
                    self.data_items.append((img_path, ann_path))

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        img_path, ann_path = self.data_items[idx]

        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with open(ann_path, "r") as f:
            data = json.load(f)

        height = data["size"]["height"]
        width = data["size"]["width"]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Build cat2label from all classes in 'objects'
        objects = data.get("objects", [])
        class_titles = []
        for obj in objects:
            title = obj["classTitle"]
            if title not in class_titles:
                class_titles.append(title)
        class_titles = sorted(list(set(class_titles)))
        cat2label = {}
        for i, t in enumerate(class_titles, start=1):
            cat2label[t] = i

        # fill polygons
        for obj in objects:
            title = obj["classTitle"]
            label_val = cat2label[title]
            polygon_points = obj["points"]["exterior"]
            pts = np.array(polygon_points, dtype=np.int32).reshape((-1,1,2))
            cv2.fillPoly(mask, [pts], color=label_val)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(
            size=(512,512),
            scale=(0.7,1.0),
            ratio=(0.8,1.2),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(),
        ToTensorV2()
    ])
