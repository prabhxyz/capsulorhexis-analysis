import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

class CataractCocoDataset(Dataset):
    """
    Custom dataset to read images and segmentation annotations (COCO format),
    filtering only the relevant classes needed for the capsulorhexis
    segmentation task (e.g., 'capsulorhexis forceps', 'cystotome',
    and the 'capsulorhexis boundary' if annotated).
    
    Assumes the user has extracted frames from each case_XXXX.mp4,
    or can read directly from videos. For simplicity, we show reading
    from frames that exist on disk plus the corresponding COCO JSON.
    """

    def __init__(
        self, 
        image_dir, 
        annotation_json, 
        transforms=None,
        relevant_class_names=None
    ):
        """
        Args:
            image_dir (str): Path to directory containing images/frames.
            annotation_json (str): Path to the COCO annotation file (instances.json).
            transforms (albumentations.Compose): Augmentations/transforms to apply.
            relevant_class_names (list of str): The class names we keep for training 
                (e.g. ['capsulorhexis forceps','capsulorhexis cystotome','capsulorhexis boundary']).
        """
        super().__init__()
        self.image_dir = image_dir
        self.coco = COCO(annotation_json)
        self.transforms = transforms

        # If the user wants to focus on certain classes, define them here:
        if relevant_class_names is None:
            self.relevant_class_names = [
                "capsulorhexis forceps",
                "capsulorhexis cystotome",
                "capsulorhexis boundary"
            ]
        else:
            self.relevant_class_names = relevant_class_names

        # Map COCO category_id --> our contiguous label index
        # e.g. 0 = background, 1 = forceps, 2 = cystotome, 3 = boundary, etc.
        # In principle, you can set these in any order you like:
        self.cat2label = {}
        self.label2cat = {}
        label_idx = 1  # Start from 1 for foreground classes, 0 is background
        for cat in self.coco.loadCats(self.coco.getCatIds()):
            cat_name = cat["name"].lower()
            if cat_name in self.relevant_class_names:
                self.cat2label[cat["id"]] = label_idx
                self.label2cat[label_idx] = cat["id"]
                label_idx += 1

        # Gather all image IDs from the COCO file
        self.img_ids = self.coco.getImgIds()

        # Filter out images that do not have at least one relevant category
        valid_img_ids = []
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(img_id)
            anns = self.coco.loadAnns(ann_ids)
            # Check if any annotation belongs to relevant classes:
            keep = False
            for ann in anns:
                if ann["category_id"] in self.cat2label:
                    keep = True
                    break
            if keep:
                valid_img_ids.append(img_id)

        self.img_ids = valid_img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        filename = img_info["file_name"]
        img_path = os.path.join(self.image_dir, filename)
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create segmentation mask
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Initialize single-channel label mask with zeros:
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Fill in each relevant annotation
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id in self.cat2label:
                label_val = self.cat2label[cat_id]
                # Use the pycocotools to create mask
                ann_mask = self.coco.annToMask(ann)  # 0/1
                mask[ann_mask == 1] = label_val

        # Optional Albumentations transforms (including random augmentations, resizing, etc.)
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

def get_train_transforms():
    """
    Example of advanced Albumentations transforms for training.
    """
    return A.Compose([
        A.RandomResizedCrop(height=512, width=512, scale=(0.8,1.0), ratio=(0.9,1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.GaussianBlur(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])

def get_val_transforms():
    """
    For validation, often just resize and normalize.
    """
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(),
        ToTensorV2()
    ])
