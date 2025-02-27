import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from PIL import Image

# Only these 3 classes are relevant; background=0
# We'll assign them: 1, 2, 3
RELEVANT_CLASSES = {
    "Capsulorhexis Forceps": 1,
    "Capsulorhexis Cystotome": 2,
    "Katena Forceps": 3
}

NUM_SEG_CLASSES = 4  # 0=background, plus 3 relevant

def parse_case_and_frame(file_name):
    """
    file_name: e.g. "case5353_01.png"
    returns (case_id_str, frame_idx_int)
    Example: ("5353", 1)
    """
    # Remove extension
    core = file_name.replace(".png", "")  # "case5353_01"
    # or if your file_name is .jpg, adapt accordingly

    # Split on underscore: "case5353" and "01"
    # or you might do a regex if the naming is more complex
    parts = core.split("_")
    # parts[0] = "case5353", parts[1] = "01"
    case_part = parts[0]  # "case5353"
    frame_part = parts[1]  # "01"
    # extract the digits from "case5353" -> "5353"
    case_id_str = case_part.replace("case", "")  # "5353"
    # parse frame index
    frame_idx = int(frame_part)
    return case_id_str, frame_idx

def create_seg_mask(height, width, annotations, cat_map):
    """
    Create an HxW mask for only RELEVANT_CLASSES (1..3).
    Background=0.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        cat_id = ann["category_id"]
        cat_name = cat_map.get(cat_id, None)
        if cat_name in RELEVANT_CLASSES:
            cls_id = RELEVANT_CLASSES[cat_name]  # 1..3

            # If we have polygon segmentation
            if "segmentation" in ann and ann["segmentation"]:
                for seg in ann["segmentation"]:
                    pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], color=cls_id)
            else:
                # fallback to bbox
                x, y, w, h = ann["bbox"]
                x1, y1 = int(x), int(y)
                x2, y2 = x1 + int(w), y1 + int(h)
                cv2.rectangle(mask, (x1,y1), (x2,y2), color=cls_id, thickness=-1)

    return mask

class CataractSegDataset(Dataset):
    """
    Loads frames from .mp4 videos based on the file_name in the COCO 'images' list.
    The file_name is something like "case5353_01.png", from which we parse:
      - case ID = 5353
      - frame index = 1
    Then we read that frame from videos/case_5353.mp4 at frame index=0-based.
    We create a segmentation mask from the relevant classes.
    """

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        ann_dir = os.path.join(root_dir, "Annotations", "Coco-Annotations")
        if not os.path.isdir(ann_dir):
            print(f"[Warning] No 'Coco-Annotations' folder at {ann_dir}")
            return

        case_dirs = glob.glob(os.path.join(ann_dir, "case_*"))
        for cdir in case_dirs:
            ann_folder = os.path.join(cdir, "annotations")
            json_path = os.path.join(ann_folder, "instances.json")
            if not os.path.isfile(json_path):
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            images = data["images"]       # list of dict
            annotations = data["annotations"]
            categories = data["categories"]

            # Build cat_id -> cat_name
            cat_map = {}
            for cinfo in categories:
                cat_map[cinfo["id"]] = cinfo["name"]

            # Group ann by image_id
            imgid_to_anns = {}
            for ann in annotations:
                iid = ann["image_id"]
                if iid not in imgid_to_anns:
                    imgid_to_anns[iid] = []
                imgid_to_anns[iid].append(ann)

            # Build final list
            for img_info in images:
                img_id = img_info["id"]
                file_name = img_info["file_name"]
                H = img_info["height"]
                W = img_info["width"]
                ann_list = imgid_to_anns.get(img_id, [])

                # parse case ID and frame index
                case_id_str, frame_idx = parse_case_and_frame(file_name)
                # e.g. "5353", 1

                # check if mp4 exists
                mp4_name = f"case_{case_id_str}.mp4"
                mp4_path = os.path.join(root_dir, "videos", mp4_name)
                if not os.path.isfile(mp4_path):
                    # If no video file, skip
                    continue

                self.samples.append({
                    "mp4_path": mp4_path,
                    "frame_idx": frame_idx,
                    "height": H,
                    "width": W,
                    "annotations": ann_list,
                    "cat_map": cat_map
                })

        print(f"Found {len(self.samples)} samples total in {self.__class__.__name__}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mp4_path = sample["mp4_path"]
        frame_idx = sample["frame_idx"]  # 1 => we want 0-based => frame_idx-1
        H = sample["height"]
        W = sample["width"]
        annotations = sample["annotations"]
        cat_map = sample["cat_map"]

        # Build mask
        mask_np = create_seg_mask(H, W, annotations, cat_map)

        # Open mp4, read the correct frame
        cap = cv2.VideoCapture(mp4_path)
        # If your frame_idx in JSON is 1-based, we do .set(..., frame_idx-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret or frame_bgr is None:
            # Fallback: black image
            frame_bgr = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            # If the actual read frame dims differ from (H,W), consider resizing or cropping
            # We'll assume the video is the same dims as the annotation. If not, rectify here.
            if frame_bgr.shape[0] != H or frame_bgr.shape[1] != W:
                # Let's just resize to (W,H) for consistency
                frame_bgr = cv2.resize(frame_bgr, (W,H), interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=frame_rgb, mask=mask_np)
            image_tensor = augmented["image"]
            mask_tensor  = augmented["mask"].long()
        else:
            import torch
            image_tensor = torch.from_numpy(frame_rgb.transpose(2,0,1)).float()/255.
            mask_tensor  = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor
