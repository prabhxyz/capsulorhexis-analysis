#!/usr/bin/env python
import argparse
import os
import cv2
import torch
import numpy as np
from collections import deque
import albumentations as A
from albumentations.pytorch import ToTensorV2

###########################################################
# Import The EXACT Phase Model Class
###########################################################
"""
This matches your definition:

class PhaseRecognitionNet(nn.Module):
    def __init__(self, num_phases=12, use_pretrained=True):
        ...
        self.base = backbone

    def forward(self, x):
        return self.base(x)
"""
from models.phase_recognition_model import PhaseRecognitionNet

###########################################################
# Import The Segmentation Model
###########################################################
from models.segmentation_model import LightweightSegModel

###########################################################
# Phase Recognition Utility
###########################################################
def sample_and_classify_frames(
    cap,
    phase_model,
    phase_transform,
    device,
    sample_interval=20,
    smoothing_window=15,
    target_phase_idx=2
):
    """
    Quickly scans the entire video by sampling every 'sample_interval' frames.
    Classifies each sampled frame with the single-frame model.
    Returns a dict:
      {
        "raw_classifications": [(frame_idx, pred_phase_idx), ...],
        "capsulorhexis_range": (start_frame, end_frame) or None
      }
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    raw_classifications = []
    recent_preds = deque(maxlen=smoothing_window)

    caps_start = None
    caps_end   = None

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            inp = phase_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            logits = phase_model(inp)
            pred_idx = torch.argmax(logits, dim=1).item()

        raw_classifications.append((frame_idx, pred_idx))
        recent_preds.append(pred_idx)

        # Majority logic: if 75% of recent == target => phase start
        if caps_start is None:
            count_target = sum(1 for x in recent_preds if x == target_phase_idx)
            if count_target >= 0.75 * len(recent_preds):
                caps_start = frame_idx
        else:
            # if started => see if ended
            count_target = sum(1 for x in recent_preds if x == target_phase_idx)
            if count_target < 0.25 * len(recent_preds):
                caps_end = frame_idx
                break

        frame_idx += sample_interval
        if frame_idx>=total_frames:
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if caps_start is not None and caps_end is None:
        caps_end = total_frames-1

    return {
        "raw_classifications": raw_classifications,
        "capsulorhexis_range": None if (caps_start is None) else (caps_start, caps_end)
    }

###########################################################
# Instrument Tracking Utilities
###########################################################
def polygons_from_mask(seg_mask):
    """
    Finds class-labeled contours from 'seg_mask'.
    Returns dict: {class_id: [contours, ...]}
    """
    polygons = {}
    unique_vals = np.unique(seg_mask)
    for cid in unique_vals:
        if cid == 0:
            continue
        mask_bin = (seg_mask==cid).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts)>0:
            polygons.setdefault(cid, []).extend(cnts)
    return polygons

def extract_instrument_tip(polygons_dict):
    """
    For each class's contours, find the boundingRect to measure size
    and pick the topmost point as the 'tip'. Returns (mean_bbox_height, tip_x, tip_y)
    or None if no instruments found.
    """
    best_area = 0
    tip = None
    heights = []
    for cid, cnts in polygons_dict.items():
        for c in cnts:
            area = cv2.contourArea(c)
            if area<1.0:
                continue
            x,y,w,h = cv2.boundingRect(c)
            heights.append(float(h))
            if area>best_area:
                best_area = area
                # topmost point as instrument tip
                min_y = np.min(c[:,:,1])
                candidates = c[c[:,:,1]==min_y]
                leftmost = candidates[np.argmin(candidates[:,:,0])]
                tip = (float(leftmost[0]), float(leftmost[1]))

    if tip is None or len(heights)==0:
        return None
    return (float(np.mean(heights)), tip[0], tip[1])

###########################################################
# Geometry: Circle-Fitting
###########################################################
def fit_circle(points):
    """
    Simple least-squares circle fit: returns (cx,cy,r).
    """
    pts = np.array(points, dtype=np.float32)
    if len(pts)<3:
        raise ValueError("Not enough points for circle fit.")

    x = pts[:,0]
    y = pts[:,1]
    A = np.column_stack([x,y,np.ones(len(pts))])
    b = x*x + y*y
    alpha,_,_,_ = np.linalg.lstsq(A,b,rcond=None)
    cx = 0.5*alpha[0]
    cy = 0.5*alpha[1]
    c  = alpha[2]
    r2 = (cx*cx + cy*cy) - c
    if r2<0:
        raise ValueError("Negative radius^2 => invalid circle.")
    r = np.sqrt(r2)
    return (cx, cy, r)

def build_mask(H, W, cx, cy, r):
    """
    Circle mask of shape (H,W).
    """
    mask = np.zeros((H,W), dtype=np.uint8)
    cv2.circle(mask, (int(cx),int(cy)), int(r), 1, thickness=-1)
    return mask

###########################################################
# Rhexis Classification
###########################################################
def classify_rhexis(diam_mm, circ, center_offset_mm):
    """
    Basic rule-based classification => "IDEAL" or "AT-RISK".
    """
    reasons = []
    if not (4.5<=diam_mm<=5.5):
        reasons.append(f"Diameter {diam_mm:.2f}mm not in [4.5,5.5].")
    if circ<0.8:
        reasons.append(f"Circularity {circ:.2f}<0.8.")
    if center_offset_mm>1.0:
        reasons.append(f"Offset {center_offset_mm:.2f}mm>1.0.")

    if len(reasons)==0:
        return ("IDEAL", reasons)
    else:
        return ("AT-RISK", reasons)

###########################################################
# MAIN
###########################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Cataract surgery video path")
    parser.add_argument("--phase_model_path", default="phase_recognition.pth")
    parser.add_argument("--seg_model_path", default="lightweight_seg.pth")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--caps_phase_idx", type=int, default=2, help="Phase idx for capsulorhexis")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    #####################################################
    # 1) Load Phase Model - EXACT as your definition
    #####################################################
    phase_model = PhaseRecognitionNet(num_phases=12, use_pretrained=False).to(device)
    # We use strict=False in case your checkpoint has naming differences
    phase_ckpt = torch.load(args.phase_model_path, map_location=device)
    phase_model.load_state_dict(phase_ckpt, strict=False)
    phase_model.eval()

    phase_transform = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    # 2) Find capsulorhexis range
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {args.video_path}")
        return

    info = sample_and_classify_frames(
        cap,
        phase_model,
        phase_transform,
        device,
        sample_interval=20,
        smoothing_window=15,
        target_phase_idx=args.caps_phase_idx
    )
    cap.release()

    c_range = info["capsulorhexis_range"]
    if not c_range:
        print("No capsulorhexis phase found => defaulting to entire video or skipping.")
        c_range = (0, None)
    start_f, end_f = c_range

    # 3) Load Seg Model
    seg_model = LightweightSegModel(num_classes=4, use_pretrained=False, aux_loss=True).to(device)
    seg_ckpt = torch.load(args.seg_model_path, map_location=device)
    seg_model.load_state_dict(seg_ckpt, strict=False)
    seg_model.eval()

    seg_transform = A.Compose([
        A.Resize(512,512),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    cap2 = cv2.VideoCapture(args.video_path)
    total_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_f is None or end_f>=total_frames:
        end_f = total_frames-1
    print(f"Capsulorhexis frames: {start_f} => {end_f}")

    # For demonstration, skip frames for speed:
    skip_frames = 5
    W = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tip_coords = []
    box_heights = []

    fidx = start_f
    while fidx<=end_f:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame_bgr = cap2.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            inp = seg_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            seg_out = seg_model(inp)["out"]
            seg_pred = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()
        # polygons
        polys = polygons_from_mask(seg_pred)
        inst_info = extract_instrument_tip(polys)
        if inst_info is not None:
            mean_h, tx, ty = inst_info
            box_heights.append(mean_h)
            # scale up from 512x512 => original
            scale_x = W/512.0
            scale_y = H/512.0
            tip_coords.append((tx*scale_x, ty*scale_y))

        fidx+=skip_frames

    cap2.release()
    print(f"Collected {len(tip_coords)} tip coords across capsulorhexis frames.")
    if len(tip_coords)<3:
        print("Not enough points => no circle => at-risk by default.")
        return

    # 4) Fit circle
    try:
        cx, cy, r = fit_circle(tip_coords)
    except ValueError:
        print("Circle fit error => at-risk.")
        return
    circle_mask = build_mask(H, W, cx, cy, r)
    cv2.imwrite("capsulorhexis_mask.png", circle_mask*255)

    # 5) Convert bounding box heights to mm
    # Suppose the instrument's bounding box height is ~2 mm in reality
    if len(box_heights)==0:
        mm_per_px = 0.02
    else:
        avg_h = np.mean(box_heights)
        real_mm_instrument = 2.0
        mm_per_px = real_mm_instrument / avg_h

    # 6) Shape analysis from circle_mask
    c_cnt, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c_cnt:
        print("No contour => at-risk.")
        return
    cnt = c_cnt[0]
    area_px = cv2.contourArea(cnt)
    perimeter_px = cv2.arcLength(cnt, True)
    x,y,w,h = cv2.boundingRect(cnt)
    diam_px = (w+h)/2.0
    diam_mm = diam_px*mm_per_px
    if perimeter_px>0:
        circ = 4.0*np.pi*(area_px/(perimeter_px**2))
    else:
        circ = 0.0
    # center offset from image center
    M = cv2.moments(cnt)
    if M["m00"]>0:
        rx = M["m10"]/M["m00"]
        ry = M["m01"]/M["m00"]
    else:
        rx, ry = 0,0
    off_x = rx-(W/2.0)
    off_y = ry-(H/2.0)
    off_px = np.sqrt(off_x**2 + off_y**2)
    off_mm = off_px*mm_per_px

    # 7) Classification
    label, reasons = classify_rhexis(diam_mm, circ, off_mm)
    if label=="IDEAL":
        print("Final Rhexis: IDEAL.")
        print(f" => diameter: {diam_mm:.2f} mm, circularity: {circ:.2f}, offset: {off_mm:.2f} mm")
    else:
        print("Final Rhexis: AT-RISK.")
        for r_ in reasons:
            print("  -", r_)

    print("Done. Check 'capsulorhexis_mask.png' for the final boundary mask.")

if __name__=="__main__":
    main()