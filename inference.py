import os
import cv2
import torch
import numpy as np
import pandas as pd

from model import get_segmentation_model
from dataset import get_val_transforms
from measure_capsulorhexis import measure_capsulorhexis_region

def inference_on_frames(
    frames_dir="/cataract-phase/temp/",
    output_dir="inference_results/",
    model_path="best_segmentation_model.pth",
    num_classes=4,  # e.g., 0=background,1=forceps,2=cystotome,3=rhexis
    forceps_class_id=1,  # whichever label was used for the known-size instrument
    rhexis_class_id=3,   # whichever label was used for capsulorhexis boundary
    known_forceps_size_mm=2.0,  # e.g. 2mm across the tip
    save_overlay=True
):
    """
    1) Loads a trained segmentation model
    2) Runs inference on frames in `frames_dir`
    3) Segments forceps (scale reference) and the capsulorhexis region
    4) Measures the rhexis diameter, circularity, and center offset in real mm
       using the known forceps tip size for mm-per-pixel conversion
    5) Saves a CSV with [frame_number, diameter_mm, circularity, center_offset_px]
    6) Optionally draws an overlay (contours) onto each frame
    """
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_segmentation_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Simple transforms: resize + normalize to match training
    transforms = get_val_transforms()

    results = []

    # Sort frame filenames by name so that "frame_000001.jpg" < "frame_000002.jpg", etc.
    frame_files = sorted([
        f for f in os.listdir(frames_dir) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        original_img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if original_img is None:
            continue

        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        aug = transforms(image=img_rgb)
        input_tensor = aug['image'].unsqueeze(0).to(device)  # shape (1,3,H,W)

        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1)  # shape (1,H,W)
        pred = pred.squeeze(0).cpu().numpy()  # shape (H,W) with class IDs

        # Undo any resizing for measurement if we want the original size
        # -- If we want pixel-accurate measurements, we should either:
        #  (A) not resize in inference transforms,
        #  (B) or carefully map predictions back to original size.
        # For demonstration, let's assume the transform in get_val_transforms()
        # only does a 512x512 resize. We'll upsample back:
        H_orig, W_orig = original_img.shape[:2]
        pred_upsampled = cv2.resize(pred.astype(np.uint8), (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

        # Forceps mask (to derive scale)
        forceps_mask = (pred_upsampled == forceps_class_id).astype(np.uint8)
        # Capsulorhexis mask
        rhexis_mask = (pred_upsampled == rhexis_class_id).astype(np.uint8)

        # Derive mm_per_pixel from the forceps:
        # For simplicity, measure bounding box width. Or find largest contour.
        forceps_contours, _ = cv2.findContours(forceps_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(forceps_contours) > 0:
            largest_forceps_contour = max(forceps_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_forceps_contour)
            forceps_width_px = max(w, h)  # approximate tip dimension
            if forceps_width_px > 0:
                mm_per_pixel = known_forceps_size_mm / float(forceps_width_px)
            else:
                mm_per_pixel = 0.0
        else:
            mm_per_pixel = 0.0

        # Measure capsulorhexis region
        from measure_capsulorhexis import measure_capsulorhexis_region
        diameter_px, circularity, center_offset_px = measure_capsulorhexis_region(rhexis_mask)

        # Convert diameter to mm
        diameter_mm = diameter_px * mm_per_pixel

        # Save result
        frame_number = os.path.splitext(frame_file)[0]
        results.append([
            frame_number, 
            diameter_mm, 
            circularity, 
            center_offset_px
        ])

        # Optionally draw overlay
        if save_overlay:
            overlay_img = original_img.copy()
            # Draw forceps contour in green
            for cnt in forceps_contours:
                cv2.drawContours(overlay_img, [cnt], -1, (0,255,0), 2)
            # Draw rhexis contour in red
            rhexis_contours, _ = cv2.findContours(rhexis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in rhexis_contours:
                cv2.drawContours(overlay_img, [cnt], -1, (0,0,255), 2)

            out_path = os.path.join(output_dir, f"{frame_number}_overlay.jpg")
            cv2.imwrite(out_path, overlay_img)

    # Write results to CSV
    df = pd.DataFrame(results, columns=["frame_number", "diameter_mm", "circularity", "center_offset_px"])
    df.to_csv(os.path.join(output_dir, "capsulorhexis_measurements.csv"), index=False)
    print(f"Saved inference results to {os.path.join(output_dir, 'capsulorhexis_measurements.csv')}")
