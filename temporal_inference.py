import os
import cv2
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from model import get_segmentation_model
from dataset import get_val_transforms
from measure_capsulorhexis import measure_capsulorhexis_region

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="deeplab")
    parser.add_argument("--encoder_name", type=str, default="resnet101")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="e.g. 0=BG,1=Forceps,2=Capsulorhexis or Cystotome. Adjust as needed.")
    parser.add_argument("--model_path", type=str, default="best_segmentation_model.pth")
    parser.add_argument("--frames_dir", type=str, default="/cataract-phase/temp/")
    parser.add_argument("--output_dir", type=str, default="temporal_inference_results/")
    parser.add_argument("--forceps_class_id", type=int, default=1,
                        help="Which label ID is the known-size forceps?")
    parser.add_argument("--rhexis_class_id", type=int, default=2,
                        help="Which label ID is the capsulorhexis boundary?")
    parser.add_argument("--known_forceps_size_mm", type=float, default=2.0,
                        help="Instrument tip real size in mm.")
    parser.add_argument("--save_overlay", action="store_true",
                        help="If set, saves color overlays for each frame + a legend.")
    parser.add_argument("--union_strategy", type=str, default="union",
                        help="How to combine rhexis masks: 'union' or 'intersection' or 'lastframe'.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load model
    model = get_segmentation_model(args.model_type, args.num_classes, args.encoder_name)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    transforms = get_val_transforms()

    # We'll accumulate a "running" capsulorhexis mask from all frames, 
    # so we can measure an overall boundary at the end.
    # If union => we keep all pixels ever predicted as rhexis.
    # If intersection => only those predicted in every frame.
    # lastframe => just measure the final frame's rhexis.
    accumulated_rhexis = None

    # We also collect instrument tip bounding boxes if we want to see movement,
    # but for final measurement we only need a scale factor from each frame’s biggest bounding box.
    # We'll do something simplistic here: store the average mm_per_pixel across frames
    # (only from frames that have a non-empty forceps).
    scales = []

    # For overlays, define color mapping for each class ID
    # Example:
    color_map = {
        1: (0,255,0),   # forceps => green BGR
        2: (0,0,255)    # rhexis => red BGR
    }

    frame_files = sorted([
        f for f in os.listdir(args.frames_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    for f in frame_files:
        frame_path = os.path.join(args.frames_dir, f)
        original_img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if original_img is None:
            continue

        H_orig, W_orig = original_img.shape[:2]
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        aug = transforms(image=img_rgb)
        input_tensor = aug["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Upsample back
        pred_upsampled = cv2.resize(
            pred.astype(np.uint8), (W_orig, H_orig),
            interpolation=cv2.INTER_NEAREST
        )

        forceps_mask = (pred_upsampled == args.forceps_class_id).astype(np.uint8)
        rhexis_mask = (pred_upsampled == args.rhexis_class_id).astype(np.uint8)

        # Accumulate rhexis
        if accumulated_rhexis is None:
            accumulated_rhexis = rhexis_mask.copy()
        else:
            if args.union_strategy == "union":
                accumulated_rhexis = cv2.bitwise_or(accumulated_rhexis, rhexis_mask)
            elif args.union_strategy == "intersection":
                accumulated_rhexis = cv2.bitwise_and(accumulated_rhexis, rhexis_mask)
            elif args.union_strategy == "lastframe":
                accumulated_rhexis = rhexis_mask.copy()
            else:
                # fallback to union
                accumulated_rhexis = cv2.bitwise_or(accumulated_rhexis, rhexis_mask)

        # measure forceps => scale
        forceps_contours, _ = cv2.findContours(forceps_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(forceps_contours)>0:
            largest_forceps = max(forceps_contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(largest_forceps)
            forceps_width_px = max(w,h)
            mm_per_pixel = args.known_forceps_size_mm / float(forceps_width_px) if forceps_width_px>0 else 0.0
            if mm_per_pixel>0:
                scales.append(mm_per_pixel)

        # Optionally create overlay for each frame
        if args.save_overlay:
            overlay = original_img.copy()
            # color each class region
            # We'll findContours for each class ID in color_map
            for class_id, color_bgr in color_map.items():
                mask_c = (pred_upsampled == class_id).astype(np.uint8)
                contours_c, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours_c:
                    cv2.drawContours(overlay, [cnt], -1, color_bgr, 2)

            # Save overlay
            out_path = os.path.join(args.output_dir, f"{os.path.splitext(f)[0]}_overlay.jpg")
            cv2.imwrite(out_path, overlay)

    # 2) Now measure final capsulorhexis from the "accumulated_rhexis" mask
    # We'll use the average mm_per_pixel from frames that had a valid forceps
    if accumulated_rhexis is None:
        print("No frames processed or no rhexis found. Exiting.")
        return

    if len(scales) > 0:
        final_scale = sum(scales)/len(scales)
    else:
        final_scale = 0.0

    from measure_capsulorhexis import measure_capsulorhexis_region
    diameter_px, circularity, offset_px = measure_capsulorhexis_region(accumulated_rhexis)
    diameter_mm = diameter_px * final_scale

    # 3) Create a single overlay that shows the final accumulated rhexis
    # (You can also show instrument paths if you track them, but we’ll just show rhexis in red.)
    H, W = accumulated_rhexis.shape[:2]
    final_overlay = np.zeros((H, W, 3), dtype=np.uint8)
    # color the union in red
    r_contours, _ = cv2.findContours(accumulated_rhexis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in r_contours:
        cv2.drawContours(final_overlay, [cnt], -1, (0,0,255), -1)

    # Optionally blend with a black background or something
    # We'll just keep it black + red for demonstration
    cv2.imwrite(os.path.join(args.output_dir, "accumulated_rhexis_overlay.jpg"), final_overlay)

    # 4) Create a small legend image if we used color_map
    # Example:
    legend_h = 100
    legend_w = 200
    legend_img = np.ones((legend_h, legend_w, 3), dtype=np.uint8)*255
    start_y = 10
    for class_id, color_bgr in color_map.items():
        label_str = f"Class {class_id}"
        cv2.rectangle(legend_img, (10,start_y), (30,start_y+20), color_bgr, -1)
        cv2.putText(legend_img, label_str, (40, start_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        start_y += 30
    cv2.imwrite(os.path.join(args.output_dir, "legend.jpg"), legend_img)

    # 5) Save final summary
    data = {
        "diameter_mm": [diameter_mm],
        "circularity": [circularity],
        "center_offset_px": [offset_px]
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(args.output_dir, "capsulorhexis_final_measurement.csv")
    df.to_csv(csv_path, index=False)
    print("Final measurement =>", df.to_dict(orient='records'))

if __name__ == "__main__":
    main()
